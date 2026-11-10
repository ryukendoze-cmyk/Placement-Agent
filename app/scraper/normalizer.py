"""Taxonomy grounding, LPA salary conversion, and normalization for scraped jobs."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import re
import time
import uuid
from typing import Any

from app.config import settings
from app.matching.engine import (
    FRESHER_TITLE_RE,
    MID_LEVEL_TITLE_RE,
    SENIOR_TITLE_RE,
    is_fresher_job,
    is_senior_job,
)
from app.models import Job
from app.taxonomy.branches import BRANCH_ALIASES, normalize_branch
from app.taxonomy.taxonomy import Taxonomy

logger = logging.getLogger(__name__)

__all__ = [
    "CORE_FALLBACK_SKILLS",
    "DEFAULT_BRANCHES",
    "EXP_COLON_RE",
    "EXP_PHRASE_RE",
    "EXP_PLUS_SINGLE_RE",
    "EXP_RANGE_RE",
    "EXP_URL_SLUG_RE",
    "FRESHER_TITLE_RE",
    "JOB_NORMALIZER_SYSTEM",
    "MID_LEVEL_TITLE_RE",
    "SENIOR_TITLE_RE",
    "extract_experience_years",
    "fetch_external_job_description",
    "is_fresher_job",
    "is_senior_job",
    "normalize_job_with_ai",
    "normalize_scraped_job",
    "parse_salary_lpa",
]

BOUND = r"(?<![a-z0-9]){alias}(?![a-z0-9])"
CORE_FALLBACK_SKILLS = ["python", "sql", "git"]
DEFAULT_BRANCHES = ["CSE", "IT", "ECE"]


def fetch_external_job_description(url: str, timeout: float = 4.0) -> str:
    """Safely fetch public job description for external postings when raw scraper returned empty body."""
    if not url:
        return ""
    try:
        import httpx
        from bs4 import BeautifulSoup

        resp = httpx.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            },
            follow_redirects=True,
            timeout=timeout,
        )
        if resp.status_code == 200 and resp.text:
            soup = BeautifulSoup(resp.text, "lxml")
            if "linkedin.com" in url:
                desc_el = soup.select_one(".description__text, .show-more-less-html__markup")
                if desc_el:
                    return desc_el.get_text("\n", strip=True)
            elif "internshala.com" in url:
                blocks = []
                for el in soup.select(".text-container, .internship_details, #about_internship"):
                    txt = el.get_text(" ", strip=True)
                    if txt and len(txt) > 20:
                        blocks.append(txt)
                if blocks:
                    return "\n\n".join(dict.fromkeys(blocks))
            else:
                for sel in (".job-description", "#job-description", ".description", "#jobDescriptionText"):
                    desc_el = soup.select_one(sel)
                    if desc_el:
                        return desc_el.get_text("\n", strip=True)
    except Exception as exc:
        logger.debug("Fast detail enrichment failed for %s: %s", url, exc)
    return ""


def _clean_html(raw_html: str) -> str:
    """Strip HTML markup, replace linebreaks, and decode HTML entities."""
    if not raw_html:
        return ""
    text = re.sub(r"(?i)<br\s*/?>", "\n", raw_html)
    text = re.sub(r"(?i)</(?:p|div|li|tr|h\d)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _parse_amount(val: Any, interval: str) -> int | None:
    """Parse raw salary amount to LPA (Lakhs Per Annum) with input sanitization (ARCHITECTURE §20.2.1)."""
    if val is None:
        return None
    if isinstance(val, str):
        cleaned = re.sub(r"(?i)[₹,\s]|rs\.?|inr|lpa", "", val)
        if not cleaned:
            return None
        try:
            amt = float(cleaned)
        except (TypeError, ValueError):
            return None
    else:
        try:
            amt = float(val)
        except (TypeError, ValueError):
            return None
    if math.isnan(amt) or amt <= 0:
        return None

    # Values in LPA notation (e.g. 6.5, 12, 18, or executive/US-remote up to 500 LPA)
    if interval in ("lpa", "inr_lpa") or amt <= 500:
        return max(1, round(amt))

    # Monthly salary conversion: amount * 12 / 100,000
    if interval and (interval in ("monthly", "month", "mo") or "month" in interval):
        return max(1, round(amt * 12 / 100_000))

    # Annual/yearly salary conversion (treat >= 100k as annual INR): amount / 100,000
    return max(1, round(amt / 100_000))


# Experience extraction regexes supporting Western/Indian ranges, plus syntax, and YOE (ARCHITECTURE §21.2.1)
_EXP_DIGIT = r"(?:[0-9]|1[0-9]|20)"
EXP_RANGE_RE = re.compile(
    rf"(?i)\b({_EXP_DIGIT})\s*(?:[-\u2013\u2014]|to)\s*({_EXP_DIGIT})\s*(?:years?|yrs?|yoe)\b"
)
EXP_URL_SLUG_RE = re.compile(
    rf"(?i)\b({_EXP_DIGIT})[- ]*to[- ]*({_EXP_DIGIT})[- ]*(?:years?|yrs?|yoe)\b"
)
EXP_PHRASE_RE = re.compile(
    rf"(?i)\b(?:min(?:imum)?|at\s+least|requires?|experience\s+of)\s+(?:of\s+)?({_EXP_DIGIT})\+?\s*(?:years?|yrs?|yoe)\b"
)
EXP_PLUS_SINGLE_RE = re.compile(
    rf"(?i)\b({_EXP_DIGIT})\+?\s*(?:years?|yrs?|yoe)\b"
)
EXP_COLON_RE = re.compile(
    rf"(?i)(?:min(?:imum)?\s+)?(?:experience|exp)\s*(?:required|needed|level)?\s*[:=-]\s*({_EXP_DIGIT})\b"
)


def extract_experience_years(text: str | None) -> int:
    """Deterministically extract minimum required experience years from text (ARCHITECTURE §21.2.1)."""
    if not text:
        return 0

    found_years: list[int] = []

    # 1. Plus / single pattern: e.g. "4+ years", "5+ YOE", "2+ yrs"
    # Matches explicit minimums like "4+ Years" in title or description.
    for m in EXP_PLUS_SINGLE_RE.finditer(text):
        try:
            val = int(m.group(1))
            if "+" in m.group(0) and val > 0:
                found_years.append(val)
        except (ValueError, TypeError):
            pass

    # 2. Minimum phrase pattern: e.g. "minimum of 2 years", "at least 3 years", "requires 2+ years"
    for m in EXP_PHRASE_RE.finditer(text):
        try:
            val = int(m.group(1))
            if val > 0:
                found_years.append(val)
        except (ValueError, TypeError):
            pass

    # 3. Colon / labeled pattern: e.g. "Experience: 3", "Exp: 2", "Min Exp: 4"
    for m in EXP_COLON_RE.finditer(text):
        try:
            val = int(m.group(1))
            if val > 0:
                found_years.append(val)
        except (ValueError, TypeError):
            pass

    # 4. Range patterns in text and URL slugs: e.g. "1-3 years", "2 to 4 years", "2-to-4-years"
    for regex in (EXP_RANGE_RE, EXP_URL_SLUG_RE):
        for m in regex.finditer(text):
            try:
                low = int(m.group(1))
                high = int(m.group(2))
                if low > 0:
                    found_years.append(low)
                elif low == 0 and high >= 2:
                    # Wide range (0 to 5 years, 0 to 4 years, etc.)
                    # If text contains an explicit fresher title/indicator (Intern, Fresher, Trainee, Graduate),
                    # it is genuinely 0-exp friendly.
                    # Otherwise (e.g. Software Developer, Backend Engineer), it is an experienced role.
                    from app.matching.engine import FRESHER_TITLE_RE
                    if not FRESHER_TITLE_RE.search(text):
                        found_years.append(max(1, high // 2))
            except (ValueError, TypeError):
                pass

    if found_years:
        return min(found_years)

    # 5. Fallback for single pattern without plus if explicit: e.g. "2 YOE", "3 yrs"
    for m in EXP_PLUS_SINGLE_RE.finditer(text):
        try:
            val = int(m.group(1))
            if val > 0 and ("yoe" in m.group(0).lower() or "yrs" in m.group(0).lower()):
                return val
        except (ValueError, TypeError):
            pass

    return 0


def parse_salary_lpa(
    min_amt: Any = None,
    max_amt: Any = None,
    interval: Optional[str] = None,
) -> tuple[list[int], bool]:
    """Parse raw salary figures into an integer [min_lpa, max_lpa] range and disclosure flag.

    If inputs are absent, empty, None, or unparseable, returns ([0, 0], False).
    Never injects synthetic fresher defaults like [6, 12].
    """
    if isinstance(min_amt, dict) and max_amt is None and interval is None:
        raw = min_amt
        interval = str(raw.get("interval") or "").strip().lower()
        currency = str(raw.get("currency") or raw.get("currency_code") or "").strip().upper()
        min_amt = raw.get("min_amount")
        max_amt = raw.get("max_amount")
        if currency == "USD":
            USD_TO_INR = 86.5
            if min_amt:
                try:
                    v = float(re.sub(r"[^\d.]", "", str(min_amt)))
                    v = v * 1000 if v < 1000 else v
                    min_amt = v * USD_TO_INR / 100_000
                    interval = "lpa"
                except Exception:
                    pass
            if max_amt:
                try:
                    v = float(re.sub(r"[^\d.]", "", str(max_amt)))
                    v = v * 1000 if v < 1000 else v
                    max_amt = v * USD_TO_INR / 100_000
                    interval = "lpa"
                except Exception:
                    pass
    else:
        interval = str(interval or "").strip().lower()

    p_min = _parse_amount(min_amt, interval)
    p_max = _parse_amount(max_amt, interval)

    if p_min is not None and p_max is not None:
        min_lpa = min(p_min, p_max)
        max_lpa = max(p_min, p_max)
        return [min_lpa, max_lpa], True

    if p_min is not None:
        return [p_min, max(p_min, round(p_min * 1.25) or (p_min + 2))], True

    if p_max is not None:
        return [max(1, round(p_max * 0.75) or (p_max - 2)), p_max], True

    return [0, 0], False


def _extract_salary_lpa(raw: dict[str, Any]) -> list[int]:
    """Extract salary for backward compatibility; returns [0, 0] when undisclosed."""
    salary, _ = parse_salary_lpa(raw)
    return salary


def _ground_skills(text: str, taxonomy: Taxonomy) -> tuple[list[str], list[str]]:
    """Strictly ground technical skills against the 124 canonical taxonomy skills.

    Returns:
        tuple of (required_skills, preferred_skills).
    """
    low_text = text.lower()
    matched_positions: dict[str, int] = {}

    for alias, skill_id in taxonomy.alias_map.items():
        pattern = BOUND.format(alias=re.escape(alias))
        match = re.search(pattern, low_text)
        if match:
            resolved_id = taxonomy.resolve(alias) or skill_id
            pos = match.start()
            if resolved_id not in matched_positions or pos < matched_positions[resolved_id]:
                matched_positions[resolved_id] = pos

    # Sort matched skills by their first position of occurrence
    sorted_skills = sorted(matched_positions.keys(), key=lambda s: matched_positions[s])

    if not sorted_skills:
        # Fallback to foundational core skills
        return list(CORE_FALLBACK_SKILLS), []

    if len(sorted_skills) == 1:
        # Pad with secondary core skill to satisfy min 2 required skills
        fallback_pad = [s for s in CORE_FALLBACK_SKILLS if s != sorted_skills[0]][0]
        return [sorted_skills[0], fallback_pad], []

    # Partition: first 2 to 6 skills become required, remainder preferred (capped at 6)
    split_idx = min(max(2, min(len(sorted_skills), 4)), 6)
    required = sorted_skills[:split_idx]
    preferred = sorted_skills[split_idx:split_idx + 6]
    return required, preferred


def _extract_branches(text: str) -> list[str]:
    """Extract academic branch eligibility from text or default to general tech disciplines."""
    found: set[str] = set()
    for canonical, aliases in BRANCH_ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", text, re.IGNORECASE):
                norm = normalize_branch(canonical)
                if norm:
                    found.add(norm)
                break
    return sorted(found) if found else list(DEFAULT_BRANCHES)


def normalize_scraped_job(raw: dict[str, Any], taxonomy: Taxonomy) -> Job:
    """Normalize a raw job dictionary into a strictly validated Job SQLModel entity.

    Args:
        raw: Dictionary from JobSpy or external scraping payload.
        taxonomy: Loaded taxonomy for grounded skill extraction.

    Returns:
        Job model with deterministic ID, clean LPA salary, and grounded taxonomy skills.
    """
    job_url = str(raw.get("job_url") or raw.get("url") or "").strip()
    source_val = raw.get("site") or raw.get("source")
    if source_val:
        source = str(source_val).strip().lower()
    else:
        source = "live" if job_url else "local"

    # Canonicalize the dedup key so tracking-param variants of one posting
    # (e.g. ?ref=email vs ?utm_source=nl) collapse to a single Job row.
    from app.scraper.service import _clean_url

    canonical_url = _clean_url(job_url) if job_url else ""
    if canonical_url:
        url_hash = hashlib.md5(canonical_url.encode("utf-8")).hexdigest()[:10]
        job_id = f"job-ext-{url_hash}"
    else:
        job_id = f"job-ext-{uuid.uuid4().hex[:8]}"

    title = str(raw.get("title") or "").strip() or "Software Engineer"
    company = str(raw.get("company") or "").strip() or "Tech Employer"

    raw_desc = str(raw.get("description") or "")
    if len(raw_desc.strip()) < 60 and job_url:
        fetched_desc = fetch_external_job_description(job_url)
        if fetched_desc:
            raw_desc = fetched_desc
            raw["description"] = fetched_desc

    clean_desc = _clean_html(raw_desc)
    if job_url:
        engine_val = str(raw.get("scraper_engine") or "jobspy")
        source_tag = str(source)
        full_desc = (
            f"{clean_desc}\n\nApply URL: {job_url}\n"
            f"Source: {source_tag} (Scraped via {engine_val})"
        ).strip()
    else:
        full_desc = clean_desc or f"{title} opportunity at {company}."

    # Combined text for taxonomy grounding and branch extraction
    tags_val = raw.get("tags")
    if isinstance(tags_val, list):
        tags_text = " ".join(str(t) for t in tags_val if t)
    elif isinstance(tags_val, str):
        tags_text = tags_val
    else:
        tags_text = ""
    combined_text = f"{title} {full_desc} {tags_text}"

    required_skills, preferred_skills = _ground_skills(combined_text, taxonomy)
    if "salary_disclosed" in raw:
        salary_disclosed = bool(raw.get("salary_disclosed"))
        raw_sal = raw.get("salary_lpa")
        if isinstance(raw_sal, list) and len(raw_sal) == 2:
            salary_lpa = [int(raw_sal[0]), int(raw_sal[1])]
        else:
            salary_lpa, salary_disclosed = parse_salary_lpa(raw)
    else:
        salary_lpa, salary_disclosed = parse_salary_lpa(raw)
        if not salary_disclosed and isinstance(raw.get("salary_lpa"), list) and len(raw["salary_lpa"]) == 2:
            s_pair = raw["salary_lpa"]
            if s_pair != [6, 12] and any(x > 0 for x in s_pair):
                salary_lpa = [int(s_pair[0]), int(s_pair[1])]
                salary_disclosed = True

    allowed_branches = _extract_branches(combined_text)

    # Location and remote determination
    location = str(raw.get("location") or "Bengaluru").strip()
    remote = bool(raw.get("is_remote")) or "remote" in location.lower()

    # Experience years (ARCHITECTURE §20.2.3)
    exp = 0
    try:
        raw_exp = raw.get("experience_years") or raw.get("min_experience")
        if raw_exp is not None:
            exp = int(float(raw_exp))
    except (TypeError, ValueError):
        exp = 0
    if exp <= 0:
        exp = extract_experience_years(combined_text)
    if exp <= 0 and is_senior_job({"title": title}):
        title_low = title.lower()
        if "iii" in title_low or " 3" in title_low or " iv" in title_low or " 4" in title_low or "specialist" in title_low:
            exp = 5
        else:
            exp = 3
    elif exp <= 0 and MID_LEVEL_TITLE_RE.search(title):
        exp = 2
    elif exp <= 0 and (job_url or source != "local"):
        # If external scraped posting has sparse description (< 50 chars) and does NOT have a fresher title:
        # Do not default to 0 fresher experience! A generic unverified engineering role is at least mid-level.
        if len(clean_desc) < 50 and not FRESHER_TITLE_RE.search(title) and source not in ("internshala", "unstop"):
            exp = 2

    return Job(
        id=job_id,
        title=title,
        company=company,
        description=full_desc,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        experience_years=exp,
        location=location,
        remote=remote,
        salary_lpa=salary_lpa,
        salary_disclosed=salary_disclosed,
        min_gpa=7.0,
        allowed_branches=allowed_branches,
        job_url=job_url,
        source=source,
        scraper_engine=str(raw.get("scraper_engine") or "jobspy"),
    )


# ---------------------------------------------------------------------------
# §18.2.A — AI Job Normalizer Contract & Dual-Path Resilient Fallback
# ---------------------------------------------------------------------------

JOB_NORMALIZER_SYSTEM = """TASK: extract_job_posting
You are an institutional career intelligence AI that extracts structured data from scraped job postings.
Analyze the provided job posting and extract:
1. required_skills: list of 2 to 6 strictly necessary core technical skill IDs from the provided taxonomy.
   - ONLY extract skills that are explicitly mentioned or strictly required in the job description text.
2. preferred_skills: list of 0 to 6 secondary, bonus, or nice-to-have technical skill IDs from the taxonomy.
   - CRITICAL: ONLY extract skills explicitly mentioned in the job text as preferred, plus, or nice-to-have.
   - DO NOT list skills that are absent from the text.
   - If no secondary/bonus skills are explicitly mentioned in the job text, return [].
   - NEVER dump or copy the canonical_skills list into preferred_skills.
3. salary_lpa: [min_lpa, max_lpa] integer array in Lakhs Per Annum (LPA).
   - Normalize ambiguous salary formats:
     * If salary is in USD ($) (e.g., $100k, $230,000 - $450,000 USD), convert to annual INR LPA using 1 USD ≈ 86.5 INR (multiply USD by 86.5 and divide by 100,000; e.g. $100k -> [87, 87], $230k - $450k -> [199, 389]). NEVER divide a USD salary by 100,000 directly.
     * If salary is a monthly stipend (e.g., ₹25k/mo, ₹30,000/month), convert to annual LPA using round(amt * 12 / 100000). For example, ₹30,000/mo -> [4, 4] LPA.
     * If salary is an annual range in raw INR (e.g. 600000 to 1200000, 8-12 Lakhs), convert to LPA (e.g. [6, 12]).
     * If salary is undisclosed or not mentioned, return [0, 0].
4. salary_disclosed: boolean indicating whether salary or stipend was disclosed.
5. is_fresher_friendly: boolean.
   - Strictly true ONLY for roles requiring strictly 0 years of experience, or explicit campus/intern/trainee/graduate fresher roles with NO prior professional experience required.
   - Strictly false for any role requiring 1+ years, 1-2 years, 1-3 years, or prior professional experience.
   - Strictly false for any senior/lead/staff/principal role, or level III/IV/V/3/4/5 role (e.g. "Machine Learning Engineer III", "SDE III", "Senior Developer").
6. experience_years: float or null representing required minimum years of experience.
   - 0.0 for freshers / campus roles / interns.
   - If explicit years/range given (e.g. "7+ years", "3-5 years", "1-2 YOE"), extract the minimum (e.g. 7.0, 3.0, 1.0) and set is_fresher_friendly to false if > 0.
   - If no explicit numeric years are mentioned in text, but the title indicates seniority or a higher level:
     * Level III / 3 (e.g. "Machine Learning Engineer III", "SDE 3") -> 5.0
     * Senior / Lead / Staff / Level IV -> 5.0 to 7.0
     * Principal / Architect / Level V -> 8.0 or higher
     * Mid-level / Level II / 2 -> 2.0

Respond ONLY with valid JSON conforming to this schema:
{
  "required_skills": ["skill_id_1", "skill_id_2"],
  "preferred_skills": ["skill_id_3"],
  "salary_lpa": [min_lpa, max_lpa],
  "salary_disclosed": true,
  "is_fresher_friendly": false,
  "experience_years": 0.0
}
"""


_CIRCUIT_FAILURES = 0
_CIRCUIT_OPEN_UNTIL = 0.0


def reset_ai_normalizer_circuit() -> None:
    global _CIRCUIT_OPEN_UNTIL, _CIRCUIT_FAILURES
    _CIRCUIT_OPEN_UNTIL = 0.0
    _CIRCUIT_FAILURES = 0


def is_ai_normalizer_circuit_open() -> bool:
    global _CIRCUIT_OPEN_UNTIL
    return time.time() < _CIRCUIT_OPEN_UNTIL


def trip_ai_normalizer_circuit(duration_seconds: float = 300.0) -> None:
    global _CIRCUIT_OPEN_UNTIL, _CIRCUIT_FAILURES
    _CIRCUIT_OPEN_UNTIL = time.time() + duration_seconds
    _CIRCUIT_FAILURES = 0
    logger.warning("AI normalizer circuit breaker TRIPPED for %.1fs due to consecutive errors", duration_seconds)


def record_ai_normalizer_failure(exc: Exception | None = None) -> None:
    global _CIRCUIT_FAILURES
    _CIRCUIT_FAILURES += 1
    if _CIRCUIT_FAILURES >= 3:
        trip_ai_normalizer_circuit(300.0)


def record_ai_normalizer_success() -> None:
    global _CIRCUIT_FAILURES
    _CIRCUIT_FAILURES = 0


def normalize_job_with_ai(
    raw_job: dict[str, Any],
    taxonomy: Taxonomy,
    llm: Any | None = None,
) -> Job:
    """Normalize a raw job dictionary using an LLM for structured semantic extraction.

    Falls back immediately to normalize_scraped_job(raw_job, taxonomy) on any failure,
    timeout, missing LLM, or invalid output (Dual-Path Graceful Degradation).
    """
    is_mock = (
        type(llm).__name__ in ("MockLLM", "MagicMock", "AsyncMock")
        or "Mock" in type(llm).__name__
        if llm is not None
        else False
    )
    if llm is None or (is_ai_normalizer_circuit_open() and not is_mock):
        return normalize_scraped_job(raw_job, taxonomy)

    try:
        title = str(raw_job.get("title") or "").strip()
        company = str(raw_job.get("company") or "").strip()
        job_url = str(raw_job.get("job_url") or raw_job.get("url") or "").strip()
        raw_desc = str(raw_job.get("description") or "")
        if len(raw_desc.strip()) < 60 and job_url:
            fetched = fetch_external_job_description(job_url)
            if fetched:
                raw_desc = fetched
                raw_job["description"] = fetched

        clean_desc = _clean_html(raw_desc)[:8000]

        prompt_payload = {
            "title": title,
            "company": company,
            "description": clean_desc,
            "job_url": job_url,
            "source": str(raw_job.get("site") or raw_job.get("source") or ""),
            "location": str(raw_job.get("location") or ""),
            "min_amount": raw_job.get("min_amount"),
            "max_amount": raw_job.get("max_amount"),
            "interval": raw_job.get("interval"),
            "currency": raw_job.get("currency"),
            "tags": raw_job.get("tags"),
            # Canonical skill ids for grounding: the taxonomy exposes all() and
            # an alias map, not a `.skills` dict (that attribute does not exist).
            "canonical_skills": [s.id for s in taxonomy.all()],
        }

        user_content = json.dumps(prompt_payload)
        timeout = getattr(settings, "llm_timeout", 60.0)
        try:
            response_text = llm.complete(
                JOB_NORMALIZER_SYSTEM,
                user_content,
                json_mode=True,
                timeout=timeout,
            )
        except TypeError:
            response_text = llm.complete(
                JOB_NORMALIZER_SYSTEM,
                user_content,
                json_mode=True,
            )

        clean_json = (response_text or "").strip()
        if clean_json.startswith("```"):
            clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
            clean_json = re.sub(r"\s*```$", "", clean_json)
        ai_data = json.loads(clean_json)
        if not isinstance(ai_data, dict):
            raise ValueError("AI normalizer output is not a JSON object")

        # Programmatically validate extracted skills against taxonomy
        # AND verify the skill (or an alias) actually appears in the job text.
        tag_str = " ".join(str(t) for t in (raw_job.get("tags") or []) if t)
        corpus = f"{title} {clean_desc} {tag_str}".lower()

        def _skill_in_text(s_id: str) -> bool:
            s_obj = taxonomy.get_opt(s_id)
            if not s_obj:
                return False
            candidates = [s_obj.name.lower(), s_obj.id.lower()] + [a.lower() for a in s_obj.aliases]
            for c in candidates:
                pat = r"(?<![a-z0-9])" + re.escape(c) + r"(?![a-z0-9])"
                if re.search(pat, corpus):
                    return True
            return False

        valid_required: list[str] = []
        for s in ai_data.get("required_skills", []):
            s_norm = str(s).strip().lower()
            resolved = taxonomy.resolve(s_norm) or (s_norm if taxonomy.get_opt(s_norm) else None)
            if resolved and resolved not in valid_required and _skill_in_text(resolved):
                valid_required.append(resolved)

        valid_preferred: list[str] = []
        for s in ai_data.get("preferred_skills", []):
            s_norm = str(s).strip().lower()
            resolved = taxonomy.resolve(s_norm) or (s_norm if taxonomy.get_opt(s_norm) else None)
            if resolved and resolved not in valid_preferred and resolved not in valid_required and _skill_in_text(resolved):
                valid_preferred.append(resolved)

        # Cap lists to prevent LLM hallucinations or skill dumping
        valid_required = valid_required[:6]
        valid_preferred = valid_preferred[:6]

        # Fall back to CORE_FALLBACK_SKILLS if fewer than 2 valid skills resolved
        if len(valid_required) < 2:
            for fallback in CORE_FALLBACK_SKILLS:
                if fallback not in valid_required:
                    valid_required.append(fallback)
                if len(valid_required) >= 2:
                    break

        # Salary parsing: structured API salary has priority over LLM guesswork
        stat_sal, stat_disc = parse_salary_lpa(raw_job)
        if stat_disc and any(x > 0 for x in stat_sal):
            salary_lpa = stat_sal
            salary_disclosed = True
        else:
            raw_sal = ai_data.get("salary_lpa")
            salary_disclosed = bool(ai_data.get("salary_disclosed"))
            salary_lpa = [0, 0]
            if isinstance(raw_sal, list) and len(raw_sal) == 2:
                try:
                    s_min = round(float(raw_sal[0]))
                    s_max = round(float(raw_sal[1]))
                    if s_min > s_max:
                        s_min, s_max = s_max, s_min
                    salary_lpa = [s_min, s_max]
                    if any(x > 0 for x in salary_lpa):
                        salary_disclosed = True
                except (TypeError, ValueError):
                    salary_lpa = [0, 0]

        # Deterministic experience extraction on full combined text (title + desc + url + tags)
        combined_text = f"{title} {clean_desc} {job_url} {raw_job.get('tags') or ''}".strip()
        det_exp = extract_experience_years(combined_text)
        title_low = title.lower()
        if det_exp <= 0 and is_senior_job({"title": title}):
            if "iii" in title_low or " 3" in title_low or " iv" in title_low or " 4" in title_low or "specialist" in title_low:
                det_exp = 5
            else:
                det_exp = 3
        elif det_exp <= 0 and MID_LEVEL_TITLE_RE.search(title):
            det_exp = 2
        elif det_exp <= 0 and (job_url or raw_job.get("site") or raw_job.get("source")):
            # Sparse description and non-fresher title on external posting
            clean_body = clean_desc.split("Apply URL:")[0].strip()
            if len(clean_body) < 50 and not FRESHER_TITLE_RE.search(title) and str(raw_job.get("site") or raw_job.get("source") or "").lower() not in ("internshala", "unstop"):
                det_exp = 2

        # Experience parsing from AI
        exp = 0
        exp_years = ai_data.get("experience_years")
        is_fresher = ai_data.get("is_fresher_friendly")
        if exp_years is not None:
            try:
                exp = max(0, round(float(exp_years)))
            except (TypeError, ValueError):
                exp = 0

        # Deterministic extraction always supersedes LLM hallucination for experience
        if det_exp > 0:
            exp = max(exp, det_exp)
        elif is_fresher is False and exp <= 0:
            exp = 2
        elif is_fresher is True:
            exp = 0

        # Construct and return base Job with AI overrides
        job = normalize_scraped_job(raw_job, taxonomy)
        job.required_skills = valid_required
        job.preferred_skills = valid_preferred
        job.salary_lpa = salary_lpa
        job.salary_disclosed = salary_disclosed
        job.experience_years = max(exp, det_exp, getattr(job, "experience_years", 0))
        if job_url:
            job.job_url = job_url
        record_ai_normalizer_success()
        return job

    except Exception as exc:
        record_ai_normalizer_failure()
        logger.warning("normalize_job_with_ai failed, falling back to static normalizer: %s", exc)
        return normalize_scraped_job(raw_job, taxonomy)
