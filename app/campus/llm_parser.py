"""Structured LLM JD parser with strict timeout, Pydantic validation, and grounding guard."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.skills.extract import _alias_variants, _safe_json
from app.taxonomy.branches import normalize_branch
from app.taxonomy.taxonomy import Skill, Taxonomy

from .policy import apply_policy_and_defaults
from .static_parser import (
    CANONICAL_BRANCHES,
    _extract_compound_unmapped_tokens,
    _mask_spans,
    _skill_boundary_re,
)

LLM_TIMEOUT_SECONDS = 6.0


class LLMParseError(Exception):
    """Typed failure from LLM parsing/validation/grounding."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


class _LLMEligibility(BaseModel):
    min_cgpa: float | None = None
    allowed_branches: list[str] = Field(default_factory=list)
    max_active_backlogs: int | None = None
    graduation_year: int | None = None


class LLMJDOutput(BaseModel):
    company: str = ""
    role_title: str = ""
    location: str = ""
    work_mode: str | None = None
    ctc_lpa_min: float | None = None
    ctc_lpa_max: float | None = None
    salary_disclosed: bool = False
    stipend_inr_per_month: int | None = None
    experience_level: str | None = None
    eligibility: _LLMEligibility = Field(default_factory=_LLMEligibility)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


def _build_system_prompt(taxonomy: Taxonomy) -> str:
    skill_ids = ", ".join(s.id for s in taxonomy.all())
    return (
        "TASK: parse_campus_jd\n"
        "You are an institutional Campus Placement JD extractor for JKLU.\n"
        "Extract ONLY facts explicitly present in the JD text. Never invent cutoffs, CTC, or skills.\n"
        "SOP-01 Tier Rules (for reference; tier is computed deterministically): "
        "Standard < 8.0 LPA, Dream 8.0-14.99 LPA, Super-Dream >= 15.0 LPA (based on ctc_lpa_max). "
        "Internships and undisclosed salaries have no tier.\n"
        f"Canonical taxonomy skill IDs ONLY: [{skill_ids}]\n"
        "Canonical branch codes: CSE, IT, ECE, EE, ME, CE, AI&DS, CHE, BT (empty list [] means open to all).\n"
        "Respond ONLY with valid JSON matching:\n"
        '{"company": str, "role_title": str, "location": str, '
        '"work_mode": "On-site"|"Remote"|"Hybrid"|null, '
        '"ctc_lpa_min": float|null, "ctc_lpa_max": float|null, "salary_disclosed": bool, '
        '"stipend_inr_per_month": int|null, '
        '"experience_level": "Fresher"|"Internship"|null, '
        '"eligibility": {"min_cgpa": float|null, "allowed_branches": [str], '
        '"max_active_backlogs": int|null, "graduation_year": int|null}, '
        '"required_skills": [str], "preferred_skills": [str], '
        '"evidence": {"field": "exact quote from JD"}}'
    )


def _number_in_text(val: float | int, text: str, *, allow_inr_lakhs: bool = False) -> bool:
    """True when numeric value (or its INR lakh representation) appears in raw_jd_text."""
    num = float(val)
    candidates: set[str] = set()
    if num == int(num):
        i = int(num)
        candidates.add(str(i))
        candidates.add(f"{i}.0")
    candidates.add(f"{num:g}")
    candidates.add(f"{num:.1f}")
    candidates.add(f"{num:.2f}".rstrip("0").rstrip("."))
    if allow_inr_lakhs:
        inr = round(num * 100_000)
        candidates.add(str(inr))
        # Indian comma formatting e.g. 12,00,000
        s = str(inr)
        if len(s) > 3:
            last3 = s[-3:]
            rest = s[:-3]
            parts = []
            while len(rest) > 2:
                parts.append(rest[-2:])
                rest = rest[:-2]
            if rest:
                parts.append(rest)
            candidates.add(",".join(reversed(parts)) + "," + last3)
        # International comma formatting e.g. 1,200,000
        candidates.add(f"{inr:,}")
    else:
        i = round(num)
        candidates.add(str(i))
        candidates.add(f"{i:,}")
        if i >= 1000 and i % 1000 == 0:
            candidates.add(f"{i // 1000}k")

    low = text.lower()
    for cand in candidates:
        if re.search(rf"(?<![0-9.]){re.escape(cand.lower())}(?![0-9.])", low):
            return True
    return False


def _skill_grounded_in_text(skill: Skill, masked_text: str, taxonomy: Taxonomy) -> bool:
    low = masked_text.lower()
    for variant in _alias_variants(skill, taxonomy):
        if variant == "go":
            if "golang" in low or re.search(r"(?<![A-Za-z0-9\-])Go(?![A-Za-z0-9+\-#])", masked_text):
                return True
            continue
        if variant == "c":
            if re.search(r"(?<![A-Za-z0-9\-])C(?![A-Za-z0-9+\-#])", masked_text):
                return True
            continue
        if _skill_boundary_re(variant).search(low):
            return True
    return False


def _text_grounded(value: str, raw_text: str) -> bool:
    clean = (value or "").strip()
    if not clean:
        return True
    low_raw = raw_text.lower()
    if clean.lower() in low_raw:
        return True
    tokens = [t for t in re.findall(r"[a-z0-9]+", clean.lower()) if len(t) >= 3]
    if not tokens:
        return False
    return any(re.search(rf"\b{re.escape(t)}\b", low_raw) for t in tokens)


def _call_llm_with_timeout(
    llm: Any,
    system: str,
    user: str,
    timeout_seconds: float | None = None,
) -> str:
    effective_timeout = LLM_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(llm.complete, system, user, True)
    try:
        return future.result(timeout=effective_timeout)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise LLMParseError(
            "llm_timeout", f"LLM call timed out after {effective_timeout:.1f}s"
        ) from exc
    except LLMParseError:
        raise
    except Exception as exc:
        raise LLMParseError("llm_error", f"LLM call failed: {exc}") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def validate_and_ground_llm_json(
    raw_json_str: str,
    raw_jd_text: str,
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    """Parse JSON, validate schema with Pydantic, and enforce strict grounding."""
    parsed_dict = _safe_json(raw_json_str)
    if parsed_dict is None:
        raise LLMParseError("llm_invalid_json", "LLM returned invalid or non-object JSON")

    try:
        model = LLMJDOutput.model_validate(parsed_dict)
    except ValidationError as exc:
        raise LLMParseError("llm_schema_validation", f"LLM JSON schema error: {exc}") from exc

    # Mask unmapped compound phrases (e.g. Blockchain-Quantum-Rust++) so sub-tokens don't falsely ground
    unmapped_static, spans = _extract_compound_unmapped_tokens(raw_jd_text, taxonomy)
    masked_text = _mask_spans(raw_jd_text, spans)

    hallucinations: list[str] = []

    # 1. Ground skills strictly against taxonomy and JD text
    grounded_req: list[str] = []
    grounded_pref: list[str] = []
    req_seen: set[str] = set()
    pref_seen: set[str] = set()
    total_claimed_skills = 0
    ungrounded_skills = 0

    for raw_sid in model.required_skills:
        total_claimed_skills += 1
        sid_clean = str(raw_sid or "").strip()
        skill = taxonomy.get_opt(sid_clean)
        if skill is None:
            resolved = taxonomy.resolve(sid_clean)
            skill = taxonomy.get_opt(resolved) if resolved else None
        if skill is None or not _skill_grounded_in_text(skill, masked_text, taxonomy):
            ungrounded_skills += 1
            hallucinations.append(f"skill:{sid_clean}")
            continue
        if skill.id not in req_seen:
            req_seen.add(skill.id)
            grounded_req.append(skill.id)

    for raw_sid in model.preferred_skills:
        total_claimed_skills += 1
        sid_clean = str(raw_sid or "").strip()
        skill = taxonomy.get_opt(sid_clean)
        if skill is None:
            resolved = taxonomy.resolve(sid_clean)
            skill = taxonomy.get_opt(resolved) if resolved else None
        if skill is None or not _skill_grounded_in_text(skill, masked_text, taxonomy):
            ungrounded_skills += 1
            hallucinations.append(f"skill:{sid_clean}")
            continue
        if skill.id not in req_seen and skill.id not in pref_seen:
            pref_seen.add(skill.id)
            grounded_pref.append(skill.id)

    # 2. Ground numeric CTC / stipend / CGPA against raw_jd_text
    ctc_min = model.ctc_lpa_min
    ctc_max = model.ctc_lpa_max
    salary_disclosed = bool(model.salary_disclosed)
    stipend = model.stipend_inr_per_month
    min_cgpa = model.eligibility.min_cgpa

    if ctc_min is not None and not _number_in_text(ctc_min, raw_jd_text, allow_inr_lakhs=True):
        hallucinations.append(f"ctc_lpa_min:{ctc_min}")
        ctc_min = None
    if ctc_max is not None and not _number_in_text(ctc_max, raw_jd_text, allow_inr_lakhs=True):
        hallucinations.append(f"ctc_lpa_max:{ctc_max}")
        ctc_max = None
    if ctc_max is None and ctc_min is None:
        salary_disclosed = False

    if stipend is not None and not _number_in_text(stipend, raw_jd_text, allow_inr_lakhs=False):
        hallucinations.append(f"stipend:{stipend}")
        stipend = None

    if min_cgpa is not None and not _number_in_text(min_cgpa, raw_jd_text, allow_inr_lakhs=False):
        hallucinations.append(f"min_cgpa:{min_cgpa}")
        min_cgpa = None

    # 3. Ground company / role_title
    company = model.company.strip()
    role_title = model.role_title.strip()
    location = model.location.strip()
    if company and not _text_grounded(company, raw_jd_text):
        hallucinations.append(f"company:{company}")
        company = ""
    if role_title and not _text_grounded(role_title, raw_jd_text):
        hallucinations.append(f"role_title:{role_title}")
        role_title = ""
    if location and not _text_grounded(location, raw_jd_text):
        location = ""

    # Fail grounding if numbers were fabricated, all/majority skills failed grounding, or core identity ungrounded
    grounded_skill_count = len(grounded_req) + len(grounded_pref)
    numeric_hallucination = any(
        h.startswith(("ctc_lpa_", "min_cgpa:", "stipend:")) for h in hallucinations
    )
    skill_hallucination = total_claimed_skills > 0 and (
        grounded_skill_count == 0 or ungrounded_skills > grounded_skill_count
    )
    if numeric_hallucination or skill_hallucination:
        raise LLMParseError(
            "llm_grounding_failed",
            f"LLM output failed grounding checks: {', '.join(hallucinations)}",
        )

    # 4. Normalize branches
    branches: list[str] = []
    for b in model.eligibility.allowed_branches:
        code = normalize_branch(b)
        if code in CANONICAL_BRANCHES and code not in branches:
            branches.append(code)

    explicit: set[str] = set()
    field_conf: dict[str, float] = {}
    if company:
        explicit.add("company")
        field_conf["company"] = 0.9
    if role_title:
        explicit.add("role_title")
        field_conf["role_title"] = 0.9
    if location:
        explicit.add("location")
        field_conf["location"] = 0.9
    if model.work_mode in ("On-site", "Remote", "Hybrid"):
        explicit.add("work_mode")
        field_conf["work_mode"] = 0.9
    if model.experience_level in ("Fresher", "Internship"):
        explicit.add("experience_level")
        field_conf["experience_level"] = 0.9
    if ctc_max is not None:
        explicit.add("ctc_lpa_max")
        field_conf["ctc_lpa_max"] = 0.9
    if min_cgpa is not None:
        explicit.add("min_cgpa")
        field_conf["min_cgpa"] = 0.9
    if model.eligibility.max_active_backlogs is not None:
        explicit.add("max_active_backlogs")
        field_conf["max_active_backlogs"] = 0.9
    if model.eligibility.graduation_year is not None:
        explicit.add("graduation_year")
        field_conf["graduation_year"] = 0.9
    if branches:
        explicit.add("allowed_branches")
        field_conf["allowed_branches"] = 0.9
    field_conf["skills"] = 0.9 if (grounded_req or grounded_pref) else 0.3

    drive = {
        "company": company,
        "role_title": role_title,
        "location": location,
        "work_mode": model.work_mode if model.work_mode in ("On-site", "Remote", "Hybrid") else None,
        "ctc_lpa_min": ctc_min,
        "ctc_lpa_max": ctc_max,
        "salary_disclosed": salary_disclosed,
        "stipend_inr_per_month": stipend,
        "policy_tier": None,
        "policy_tier_source": None,
        "experience_level": (
            model.experience_level
            if model.experience_level in ("Fresher", "Internship")
            else None
        ),
        "source_portal": "On-Campus (Local)",
        "source_badge": "On-Campus · JKLU",
        "eligibility": {
            "min_cgpa": min_cgpa,
            "allowed_branches": branches,
            "max_active_backlogs": model.eligibility.max_active_backlogs,
            "graduation_year": model.eligibility.graduation_year,
        },
        "required_skills": grounded_req,
        "preferred_skills": grounded_pref,
        "unmapped_skill_mentions": unmapped_static,
        "raw_jd_text": raw_jd_text,
        "parser_metadata": {
            "engine_used": "llm_structured",
            "fallback_reason": None,
            "attempts": [],
            "confidence_score": 0.0,
            "field_confidence": field_conf,
            "evidence": {
                str(k): str(v)
                for k, v in (model.evidence or {}).items()
                if str(v or "").strip()
            },
            "missing_fields_flagged": [],
            "defaults_applied": [],
        },
        "_explicit_fields": sorted(explicit),
    }
    return apply_policy_and_defaults(drive, explicitly_found=explicit)


def parse_jd_with_llm(
    text: str,
    taxonomy: Taxonomy,
    llm: Any,
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run LLM extraction in a worker thread with hard 6.0s timeout + grounding validation."""
    if llm is None:
        raise LLMParseError("offline_mode", "No LLM client configured")
    system = _build_system_prompt(taxonomy)
    user = json.dumps({"jd_text": text})
    raw_output = _call_llm_with_timeout(llm, system, user, timeout_seconds=timeout_seconds)
    return validate_and_ground_llm_json(raw_output, text, taxonomy)
