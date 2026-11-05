"""Dynamic, AI-powered resume skill & proficiency extraction (Milestone 1).

Replaces the legacy static keyword interception (``ADVANCED_HINT`` /
``INTERMEDIATE_HINT``) and naive line-mention counting with a contextual
pipeline driven by the configured ``LLMClient`` (``TASK: extract_resume``):

* ``extract_resume_with_ai`` - the primary entry point. Sends the scrubbed
  resume plus the 124-skill taxonomy to the LLM, strictly grounds every
  returned skill against the taxonomy (hallucination guard), verifies that
  each accepted skill is actually present in the resume text, merges any
  explicitly-mentioned skill the LLM omitted, and falls back to deterministic
  contextual analysis whenever the LLM is unavailable, errors, or returns
  invalid JSON - so a resume upload never fails with HTTP 500.
* ``extract_skills_with_llm`` - convenience wrapper returning only the
  ``{skill_id: skill_record}`` mapping.
* ``extract_skills`` - legacy contract preserved for existing callers
  (``{skill_id: [evidence lines]}``); delegates to the AI engine when an
  ``llm`` is supplied and performs the word-boundary alias scan otherwise.
* ``level_for`` - proficiency heuristic with an optional AI-assisted path,
  fully backward compatible with the original ``level_for(5) == "advanced"``
  unit-test contract.

``app/resume/extractor.py`` re-exports the public API for module-path
compatibility.
"""
from __future__ import annotations

import json
import logging
import re

from ..taxonomy.branches import normalize_branch
from ..taxonomy.taxonomy import Skill, Taxonomy

logger = logging.getLogger(__name__)

BOUND = r"(?<![a-z0-9]){alias}(?![a-z0-9])"

LEVELS = ("basic", "intermediate", "advanced")

EXTRACT_RESUME_SYSTEM = (
    "TASK: extract_resume\n"
    "You are an expert technical recruiter and AI resume parser. Analyze the candidate's "
    "resume dynamically and extract:\n"
    "1. Candidate profile metadata (name, branch, year, gpa, summary) if present in the text.\n"
    "For candidate profile 'branch', return standard canonical branch codes (e.g. 'CSE', 'IT', 'ECE', 'EE', 'ME', 'CE', or 'AI&DS') whenever applicable.\n"
    "2. Every technical and soft skill from the provided taxonomy that the candidate demonstrates.\n"
    "For each skill, determine proficiency level ('basic', 'intermediate', or 'advanced') "
    "dynamically based on contextual depth in the resume:\n"
    "- 'advanced': production internships/work experience, complex end-to-end systems, architectural ownership, quantified impact metrics, or deep multi-project mastery.\n"
    "- 'intermediate': applied hands-on project implementation, practical framework/tool usage, or multiple distinct applied contexts.\n"
    "- 'basic': listed only in a skills section, introductory coursework, or brief single mention without applied project depth.\n"
    "You MUST only return skills that exist in the provided taxonomy and are grounded in the resume text.\n"
    "Respond ONLY with valid JSON matching this schema:\n"
    '{"profile": {"name": str, "branch": str, "year": int, "gpa": float, "summary": str}, '
    '"skills": [{"skill_id": str, "level": "basic"|"intermediate"|"advanced", "mentions": int, "evidence": [str]}]}'
)

# --- contextual (deterministic) analysis primitives -------------------------

_METRIC_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|percent|ms|s\b|k\b|m\b|x\b|requests?|users?|customers?|rows?|"
    r"gb|mb|tb|qps|rps|lakh|crore|million|billion|times?|faster)",
    re.I,
)
_STRONG_VERBS = {
    "architected", "engineered", "optimized", "scaled", "led", "launched", "shipped",
    "orchestrated", "spearheaded", "automated", "migrated", "deployed", "refactored",
    "streamlined", "accelerated", "reduced", "improved",
}
_APPLIED_VERBS = {
    "built", "developed", "implemented", "created", "designed", "integrated", "tested",
    "configured", "maintained", "analyzed", "modeled", "delivered",
}
_EXPERIENCE_HINT = re.compile(
    r"\b(intern(ship)?|experience|employment|production|deployed|employer|company|"
    r"work(ed)? with|spearheaded|led)\b",
    re.I,
)
_PROJECT_HINT = re.compile(r"\b(project|built|developed|implemented|capstone|hackathon)\b", re.I)
_SKILLS_SECTION_HINT = re.compile(r"^\s*(skills?|technologies|tech stack|tools)\s*:", re.I)
_QUANTIFIED_SECTION_HINT = re.compile(
    r"\b(achievement|award|experience|internship|project|portfolio)\b", re.I
)


def _strip_code_fences(raw: str) -> str:
    """Drop optional markdown code fences around a JSON payload."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = [
            ln for ln in text.splitlines()
            if not ln.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()
    return text


def _safe_json(raw: str) -> dict | None:
    """Parse ``raw`` as a JSON object, tolerating markdown fences; None on failure."""
    if not isinstance(raw, str):
        return None
    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned or raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_skill(raw_id, taxonomy: Taxonomy) -> Skill | None:
    """Resolve an LLM-returned skill token to a canonical ``Skill`` (or None).

    Accepts the skill id, an alias, or a case-insensitive skill-name match so
    slightly-off LLM outputs still ground against the taxonomy.
    """
    if not isinstance(raw_id, str) or not raw_id.strip():
        return None
    token = raw_id.strip()
    skill = taxonomy.get_opt(token)
    if skill is not None:
        return skill
    resolved_id = taxonomy.resolve(token)
    if resolved_id is not None:
        return taxonomy.get_opt(resolved_id)
    low = token.lower()
    for candidate in taxonomy.all():
        if candidate.name.lower() == low:
            return candidate
    return None


def _alias_variants(skill: Skill, taxonomy: Taxonomy) -> list[str]:
    """Lower-cased name/id/alias variants used for grounding checks."""
    variants = {skill.id.lower(), skill.name.lower()}
    variants.update(a.lower() for a in (skill.aliases or ()))
    # alias_map is keyed by lower-cased alias/id and maps back to skill ids
    for alias, sid in taxonomy.alias_map.items():
        if sid == skill.id:
            variants.add(alias)
    return sorted(v for v in variants if v)


def _grounded(skill: Skill, text: str, evidence: list[str], taxonomy: Taxonomy) -> bool:
    """True when the skill (or one of its aliases) appears in ``text`` OR at
    least one non-empty evidence snippet is a substring of ``text``."""
    low_text = text.lower()
    for variant in _alias_variants(skill, taxonomy):
        if re.search(BOUND.format(alias=re.escape(variant)), low_text):
            return True
    for snippet in evidence:
        snippet = str(snippet).strip().lower()
        if snippet and snippet in low_text:
            return True
    return False


def _skill_lines(skill: Skill, text: str, taxonomy: Taxonomy) -> list[str]:
    """Resume lines in which this skill's name/id/aliases appear (word-bounded)."""
    out: list[str] = []
    for line in text.splitlines():
        low = line.lower()
        for variant in _alias_variants(skill, taxonomy):
            if re.search(BOUND.format(alias=re.escape(variant)), low):
                stripped = line.strip()
                if stripped and stripped not in out:
                    out.append(stripped)
                break
    return out


def _line_has_metric(line: str) -> bool:
    return bool(_METRIC_RE.search(line))


def _leading_verb(line: str) -> str:
    for token in re.findall(r"[A-Za-z]+", line):
        if token.isalpha():
            return token.lower()
    return ""


def _contextual_level(lines: list[str]) -> str:
    """Deterministic proficiency assessment from a skill's evidence lines.

    ``advanced``: quantified impact or strong engineering verbs / production or
    experience context, or >=3 distinct evidence lines.
    ``intermediate``: applied project usage or >=2 distinct evidence lines.
    ``basic``: a single flat mention (skills list, coursework, brief note).
    """
    if not lines:
        return "basic"
    joined = " ".join(lines).lower()
    metric_hits = sum(1 for ln in lines if _line_has_metric(ln))
    strong_verb_hits = sum(
        1 for ln in lines
        if _leading_verb(ln) in _STRONG_VERBS
        or any(v in ln.lower() for v in _STRONG_VERBS)
    )
    applied_hits = sum(
        1 for ln in lines
        if _leading_verb(ln) in _APPLIED_VERBS
        or any(v in ln.lower() for v in _APPLIED_VERBS)
    )
    experience_ctx = bool(_EXPERIENCE_HINT.search(joined))
    applied_ctx = experience_ctx or bool(_PROJECT_HINT.search(joined))
    flat_list_only = all(_SKILLS_SECTION_HINT.search(ln) for ln in lines)

    # >=3 distinct evidence lines means the skill shows up repeatedly across
    # sections (experience/projects/achievements) - treat as deep, multi-context
    # mastery even when the lines themselves are not verb-rich.
    if metric_hits >= 1 and (applied_ctx or strong_verb_hits >= 1):
        return "advanced"
    if strong_verb_hits >= 1 and (experience_ctx or metric_hits >= 1):
        return "advanced"
    if len(lines) >= 3 and not flat_list_only:
        return "advanced"
    if metric_hits >= 1 or strong_verb_hits >= 1:
        return "intermediate"
    if applied_ctx or applied_hits >= 1 or strong_verb_hits >= 1:
        return "intermediate"
    if len(lines) >= 2:
        return "intermediate"
    return "basic"


def _normalize_level(value, default: str = "basic") -> str:
    level = str(value or "").strip().lower()
    return level if level in LEVELS else default


def _normalize_evidence(value, text: str) -> list[str]:
    """Deduplicated, non-empty evidence snippets grounded in ``text``."""
    raw_items = value if isinstance(value, (list, tuple)) else [value]
    low_text = text.lower()
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        snippet = str(item or "").strip()
        if not snippet:
            continue
        key = snippet.lower()
        if key in seen:
            continue
        # keep only snippets traceable to the resume text (anti-hallucination)
        if snippet in text or key in low_text:
            seen.add(key)
            out.append(snippet)
    return out


def _normalize_mentions(value, evidence: list[str], lines: list[str]) -> int:
    try:
        mentions = int(value)
    except (TypeError, ValueError):
        mentions = 0
    if mentions <= 0:
        mentions = max(1, len(evidence) or len(lines) or 1)
    return max(1, mentions)


def _taxonomy_payload(taxonomy: Taxonomy) -> list[dict]:
    return [
        {"id": s.id, "name": s.name, "aliases": list(s.aliases)}
        for s in taxonomy.all()
    ]


def _empty_profile() -> dict:
    return {"name": "", "branch": "", "year": None, "gpa": None, "summary": ""}


def _normalize_profile(value) -> dict:
    profile = _empty_profile()
    if not isinstance(value, dict):
        return profile
    profile["name"] = str(value.get("name") or "").strip()
    profile["branch"] = normalize_branch(str(value.get("branch") or ""))
    summary = str(value.get("summary") or "").strip()
    profile["summary"] = summary
    year = value.get("year")
    try:
        profile["year"] = int(year) if year is not None and str(year).strip() != "" else None
    except (TypeError, ValueError):
        profile["year"] = None
    gpa = value.get("gpa")
    try:
        profile["gpa"] = float(gpa) if gpa is not None and str(gpa).strip() != "" else None
    except (TypeError, ValueError):
        profile["gpa"] = None
    return profile


def _contextual_profile(text: str) -> dict:
    """Lightweight deterministic profile guess used by the fallback path.

    Extracts a GPA, a branch (CSE/ECE/IT/AI-DS/... style token), a graduation
    year and a leading candidate name when the resume header exposes one.
    """
    profile = _empty_profile()
    if not text.strip():
        return profile

    gpa_match = re.search(
        r"\b(?:gpa|cgpa)\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,3})?)\b", text, re.I
    )
    if gpa_match:
        try:
            profile["gpa"] = float(gpa_match.group(1))
        except ValueError:
            profile["gpa"] = None

    branch_match = re.search(
        r"\b(CSE|CS|IT|ECE|EEE|ME|CE|AIML|AI|DS|AI&DS|AIDS|BT|CH)\b"
        r"|(computer science(?:\s+&\s*engineering)?|information technology|"
        r"artificial intelligence(?:\s*&\s*data science)?|electronics(?:\s*&\s*communication)?)",
        text, re.I,
    )
    if branch_match:
        matched_branch = (branch_match.group(1) or branch_match.group(2) or "").strip()
        profile["branch"] = normalize_branch(matched_branch)

    year_match = re.search(
        r"\b(?:graduat|batch|class of|class of \d{4}|b\.?tech.*?)(?:e?d)?\s*(?:year)?\s*[:\-]?\s*(20\d{2}|\d)\b",
        text, re.I,
    )
    if year_match:
        token = year_match.group(1)
        profile["year"] = int(token) if len(token) == 4 else int(token)

    # Leading candidate-name heuristic: first capitalized two-or-three word line
    # that is not a section header, contact line, degree/year/gpa marker, or date.
    contact_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|https?://|www\.|linkedin|github", re.I)
    header_re = re.compile(
        r"^(resume|curriculum vitae|cv|skills?|education|experience|projects?|"
        r"achievements?|summary|objective|contact|profile|certifications?)\b",
        re.I,
    )
    non_name_re = re.compile(
        r"\b(b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?e\b|m\.?e\b|bca|mca|"
        r"year|cgpa|gpa|semester|batch|student|engineer|developer|aspirant)\b",
        re.I,
    )
    verb_start_re = re.compile(
        r"^(built|developed|designed|implemented|created|architected|engineered|"
        r"led|ran|made|worked|spearheaded|launched|optimized|deployed|maintained|"
        r"used|wrote|configured|delivered|automated|integrated|tested)\b",
        re.I,
    )
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if not stripped or len(stripped) > 60:
            continue
        if contact_re.search(stripped) or header_re.match(stripped):
            continue
        words = re.findall(r"[A-Z][a-z]+", stripped)
        if not (2 <= len(words) <= 4):
            continue
        if non_name_re.search(stripped) or verb_start_re.match(stripped):
            continue
        profile["name"] = " ".join(words)
        break

    # Comma-separated header form: "Rohan Gupta, B.Tech CSE, Year 3, CGPA 8.4"
    # The leading segment before the first comma is often the clean candidate
    # name; validate it has no degree/year/gpa/verb markers before accepting.
    if not profile["name"]:
        for line in text.splitlines()[:8]:
            stripped = line.strip()
            if not stripped or len(stripped) > 60:
                continue
            if contact_re.search(stripped) or header_re.match(stripped):
                continue
            head, sep, _tail = stripped.partition(",")
            seg = head if sep else stripped
            if non_name_re.search(seg) or verb_start_re.match(seg):
                continue
            seg_words = re.findall(r"[A-Za-z]+", re.sub(r"[^A-Za-z\s]", " ", seg))
            if not (2 <= len(seg_words) <= 3):
                continue
            if any(not w[0].isupper() for w in seg_words):
                continue
            profile["name"] = " ".join(seg_words)
            break
    return profile


def _contextual_extract_resume(text: str, taxonomy: Taxonomy) -> dict:
    """Deterministic fallback: word-boundary alias scan + contextual levels."""
    skills: dict[str, dict] = {}
    if not text.strip():
        return {"profile": _empty_profile(), "skills": {}, "ai_parsed": False}

    for skill in taxonomy.all():
        lines = _skill_lines(skill, text, taxonomy)
        if not lines:
            continue
        skills[skill.id] = {
            "skill_id": skill.id,
            "level": _contextual_level(lines),
            "mentions": max(1, len(lines)),
            "evidence": list(lines),
        }
    return {
        "profile": _contextual_profile(text),
        "skills": skills,
        "ai_parsed": False,
    }


def extract_resume_with_ai(text: str, taxonomy: Taxonomy, llm: object | None = None) -> dict:
    """Parse a resume into ``{profile, skills, ai_parsed}``.

    When ``llm`` is provided the model is called once with
    ``EXTRACT_RESUME_SYSTEM`` and a ``{"resume_text", "taxonomy"}`` payload.
    Every returned skill is resolved against the taxonomy and must be grounded
    in the resume text (alias/name match or evidence substring), otherwise it is
    dropped as a hallucination. Explicitly mentioned skills the model omitted
    are merged back with a contextual level.

    If ``llm`` is ``None``, ``llm.complete`` raises, or the response is not
    valid JSON, the function logs a warning and falls back to deterministic
    contextual extraction - resume uploads never fail because of the LLM.
    """
    if not isinstance(text, str) or not text.strip():
        return {"profile": _empty_profile(), "skills": {}, "ai_parsed": False}

    if llm is None:
        logger.warning("extract_resume_with_ai: no LLM configured; using contextual fallback")
        return _contextual_extract_resume(text, taxonomy)

    payload = json.dumps({"resume_text": text, "taxonomy": _taxonomy_payload(taxonomy)})
    try:
        raw = llm.complete(system=EXTRACT_RESUME_SYSTEM, user=payload, json_mode=True)
    except Exception as exc:  # noqa: BLE001 - resilience contract: never fail the upload
        logger.warning("extract_resume_with_ai: LLM call failed (%s); falling back", exc)
        return _contextual_extract_resume(text, taxonomy)

    parsed = _safe_json(raw)
    if parsed is None:
        logger.warning("extract_resume_with_ai: LLM returned invalid JSON; falling back")
        return _contextual_extract_resume(text, taxonomy)

    profile = _normalize_profile(parsed.get("profile"))
    skills: dict[str, dict] = {}
    raw_skills = parsed.get("skills")
    if not isinstance(raw_skills, list):
        raw_skills = []

    for entry in raw_skills:
        if not isinstance(entry, dict):
            continue
        skill = _resolve_skill(entry.get("skill_id"), taxonomy)
        if skill is None:
            continue  # not in taxonomy -> reject (FK-safe)
        evidence = _normalize_evidence(entry.get("evidence"), text)
        if not _grounded(skill, text, evidence, taxonomy):
            continue  # ungrounded -> reject (hallucination guard)
        lines = _skill_lines(skill, text, taxonomy)
        evidence = evidence or lines
        if not evidence:
            continue
        mentions = _normalize_mentions(entry.get("mentions"), evidence, lines)
        skills[skill.id] = {
            "skill_id": skill.id,
            "level": _normalize_level(entry.get("level")),
            "mentions": mentions,
            "evidence": evidence,
        }

    # Merge explicit taxonomy skills the LLM omitted so they are never lost.
    for skill in taxonomy.all():
        if skill.id in skills:
            continue
        lines = _skill_lines(skill, text, taxonomy)
        if not lines:
            continue
        skills[skill.id] = {
            "skill_id": skill.id,
            "level": _contextual_level(lines),
            "mentions": max(1, len(lines)),
            "evidence": list(lines),
        }

    # Backfill any remaining blank fields from the contextual pass. The inner
    # per-field check makes this a no-op for keys the LLM already set, so no
    # outer gate is needed - any non-trivial condition would miss some
    # filled/empty field combination (regression-pinned by Test-Workhorse).
    ctx_profile = _contextual_profile(text)
    for key in ("name", "branch", "year", "gpa", "summary"):
        if not profile[key] and ctx_profile[key]:
            profile[key] = ctx_profile[key]

    return {"profile": profile, "skills": skills, "ai_parsed": True}


def extract_skills_with_llm(
    text: str, taxonomy: Taxonomy, llm: object | None = None
) -> dict[str, dict]:
    """Return ``{skill_id: skill_record}`` from the AI resume extraction."""
    return extract_resume_with_ai(text, taxonomy, llm)["skills"]


def extract_skills(
    text: str, taxonomy: Taxonomy, llm: object | None = None
) -> dict[str, list[str]]:
    """Return ``{skill_id: [evidence lines]}`` (legacy contract).

    With ``llm`` this delegates to the AI engine and maps each skill record to
    its evidence list; without it the original word-boundary alias scan runs
    unchanged for 100% backward compatibility.
    """
    if llm is not None:
        return {
            skill_id: list(record.get("evidence") or [])
            for skill_id, record in extract_resume_with_ai(text, taxonomy, llm)["skills"].items()
        }
    found: dict[str, list[str]] = {}
    for line in text.splitlines():
        low = line.lower()
        for alias, skill_id in taxonomy.alias_map.items():
            if re.search(BOUND.format(alias=re.escape(alias)), low):
                bucket = found.setdefault(skill_id, [])
                if line.strip() not in bucket:
                    bucket.append(line.strip())
    return found


ADVANCED_HINT = ("advanced", "expert", "proficient")
INTERMEDIATE_HINT = ("intermediate", "experienced", "worked with")


def level_for(
    mentions: int,
    evidence_lines: list[str] | None = None,
    *,
    llm: object | None = None,
    skill_id: str | None = None,
    resume_text: str | None = None,
    taxonomy: Taxonomy | None = None,
) -> str:
    """Proficiency level for a skill.

    Backward compatible with ``level_for(mentions, lines)`` and the
    ``level_for(5) == "advanced"`` / ``level_for(2) == "intermediate"`` /
    ``level_for(1) == "basic"`` unit-test contract.

    When ``llm`` and a resume context (``resume_text`` or ``evidence_lines``)
    are supplied, the AI contextual assessment
    (``extract_resume_with_ai``) decides the level; otherwise deterministic
    contextual signals in ``evidence_lines`` are evaluated.
    """
    if llm is not None and taxonomy is not None and (resume_text or evidence_lines):
        source = resume_text if (resume_text or "").strip() else "\n".join(evidence_lines or [])
        parsed = extract_resume_with_ai(source, taxonomy, llm)
        record = parsed["skills"].get(skill_id) if skill_id else None
        if record is not None:
            return _normalize_level(record.get("level"))

    lines = [str(ln) for ln in (evidence_lines or []) if str(ln).strip()]
    joined = " ".join(lines).lower()

    if mentions >= 4 or any(h in joined for h in ADVANCED_HINT):
        return "advanced"
    if mentions >= 2 or any(h in joined for h in INTERMEDIATE_HINT):
        return "intermediate"

    # Contextual signals (no static keyword trigger): a lone mention that shows
    # quantified impact / engineering depth in its evidence still counts.
    if lines and _contextual_level(lines) in ("advanced", "intermediate"):
        return _contextual_level(lines)
    return "basic"
