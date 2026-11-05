"""Candidate skill-targeted AI job search query synthesizer (Milestone 5, Section 10.2 A).

Turns a candidate's verified ``StudentSkill`` rows (depth + mentions) and academic
branch into 2-3 realistic search terms for Indian tech hiring platforms such as
Naukri, LinkedIn India and Indeed India. The LLM path is preferred when a client
is supplied; a deterministic contextual synthesizer guarantees a usable result
whenever the LLM is absent, misconfigured or returns malformed output.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.models import Student, StudentSkill

logger = logging.getLogger(__name__)

SUGGEST_JOB_QUERIES_SYSTEM = (
    "TASK: suggest_job_queries\n"
    "You are an Indian tech recruitment assistant. Given a candidate's verified skills "
    "and branch, generate targeted, realistic job search query terms for Indian "
    "hiring platforms (e.g. 'Python FastAPI Backend Engineer', 'React TypeScript "
    "Developer'). If the candidate has a diverse skillset (multiple distinct technical skills), "
    "generate 2 to 3 targeted queries. If the candidate lacks a diverse skillset "
    "(0 or only 1 technical skill), generate exactly 1 focused query term. "
    "Respond ONLY with JSON: {\"queries\": [\"...\"]}"
)

#: Verified skill depth ranking: advanced > intermediate > basic.
LEVEL_RANK: dict[str, int] = {"advanced": 3, "intermediate": 2, "basic": 1}

#: Results can be 1 query (single/non-diverse skill) or up to 3 queries (diverse skillset).
MAX_QUERIES = 3
MIN_QUERIES = 2

#: Ordered fallback role suffixes used when combining skills into search terms.
_FALLBACK_TITLES: tuple[str, ...] = (
    "Software Engineer",
    "Developer",
    "Graduate Engineer Trainee",
)


def has_diverse_skillset(skills: list[StudentSkill]) -> bool:
    """Determine whether the candidate has a sufficiently diverse technical skillset.

    A skillset is considered diverse if the student possesses at least 2 distinct
    technical skills (ignoring soft skills).
    """
    if not skills:
        return False
    try:
        from app.taxonomy.taxonomy import load_taxonomy

        tax = load_taxonomy()
    except Exception:
        tax = None

    tech_skills: set[str] = set()
    for s in skills:
        sk_id = (s.skill_id or "").strip().lower()
        if not sk_id:
            continue
        if tax is not None:
            sk = tax.get_opt(sk_id)
            if sk is not None:
                if getattr(sk, "category", "") != "soft_skills":
                    tech_skills.add(sk.id)
                continue
        tech_skills.add(sk_id)

    return len(tech_skills) >= 2


def _skill_label(skill_id: str) -> str:
    """Render a taxonomy skill id as a human-readable search-term label.

    Tries the taxonomy first (falling back to the raw id when the taxonomy is
    unavailable), then prettifies ``skill_id``/separators so queries read like the
    free-text recruiters actually type (e.g. ``ci_cd`` -> ``CI CD``).
    """
    label = skill_id
    try:
        from app.taxonomy.taxonomy import load_taxonomy

        skill = load_taxonomy().get_opt(skill_id)
        if skill is not None and skill.name.strip():
            label = skill.name
    except Exception:  # noqa: BLE001 - taxonomy must never break query synthesis
        logger.debug("taxonomy lookup unavailable for skill %r", skill_id, exc_info=True)

    pretty = str(label).replace("_", " ").replace("-", " ").strip()
    if pretty.isupper() and len(pretty) <= 4:
        return pretty  # keep acronyms such as SQL/ML as typed
    return " ".join(word[:1].upper() + word[1:] for word in pretty.split())


def _sort_skills(skills: list[StudentSkill]) -> list[StudentSkill]:
    """Order skills by verified depth (advanced > intermediate > basic), then mentions desc."""
    return sorted(
        skills,
        key=lambda s: (LEVEL_RANK.get(str(s.level).lower(), 0), int(s.mentions or 0)),
        reverse=True,
    )


def _clean_queries(raw: Any) -> list[str]:
    """Coerce an LLM-provided payload into 2-3 non-empty, de-duplicated query strings.

    Accepts either ``{"queries": [...]}`` objects or bare JSON lists of strings.
    Returns an empty list when the payload is unusable so callers can fall back.
    """
    if isinstance(raw, dict):
        items = raw.get("queries")
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    if not isinstance(items, list):
        return []

    cleaned: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        if text and text.lower() not in {q.lower() for q in cleaned}:
            cleaned.append(text)
        if len(cleaned) >= MAX_QUERIES:
            break
    return cleaned


def _llm_queries(student: Student, sorted_skills: list[StudentSkill], llm: Any) -> list[str]:
    """Prompt the LLM (TASK: suggest_job_queries, json_mode=True) and parse its answer.

    Returns an empty list on any failure so the caller transparently falls back
    to the deterministic synthesizer.
    """
    if llm is None or not callable(getattr(llm, "complete", None)):
        return []

    skill_views = [
        {
            "skill_id": s.skill_id,
            "level": str(s.level),
            "mentions": int(s.mentions or 0),
        }
        for s in sorted_skills
    ]
    payload = json.dumps(
        {
            "branch": student.branch or "",
            "skills": skill_views,
        },
        ensure_ascii=False,
    )
    try:
        raw = llm.complete(system=SUGGEST_JOB_QUERIES_SYSTEM, user=payload, json_mode=True)
    except Exception:  # noqa: BLE001 - LLM outages must degrade to the fallback
        logger.warning("suggest_job_queries LLM call failed for student %s", student.id, exc_info=True)
        return []

    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return _clean_queries(json.loads(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("suggest_job_queries returned non-JSON output for student %s", student.id)
        return []


def _deterministic_queries(
    branch: str, labels: list[str], diverse: bool = True
) -> list[str]:
    """Contextual deterministic fallback: combine top skills and branch into realistic queries.

    If diverse is True, returns 2 to 3 targeted queries.
    If diverse is False, returns exactly 1 focused query string.
    """
    branch_label = (branch or "").strip() or "Engineering"
    if not labels:
        single = f"{branch_label} Graduate Engineer Trainee"
        if not diverse:
            return [single]
        return [
            single,
            "Junior Software Engineer",
            "Associate Software Developer",
        ][:MAX_QUERIES]

    if not diverse:
        top1 = labels[0]
        return [f"{top1} Developer"]

    top1 = labels[0]
    candidates: list[str] = []
    if len(labels) >= 2:
        candidates.append(f"{top1} {labels[1]} {_FALLBACK_TITLES[0]}")
    candidates.append(f"{top1} {_FALLBACK_TITLES[1]}")
    candidates.append(f"{branch_label} {_FALLBACK_TITLES[2]}")

    queries: list[str] = []
    for candidate in candidates:
        text = " ".join(candidate.split())
        if text and text.lower() not in {q.lower() for q in queries}:
            queries.append(text)
        if len(queries) >= MAX_QUERIES:
            break
    return queries[:MAX_QUERIES]


def suggest_job_queries_for_student(
    student: Student,
    skills: list[StudentSkill],
    llm: Any = None,
) -> list[str]:
    """Generate skill-targeted job search queries for an Indian tech candidate.

    If the candidate has a sufficient diverse skillset (multiple distinct technical skills),
    generates 2-3 targeted queries. If the candidate lacks diversity (0 or 1 skill),
    generates exactly 1 targeted query.

    Ordering contract (deterministic path): verified depth
    ``advanced`` > ``intermediate`` > ``basic``, then ``mentions`` descending.

    Args:
        student: Candidate whose ``branch`` grounds branch-specific fallbacks.
        skills: The candidate's verified ``StudentSkill`` rows.
        llm: Optional LLM client exposing ``complete(system, user, json_mode)``.

    Returns:
        List of targeted query strings (1 if non-diverse, 2-3 if diverse).
    """
    ordered = _sort_skills(list(skills or []))
    diverse = has_diverse_skillset(ordered)
    target_count = MAX_QUERIES if diverse else 1

    llm_queries = _llm_queries(student, ordered, llm)
    if not diverse:
        if llm_queries:
            return llm_queries[:1]
        labels = [_skill_label(s.skill_id) for s in ordered]
        return _deterministic_queries(student.branch or "", labels, diverse=False)

    if len(llm_queries) >= MIN_QUERIES:
        return llm_queries[:target_count]

    labels = [_skill_label(s.skill_id) for s in ordered]
    fallback = _deterministic_queries(student.branch or "", labels, diverse=True)
    if llm_queries:
        # Salvage any valid LLM query(s) ahead of the deterministic ones.
        merged = list(llm_queries)
        for query in fallback:
            if query.lower() not in {q.lower() for q in merged}:
                merged.append(query)
            if len(merged) >= target_count:
                break
        return merged[:target_count]
    return fallback[:target_count]
