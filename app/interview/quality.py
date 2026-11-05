"""Automated Question Quality Reviewer, Validator, and Auditor."""

from __future__ import annotations

import re
from typing import Any

from app.interview.bank import Question, load_bank
from app.taxonomy.taxonomy import load_taxonomy

VALID_ROLES = {"backend", "frontend", "data", "devops"}
TRIVIAL_PATTERNS = [
    re.compile(r"^(?:what\s+is|what\s+are|define|explain)\s+[a-z0-9_\s-]+\??$", re.IGNORECASE),
    re.compile(r"in\s+one\s+(?:word|sentence)", re.IGNORECASE),
]


def validate_question_quality(q: Question | dict[str, Any]) -> tuple[bool, str]:
    """Validate a technical interview question against quality standards.

    Checks:
    - Rejects simplistic trivial definition patterns (e.g. 'What is X?').
    - Rejects questions with length < 35 characters.
    - Rejects questions with < 3 keywords or single-letter/empty keywords.
    - Rejects questions with < 2 outline points.
    - Rejects invalid roles not in {'backend', 'frontend', 'data', 'devops'}.
    - Rejects skill IDs not in the canonical taxonomy (unless 'generic').

    Returns:
        (True, "OK") if question meets quality standards, else (False, reason).
    """
    if isinstance(q, dict):
        question = str(q.get("question") or "")
        keywords = q.get("keywords") or []
        outline = q.get("outline") or []
        role = str(q.get("role") or "")
        skill_id = str(q.get("skill_id") or "")
    else:
        question = getattr(q, "question", "") or ""
        keywords = getattr(q, "keywords", ()) or ()
        outline = getattr(q, "outline", ()) or ()
        role = getattr(q, "role", "") or ""
        skill_id = getattr(q, "skill_id", "") or ""

    q_strip = question.strip()
    is_trivial = any(pat.search(q_strip) for pat in TRIVIAL_PATTERNS)
    is_short = len(q_strip) < 35

    # Trivial and short rejection
    if is_trivial and is_short:
        return (False, "Trivial or simplistic definition pattern (question text too short, < 35 chars)")
    if is_trivial:
        return (False, "Trivial or simplistic definition question pattern")
    if is_short:
        return (False, f"Question text too short ({len(q_strip)} chars < 35)")

    # Keywords validation
    if len(keywords) < 3:
        return (False, f"Insufficient keywords ({len(keywords)} < 3)")
    if any(len(str(kw).strip()) <= 1 for kw in keywords):
        return (False, "Invalid trivial keyword (must be >= 2 characters)")

    # Outline points validation
    if len(outline) < 2:
        return (False, f"Insufficient outline points ({len(outline)} < 2)")

    # Role validation
    if role not in VALID_ROLES:
        return (False, f"Invalid role: '{role}'; must be one of {sorted(VALID_ROLES)}")

    # Skill taxonomy validation
    if skill_id != "generic":
        taxonomy = load_taxonomy()
        if taxonomy.get_opt(skill_id) is None:
            return (False, f"Invalid skill_id: '{skill_id}' not found in canonical taxonomy")

    return (True, "OK")


def audit_question_bank(bank: tuple[Question, ...] | None = None) -> dict[str, Any]:
    """Audit the interview question bank and generate a quality report."""
    if bank is None:
        bank = load_bank()

    approved: list[Question] = []
    rejections: list[dict[str, Any]] = []

    for q in bank:
        ok, reason = validate_question_quality(q)
        if ok:
            approved.append(q)
        else:
            rejections.append({"id": getattr(q, "id", None), "reason": reason})

    role_dist: dict[str, int] = {}
    diff_dist: dict[int, int] = {}
    skills: set[str] = set()

    for q in approved:
        r = getattr(q, "role", "unknown")
        role_dist[r] = role_dist.get(r, 0) + 1
        d = getattr(q, "difficulty", 0)
        diff_dist[d] = diff_dist.get(d, 0) + 1
        s = getattr(q, "skill_id", "")
        if s:
            skills.add(s)

    return {
        "total_questions": len(bank),
        "approved_questions": len(approved),
        "rejected_questions": len(rejections),
        "rejections": rejections,
        "skill_coverage_count": len(skills),
        "role_distribution": role_dist,
        "difficulty_distribution": diff_dist,
    }


def load_vetted_bank() -> tuple[Question, ...]:
    """Load questions filtered through validate_question_quality."""
    bank = load_bank()
    return tuple(q for q in bank if validate_question_quality(q)[0])
