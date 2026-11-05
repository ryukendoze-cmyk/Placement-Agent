"""Zero-hallucination grounding verifier.

Scans any text/JSON for entity references (job/course/skill/SOP/run ids) and checks
each against a known-entity set built from the DB + run memory. Unverified ids are
annotated `[UNVERIFIED: x]` (never deleted, never raised) and a grounding_score is
returned so callers can report how grounded a synthesis is.
"""
from __future__ import annotations

import re
from uuid import uuid4

PATTERNS = [
    # campus jobs (job-001) and scraped external jobs (job-ext-<hex8..12>)
    ("job", re.compile(r"\bjob-(?:\d{3}|ext-[0-9a-f]{8,12})\b")),
    ("course", re.compile(r"\bcourse-\d+\b")),
    ("sop", re.compile(r"\bSOP-\d{2}\b", re.IGNORECASE)),
    ("run", re.compile(r"\brun-[0-9a-f]{8}\b")),
    ("approval", re.compile(r"\bappr-[0-9a-f]{8}\b")),
    ("student", re.compile(r"\bstu-[0-9a-f]{8}\b")),
    ("interview", re.compile(r"\bint-[0-9a-f]{8}\b")),
    # plan ids are always `plan-<student_id>-<job_id>` (planner.py). Anchoring the
    # second `-` after the student-id segment stops plain English hyphenations
    # ("plan-based", "plan-driven") from being flagged as hallucinated ids.
    ("plan", re.compile(r"\bplan-[A-Za-z0-9_]+-[A-Za-z0-9_-]+")),
]
# taxonomy skill ids: lowercase snake/word tokens present in the taxonomy
_SKILL_CANDIDATE = re.compile(r"\bskill[:=]?\s*([a-z0-9_]+)\b")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


def build_known_entities(session, extra: set[str] | None = None) -> set[str]:
    from sqlmodel import select

    from ..models import Course, Interview, Job, Skill, Student, UpskillPlan
    known: set[str] = set()
    for m in (Job, Course, Student, Skill, Interview, UpskillPlan):
        try:
            for row in session.exec(select(m.id)).all():
                known.add(row if isinstance(row, str) else row[0])
        except Exception:  # pragma: no cover - defensive
            continue
    # institutional SOP ids (Phase 3): so citing [SOP-04] isn't flagged as a hallucination
    try:
        from ..rag.policy_rag import sop_ids
        for sid in sop_ids():
            known.add(sid)
            known.add(sid.upper())
            known.add(sid.lower())
    except Exception:  # pragma: no cover - defensive
        pass
    if extra:
        known |= extra
    return known


def _referenced_ids(text: str) -> list[tuple[str, str]]:
    """Return (type, id) for every entity reference found in the text."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for typ, pat in PATTERNS:
        for m in pat.finditer(text):
            tok = m.group(0)
            if tok in seen:
                continue
            seen.add(tok)
            found.append((typ, tok))
    return found


def verify_text(text: str, known: set[str]) -> dict:
    """Split `text` into verified/unverified citations; annotate unverified ones.

    Returns {"verified": [...], "unverified": [...], "grounding_score": 0..100,
             "annotated_text": text-with-UNVERIFIED-markers}.
    """
    if not isinstance(text, str) or not text:
        return {"verified": [], "unverified": [], "grounding_score": 100, "annotated_text": text or ""}
    annotated = text
    verified: list[str] = []
    unverified: list[str] = []
    for typ, tok in _referenced_ids(text):
        if tok in known:
            if tok not in verified:
                verified.append(tok)
        else:
            if tok not in unverified:
                unverified.append(tok)
            marker = f"[UNVERIFIED: {tok}]"
            if marker not in annotated:
                # annotate every occurrence (word-bounded, skipping ones already marked)
                annotated = re.sub(rf"\b{re.escape(tok)}\b(?!\s*\[UNVERIFIED:)",
                                   f"{tok} {marker}", annotated)
    total = len(verified) + len(unverified)
    score = round(100 * len(verified) / total) if total else 100
    return {"verified": verified, "unverified": unverified,
            "grounding_score": score, "annotated_text": annotated}


def verify_json(obj, known: set[str]) -> dict:
    """Verify every string value inside a JSON-ish structure; return the same shape
    with unverified ids annotated. Useful for a synthesized final answer object."""
    if isinstance(obj, str):
        return verify_text(obj, known)["annotated_text"]
    if isinstance(obj, list):
        return [verify_json(v, known) for v in obj]
    if isinstance(obj, dict):
        return {k: verify_json(v, known) for k, v in obj.items()}
    return obj
