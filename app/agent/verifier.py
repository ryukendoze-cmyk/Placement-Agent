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


_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+", re.IGNORECASE)
_COURSE_ID_EXT_RE = re.compile(r"\bcourse-(?:\d+|[a-z]{2,12}-\d+)\b", re.IGNORECASE)

_SOP_MOCK_CLAIM = re.compile(
    r"SOP-04[^.\n]{0,80}?\b(?:score|cutoff|threshold|minimum|requires?|at\s+least|>=)\s*(?:of\s*|is\s*)?(\d{1,3})\b",
    re.IGNORECASE,
)
_SOP_SUPER_BACKLOG_CLAIM = re.compile(
    r"SOP-03[^.\n]{0,80}?\b(?:allows?|permits?)\s+(\d+)\s+(?:active\s+)?backlogs?\s+for\s+super[- ]?dream",
    re.IGNORECASE,
)
_SOP_DREAM_BACKLOG_CLAIM = re.compile(
    r"(?:SOP-03[^.\n]{0,80}?\b|\bdream[^.\n]{0,40}?\b)(?:allows?|permits?)(?:\s+up\s+to)?\s+(\d+)\s+(?:active\s+)?backlogs?(?:\s+for\s+dream)?",
    re.IGNORECASE,
)
_STANDARD_BACKLOG_CLAIM = re.compile(
    r"(?:SOP-03[^.\n]{0,80}?|\bstandard\b[^.\n]{0,40}?)(?:allows?|permits?)(?:\s+up\s+to)?\s+(\d+)\s+(?:active\s+)?backlogs?",
    re.IGNORECASE,
)
_DREAM_LPA_CLAIM = re.compile(
    r"\bdream(?:\s+(?:tier|offers?|drives?|roles?))?\s*(?:starts?\s+at|above|over|exceeds?|minimum(?:\s+of)?|(?:go\s+)?up\s+to|>=|is|:)\s*(\d+(?:\.\d+)?)\s*(?:to\s*\d+(?:\.\d+)?\s*)?lpa\b",
    re.IGNORECASE,
)
_SUPER_DREAM_LPA_CLAIM = re.compile(
    r"\bsuper[- ]?dream(?:\s+(?:tier|offers?|drives?|roles?))?\s*(?:starts?\s+at|above|from|minimum(?:\s+of)?|at\s+least|>=|is|:)\s*(\d+(?:\.\d+)?)\s*lpa\b",
    re.IGNORECASE,
)
_STANDARD_LPA_CLAIM = re.compile(
    r"\bstandard(?:\s+(?:tier|offers?|drives?|roles?))?\s*(?:below|under|up\s+to|max(?:imum)?|starts?\s+at|<|is|:)\s*(\d+(?:\.\d+)?)\s*lpa\b",
    re.IGNORECASE,
)
_SOP10_ATTENDANCE_CLAIM = re.compile(
    r"SOP-10[^.\n]{0,80}?\b(\d{1,3})\s*%",
    re.IGNORECASE,
)
_SOP14_CGPA_CLAIM = re.compile(
    r"SOP-14[^.\n]{0,80}?\b(\d+(?:\.\d+)?)\s*(?:cgpa|gpa)\b",
    re.IGNORECASE,
)
_GPA_VALUE_CLAIM = re.compile(
    r"\b(?:cgpa|gpa)\s*(?:of|>=|minimum|requires?|at\s+least|is|:)?\s*(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_SOP05_COVERAGE_CLAIM = re.compile(
    r"SOP-05[^.\n]{0,80}?\b(\d{1,3})\s*%\s*(?:(?:required[- ]?)?skill\s*)?coverage",
    re.IGNORECASE,
)


def _load_sop_facts(sop_facts: dict | None = None) -> dict[str, float | int]:
    """Build numeric fact table dynamically from placement_sops.json with safe defaults."""
    facts: dict[str, float | int] = {
        "min_mock_score": 65,
        "min_coverage_pct": 40,
        "standard_max_lpa": 8.0,
        "dream_min_lpa": 8.0,
        "dream_max_lpa": 14.99,
        "super_dream_min_lpa": 15.0,
        "super_dream_max_backlogs": 0,
        "dream_max_backlogs": 1,
        "min_attendance_pct": 75,
        "internship_min_cgpa": 7.0,
    }
    try:
        from ..rag.policy_rag import load_sops
        for row in load_sops():
            params = row.get("parameters") or {}
            for k in list(facts):
                if k in params and isinstance(params[k], (int, float)):
                    facts[k] = params[k]
            if row.get("id") == "SOP-14" and isinstance(params.get("min_cgpa"), (int, float)):
                facts["internship_min_cgpa"] = params["min_cgpa"]
    except Exception:
        pass
    if sop_facts:
        for k, v in sop_facts.items():
            if isinstance(v, (int, float)):
                facts[k] = v
    return facts


def verify_sop_numeric_claims(text: str, *, sop_facts: dict | None = None) -> list[str]:
    """Check numeric policy claims anchored to SOP citations/rules against official parameters.

    Returns a list of unverified claim tokens (empty when all numeric claims match SOPs).
    """
    if not isinstance(text, str) or not text:
        return []
    facts = _load_sop_facts(sop_facts)
    min_mock = int(facts["min_mock_score"])
    sd_max_bl = int(facts["super_dream_max_backlogs"])
    d_max_bl = int(facts["dream_max_backlogs"])
    d_min_lpa = float(facts["dream_min_lpa"])
    d_max_lpa = float(facts["dream_max_lpa"])
    sd_min_lpa = float(facts["super_dream_min_lpa"])
    std_max_lpa = float(facts["standard_max_lpa"])
    min_att = int(facts["min_attendance_pct"])
    intern_cgpa = float(facts["internship_min_cgpa"])
    min_cov = int(facts["min_coverage_pct"])

    mismatches: list[str] = []

    def _add(err: str) -> None:
        if err not in mismatches:
            mismatches.append(err)

    for m in _SOP_MOCK_CLAIM.finditer(text):
        try:
            val = int(m.group(1))
            if val != min_mock and val != 100:
                _add(f"SOP-04:mock_threshold={val}!={min_mock}")
        except ValueError:
            pass
    for m in _SOP_SUPER_BACKLOG_CLAIM.finditer(text):
        try:
            val = int(m.group(1))
            if val != sd_max_bl:
                _add(f"SOP-03:super_dream_backlogs={val}!={sd_max_bl}")
        except ValueError:
            pass
    for m in _SOP_DREAM_BACKLOG_CLAIM.finditer(text):
        span = m.group(0).lower()
        if "super" in span:
            continue
        try:
            val = int(m.group(1))
            if val > d_max_bl:
                _add(f"SOP-03:dream_backlogs={val}>{d_max_bl}")
        except ValueError:
            pass
    for m in _STANDARD_BACKLOG_CLAIM.finditer(text):
        span = m.group(0).lower()
        if "super" in span or "dream" in span:
            continue
        try:
            val = int(m.group(1))
            if val > d_max_bl:
                _add(f"SOP-03:backlog_limit={val}>{d_max_bl}")
        except ValueError:
            pass
    for m in _DREAM_LPA_CLAIM.finditer(text):
        span = m.group(0).lower()
        if "super" in span:
            continue
        try:
            val = float(m.group(1))
            if val not in (d_min_lpa, d_max_lpa, sd_min_lpa) and abs(val - d_min_lpa) > 0.01:
                _add(f"SOP-01:dream_lpa={val:g}!={d_min_lpa:g}")
        except ValueError:
            pass
    for m in _SUPER_DREAM_LPA_CLAIM.finditer(text):
        try:
            val = float(m.group(1))
            if abs(val - sd_min_lpa) > 0.01:
                _add(f"SOP-01:super_dream_lpa={val:g}!={sd_min_lpa:g}")
        except ValueError:
            pass
    for m in _STANDARD_LPA_CLAIM.finditer(text):
        try:
            val = float(m.group(1))
            if abs(val - std_max_lpa) > 0.01:
                _add(f"SOP-01:standard_lpa={val:g}!={std_max_lpa:g}")
        except ValueError:
            pass
    for m in _SOP10_ATTENDANCE_CLAIM.finditer(text):
        try:
            val = int(m.group(1))
            if val != min_att:
                _add(f"SOP-10:attendance_pct={val}!={min_att}")
        except ValueError:
            pass
    for m in _SOP14_CGPA_CLAIM.finditer(text):
        try:
            val = float(m.group(1))
            if abs(val - intern_cgpa) > 0.01:
                _add(f"SOP-14:internship_cgpa={val:g}!={intern_cgpa:g}")
        except ValueError:
            pass
    for m in _GPA_VALUE_CLAIM.finditer(text):
        try:
            val = float(m.group(1))
            if val > 10.0:
                _add(f"SOP-03:invalid_gpa={val:g}>10")
        except ValueError:
            pass
    for m in _SOP05_COVERAGE_CLAIM.finditer(text):
        try:
            val = int(m.group(1))
            if val != min_cov:
                _add(f"SOP-05:min_coverage_pct={val}!={min_cov}")
        except ValueError:
            pass
    return mismatches


def verify_text(
    text: str,
    known: set[str] | list[str] | tuple[str, ...] | object | None = None,
    *,
    allowed_urls: set[str] | list[str] | None = None,
    allowed_course_ids: set[str] | list[str] | None = None,
    sop_facts: dict | None = None,
) -> dict:
    """Split `text` into verified/unverified citations; annotate unverified ones.

    Returns {"verified": [...], "unverified": [...], "grounding_score": 0..100,
             "annotated_text": text-with-UNVERIFIED-markers}.
    """
    if not isinstance(text, str) or not text:
        return {"verified": [], "unverified": [], "grounding_score": 100, "annotated_text": text or ""}

    if hasattr(known, "exec"):
        known_set = build_known_entities(known)
    else:
        known_set = set(known or ())

    allowed_url_norm = (
        {str(u).strip().rstrip("/") for u in allowed_urls if str(u).strip()}
        if allowed_urls is not None
        else None
    )
    allowed_course_set = (
        {str(c).strip() for c in allowed_course_ids if str(c).strip()}
        if allowed_course_ids is not None
        else None
    )

    annotated = text
    verified: list[str] = []
    unverified: list[str] = []

    refs = _referenced_ids(text)
    if allowed_course_set is not None:
        seen_toks = {t for _, t in refs}
        for m in _COURSE_ID_EXT_RE.finditer(text):
            tok = m.group(0)
            if tok not in seen_toks:
                seen_toks.add(tok)
                refs.append(("course", tok))

    for typ, tok in refs:
        if typ == "course" and allowed_course_set is not None:
            is_known = tok in allowed_course_set
        else:
            is_known = tok in known_set or tok.upper() in known_set or tok.lower() in known_set

        if is_known:
            if tok not in verified:
                verified.append(tok)
        else:
            if tok not in unverified:
                unverified.append(tok)
            marker = f"[UNVERIFIED: {tok}]"
            if marker not in annotated:
                # annotate every occurrence (word-bounded, skipping ones already marked)
                annotated = re.sub(
                    rf"\b{re.escape(tok)}\b(?!\s*\[UNVERIFIED:)",
                    f"{tok} {marker}",
                    annotated,
                )

    if allowed_url_norm is not None:
        seen_urls: set[str] = set()
        for m in _URL_RE.finditer(text):
            url = m.group(0).rstrip(".,;:!?")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            norm = url.rstrip("/")
            if norm in allowed_url_norm:
                if url not in verified:
                    verified.append(url)
            else:
                if url not in unverified:
                    unverified.append(url)
                marker = f"[UNVERIFIED: {url}]"
                if marker not in annotated:
                    annotated = annotated.replace(url, f"{url} {marker}")

    for num_err in verify_sop_numeric_claims(text, sop_facts=sop_facts):
        if num_err not in unverified:
            unverified.append(num_err)
        marker = f"[UNVERIFIED: {num_err}]"
        if marker not in annotated:
            annotated = f"{annotated} {marker}"
    total = len(verified) + len(unverified)
    score = round(100 * len(verified) / total) if total else 100
    return {"verified": verified, "unverified": unverified,
            "grounding_score": score, "annotated_text": annotated}


def verify_json(
    obj,
    known: set[str] | object | None = None,
    *,
    allowed_urls: set[str] | list[str] | None = None,
    allowed_course_ids: set[str] | list[str] | None = None,
    sop_facts: dict | None = None,
):
    """Verify every string value inside a JSON-ish structure; return the same shape
    with unverified ids annotated. Useful for a synthesized final answer object."""
    if isinstance(obj, str):
        return verify_text(
            obj,
            known,
            allowed_urls=allowed_urls,
            allowed_course_ids=allowed_course_ids,
            sop_facts=sop_facts,
        )["annotated_text"]
    if isinstance(obj, list):
        return [
            verify_json(
                v,
                known,
                allowed_urls=allowed_urls,
                allowed_course_ids=allowed_course_ids,
                sop_facts=sop_facts,
            )
            for v in obj
        ]
    if isinstance(obj, dict):
        return {
            k: verify_json(
                v,
                known,
                allowed_urls=allowed_urls,
                allowed_course_ids=allowed_course_ids,
                sop_facts=sop_facts,
            )
            for k, v in obj.items()
        }
    return obj
