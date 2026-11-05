"""Institutional placement-policy RAG: hybrid lexical + dense retrieval over the
18 SOPs in data/placement_sops.json, plus a citation guard and an explainable
eligibility engine.

Zero-key offline: dense vectors come from the active Embedder (HashingEmbedder by
default); no network, no API key. Retrieval is deterministic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..taxonomy.branches import is_branch_eligible

SOP_DIR = Path(__file__).resolve().parents[2] / "data"
SOP_FILE = SOP_DIR / "placement_sops.json"

_SOP_ID = re.compile(r"\bSOP-(\d{2})\b", re.IGNORECASE)
_SOP_TOKEN = re.compile(r"\bSOP-\d{2}\b", re.IGNORECASE)

_cache: dict[str, list[dict]] = {}
_index_cache: dict[str, list[dict]] = {}


def load_sops() -> list[dict]:
    """Load the 18 institutional SOP records (cached)."""
    if "sops" not in _cache:
        if not SOP_FILE.exists():
            _cache["sops"] = []
        else:
            _cache["sops"] = json.loads(SOP_FILE.read_text(encoding="utf-8"))
    return _cache["sops"]


def get_sop(sop_id: str) -> dict | None:
    sid = (sop_id or "").upper()
    for s in load_sops():
        if s.get("id", "").upper() == sid:
            return s
    return None


def sop_ids() -> set[str]:
    return {s["id"].upper() for s in load_sops()}


def _default_embedder():
    from ..embeddings.embedder import HashingEmbedder
    return HashingEmbedder()


def index_sops(embedder=None) -> list[dict]:
    """Build (and cache) an in-memory hybrid index: dense vectors + keyword sets."""
    embedder = embedder or _default_embedder()
    key = type(embedder).__name__
    if key in _index_cache:
        return _index_cache[key]
    sops = load_sops()
    index = []
    for sop in sops:
        text = f"{sop.get('title','')} {sop.get('rule_text','')}"
        emb = embedder.embed(text)
        index.append({
            "sop": sop,
            "embedding": emb,
            "keywords": {k.lower() for k in sop.get("keywords", [])},
            "title_tokens": set(re.findall(r"[a-z0-9]+", sop.get("title", "").lower())),
            "rule_tokens": set(re.findall(r"[a-z0-9]+", sop.get("rule_text", "").lower())),
        })
    _index_cache[key] = index
    return index


def _cosine(a, b) -> float:
    from ..embeddings.store import cosine
    return cosine(a, b)


def search_sops(query: str, top_k: int = 3, embedder=None) -> dict:
    """Hybrid retrieval: exact SOP-id pinning > keyword/lexical overlap + dense cosine."""
    query = (query or "").strip()
    if not query:
        return {"query": query, "results": [], "count": 0, "top_sop_id": None,
                "status": "empty"}
    index = index_sops(embedder)
    if not index:
        return {"query": query, "results": [], "count": 0, "top_sop_id": None,
                "status": "unavailable"}

    from ..embeddings.store import cosine
    embedder = embedder or _default_embedder()
    q_emb = embedder.embed(query)
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    pinned_ids = {f"SOP-{m.group(1).upper()}" for m in _SOP_ID.finditer(query)}

    scored = []
    for item in index:
        sop = item["sop"]
        s_exact = 1.0 if (sop["id"].upper() in pinned_ids) else 0.0
        overlap = q_tokens & (item["keywords"] | item["title_tokens"] | item["rule_tokens"])
        s_lex = len(overlap) / max(1, len(q_tokens))
        s_dense = max(0.0, cosine(q_emb, item["embedding"]))
        s_final = max(s_exact, 0.55 * s_lex + 0.45 * s_dense)
        scored.append((s_final, sop, s_lex, s_dense))

    scored.sort(key=lambda t: (-t[0], t[1]["id"]))
    results = []
    for s_final, sop, s_lex, s_dense in scored[: max(int(top_k), 0)]:
        results.append({
            "sop_id": sop["id"],
            "title": sop["title"],
            "section": sop["section"],
            "rule_text": sop["rule_text"],
            "score": round(float(s_final), 4),
            "lexical": round(float(s_lex), 4),
            "dense": round(float(s_dense), 4),
            "parameters": sop.get("parameters", {}),
        })
    top_id = results[0]["sop_id"] if results else None
    return {"query": query, "results": results, "count": len(results),
            "top_sop_id": top_id, "status": "grounded"}


def verify_policy_citations(text: str, known: set[str] | None = None) -> dict:
    """Audit SOP-XX references in text: verified / unverified / grounding_score.
    Unknown ids are annotated inline as [UNVERIFIED: SOP-XX] (flagged, not deleted)."""
    known = {k.upper() for k in (known or sop_ids())}
    if not isinstance(text, str) or not text:
        return {"verified": [], "unverified": [], "grounding_score": 100,
                "annotated_text": text or ""}
    seen: set[str] = set()
    verified: list[str] = []
    unverified: list[str] = []
    annotated = text
    for m in _SOP_TOKEN.finditer(text):
        tok = m.group(0).upper()
        if tok in seen:
            continue
        seen.add(tok)
        if tok in known:
            verified.append(tok)
        else:
            unverified.append(tok)
            marker = f"[UNVERIFIED: {tok}]"
            if marker in annotated:
                continue
            raw = m.group(0)
            # 1) bracketed citations [SOP-99] -> flag every occurrence
            if f"[{raw}]" in annotated:
                annotated = annotated.replace(f"[{raw}]", f"[{raw}] {marker}")
            # 2) any remaining bare occurrences -> flag each one
            annotated = re.sub(rf"(?<![\[\w]){re.escape(raw)}(?![\]\w])",
                               f"{raw} {marker}", annotated, flags=re.IGNORECASE)
    total = len(verified) + len(unverified)
    score = round(100 * len(verified) / total) if total else 100
    return {"verified": verified, "unverified": unverified,
            "grounding_score": score, "annotated_text": annotated}


# ---------------- eligibility engine (explainable, SOP-cited) ----------------

def _latest_mock_score(session, student_id: str, job_id: str | None = None) -> int | None:
    from sqlmodel import select

    from ..models import Interview
    rows = session.exec(
        select(Interview).where(Interview.student_id == student_id,
                                Interview.finished == True)  # noqa: E712
        .order_by(Interview.started_at.desc())
    ).all()
    if not rows:
        return None
    if job_id:
        job_rows = [r for r in rows if r.job_id == job_id and r.score is not None]
        if job_rows:
            # newest-first: a retake supersedes the earlier attempt
            return int(job_rows[0].score)
    scores = [int(r.score) for r in rows if r.score is not None]
    return max(scores) if scores else None


def _tier_of(job) -> str:
    from ..matching.engine import job_tier
    return job_tier(job)


def evaluate_student_eligibility(session, ctx, student_id: str,
                                 job_id: str | "Job") -> dict:
    """Multi-dimensional policy gate with per-check explainability and SOP citations.

    ``job_id`` may be a persisted job id (str) or an ephemeral ``Job`` instance
    (M20 drive simulator), in which case no DB lookup happens for the job.
    """
    from ..models import Job, Student
    student = session.get(Student, student_id)
    if isinstance(job_id, Job):
        job = job_id
    else:
        job = session.get(Job, job_id)
    if student is None or job is None:
        raise ValueError("student or job not found")

    tier = _tier_of(job)
    checks: list[dict] = []

    # SOP-03: GPA cutoff + active backlog limits by tier
    min_gpa = float(getattr(job, "min_gpa", 0.0) or 0.0)
    gpa_ok = (min_gpa == 0.0) or (float(student.gpa or 0.0) >= min_gpa)
    backlogs = int(getattr(student, "backlogs", 0) or 0)
    max_backlogs = 0 if tier == "super_dream" else (1 if tier == "dream" else 1)
    backlog_ok = backlogs <= max_backlogs
    academic_ok = gpa_ok and backlog_ok
    checks.append({
        "name": "gpa_cutoff", "sop_id": "SOP-03", "passed": academic_ok,
        "detail": (f"Student GPA {float(student.gpa or 0):.2f} meets the {min_gpa:.2f} cutoff; "
                   f"{backlogs} active backlogs within the {max_backlogs} limit for {tier}"
                   if academic_ok else
                   (f"Student GPA {float(student.gpa or 0):.2f} is below the {min_gpa:.2f} cutoff"
                    if not gpa_ok else
                    f"{backlogs} active backlogs exceed the {max_backlogs} limit for {tier}")),
    })

    # SOP-09: branch restriction
    allowed = [str(b).strip() for b in (getattr(job, "allowed_branches", None) or []) if str(b).strip()]
    branch = str(getattr(student, "branch", "") or "").strip()
    allowed_set = {str(b).strip().upper() for b in allowed}
    branch_ok = is_branch_eligible(branch, allowed)
    checks.append({
        "name": "branch_restriction", "sop_id": "SOP-09", "passed": branch_ok,
        "detail": (f"Branch '{branch.upper() or 'n/a'}' is in allowed branches {sorted(allowed_set)}"
                   if branch_ok else
                   f"Branch '{branch.upper() or 'n/a'}' is not in the allowed branches {sorted(allowed_set)}"),
    })

    # SOP-04: mock-interview gate for dream / super_dream
    needs_mock = tier in {"dream", "super_dream"}
    mock_score = _latest_mock_score(session, student_id, job_id)
    min_mock = 65
    if needs_mock:
        mock_ok = mock_score is not None and mock_score >= min_mock
        detail = (f"Latest completed mock interview score is {mock_score}; required >= {min_mock} for {tier} tier"
                  if mock_score is not None else
                  f"No completed mock interview on record; {tier} tier requires a score >= {min_mock}")
    else:
        mock_ok = True
        detail = f"Tier '{tier}' does not require the mock-interview gate"
    checks.append({"name": "tier_mock_gate", "sop_id": "SOP-04", "passed": mock_ok, "detail": detail})

    # SOP-05: skill coverage / readiness band
    from sqlmodel import select

    from ..embeddings.store import cosine
    from ..matching.engine import score_job
    from ..models import StudentSkill
    sv = ctx.vector_store.get("student", student_id)
    jv = ctx.job_vectors.get(job.id) or ctx.vector_store.get("job", job.id)
    if jv is None:
        jv = ctx.embedder.embed(job.description)
        ctx.vector_store.upsert("job", job.id, jv)
        ctx.job_vectors[job.id] = jv
    sim = cosine(sv, jv) if sv else 0.0
    ss_rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == student_id)).all()
    student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
    levels = {r.skill_id: r.level for r in ss_rows}
    js = score_job(student_skills, job, sim, ctx.taxonomy, levels=levels)
    required = len(job.required_skills or [])
    matched = len(js.matched_required)
    coverage = round(100 * matched / required) if required else 100
    skill_ok = coverage >= 40 and js.band != "early"
    checks.append({
        "name": "skill_readiness", "sop_id": "SOP-05", "passed": skill_ok,
        "detail": f"Matched {matched}/{required} required skills ({coverage}% coverage; {js.band} band)",
    })

    cited = [c["sop_id"] for c in checks]
    reasons = [c["detail"] for c in checks if not c["passed"]]
    eligible = all(c["passed"] for c in checks)
    suggested = []
    if not mock_ok and needs_mock:
        suggested.append(f"Retake the platform technical mock interview to raise score from "
                         f"{mock_score if mock_score is not None else 'none'} to >= {min_mock} [SOP-04]")
    if not skill_ok and js.missing_required:
        missing = ", ".join(d["name"] for d in js.missing_required[:3])
        suggested.append(f"Complete recommended courses to close missing gap: {missing} [SOP-05]")
    if not gpa_ok:
        suggested.append(f"Meet the {min_gpa:.2f} CGPA cutoff before applying [SOP-03]")
    if not backlog_ok:
        suggested.append(f"Clear active backlogs to <= {max_backlogs} for {tier} tier eligibility [SOP-03]")
    if not branch_ok:
        suggested.append(f"Confirm branch eligibility with the TPO [SOP-09]")

    return {
        "student_id": student.id, "student_name": student.name,
        "job_id": job.id, "job_title": job.title, "job_tier": tier,
        "salary_lpa": list(getattr(job, "salary_lpa", None) or [0, 0]),
        "eligible": eligible, "status": "eligible" if eligible else "blocked",
        "reasons": reasons, "checks": checks, "cited_sops": cited,
        "grounding_score": 100, "suggested_actions": suggested,
        "mock_interview_score": mock_score, "skill_coverage_pct": coverage,
        "match_score": int(js.score),
        "missing_required": [d.get("skill", "") for d in js.missing_required],
    }
