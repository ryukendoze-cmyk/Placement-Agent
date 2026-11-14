"""Institutional placement-policy RAG: hybrid lexical + dense retrieval over the
18 SOPs in data/placement_sops.json, plus a citation guard and an explainable
eligibility engine.

Zero-key offline: dense vectors come from the active Embedder (HashingEmbedder by
default); no network, no API key. Retrieval is deterministic.
"""
from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from pathlib import Path

from ..taxonomy.branches import is_branch_eligible

SOP_DIR = Path(__file__).resolve().parents[2] / "data"
_DEFAULT_SOP_FILE = SOP_DIR / "placement_sops.json"
SOP_FILE = _DEFAULT_SOP_FILE
_DEFAULT_OVERRIDES_FILE = SOP_DIR / "policies_overrides.json"
OVERRIDES_FILE = _DEFAULT_OVERRIDES_FILE

_SOP_ID = re.compile(r"\bSOP-(\d{2,3})\b", re.IGNORECASE)
_SOP_TOKEN = re.compile(r"\bSOP-\d{2,3}\b", re.IGNORECASE)

_cache: dict[str, list[dict]] = {}
_index_cache: dict[str, list[dict]] = {}
_cache_source: tuple[Path, Path] | None = None

_DEFAULT_SOPS_SNAPSHOT: list[dict] = (
    json.loads(SOP_FILE.read_text(encoding="utf-8")) if SOP_FILE.exists() else []
)


def _active_overrides_file() -> Path:
    if OVERRIDES_FILE != _DEFAULT_OVERRIDES_FILE:
        return OVERRIDES_FILE
    if SOP_FILE != _DEFAULT_SOP_FILE:
        return SOP_FILE.parent / "policies_overrides.json"
    return OVERRIDES_FILE


def _invalidate_cache() -> None:
    _cache.clear()
    _index_cache.clear()


def _write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_overrides(ov_file: Path) -> list[dict]:
    if not ov_file.exists():
        return []
    raw = json.loads(ov_file.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict) and r.get("id")]
    if isinstance(raw, dict) and isinstance(raw.get("sops"), list):
        return [r for r in raw["sops"] if isinstance(r, dict) and r.get("id")]
    return []


def load_sops() -> list[dict]:
    """Load the institutional SOP records merged with overrides (cached)."""
    global _cache_source
    ov_file = _active_overrides_file()
    src = (SOP_FILE, ov_file)
    if _cache_source != src:
        _invalidate_cache()
        _cache_source = src
    if "sops" not in _cache:
        base = json.loads(SOP_FILE.read_text(encoding="utf-8")) if SOP_FILE.exists() else []
        overrides = _read_overrides(ov_file)
        if overrides:
            by_id: dict[str, dict] = {}
            order: list[str] = []
            for row in base:
                if isinstance(row, dict) and row.get("id"):
                    sid = str(row["id"]).strip().upper()
                    by_id[sid] = row
                    order.append(sid)
            for row in overrides:
                sid = str(row["id"]).strip().upper()
                if row.get("_deleted"):
                    by_id.pop(sid, None)
                    continue
                if sid not in by_id:
                    order.append(sid)
                by_id[sid] = row
            _cache["sops"] = [by_id[sid] for sid in order if sid in by_id]
        else:
            _cache["sops"] = base
    return _cache["sops"]


list_sops = load_sops


def reload_sops() -> list[dict]:
    """Clear in-memory SOP and RAG caches and reload from SOP_FILE and overrides."""
    global _cache_source
    _invalidate_cache()
    _cache_source = (SOP_FILE, _active_overrides_file())
    loaded = load_sops()
    index_sops()
    return loaded


def get_sop(sop_id: str) -> dict | None:
    sid = (sop_id or "").strip().upper()
    for s in load_sops():
        if str(s.get("id", "")).strip().upper() == sid:
            return s
    return None


def load_sop_params(sop_id: str) -> dict:
    """Return a copy of the parameter dictionary for an SOP (live-synced)."""
    sop = get_sop(sop_id)
    params = sop.get("parameters") if isinstance(sop, dict) else None
    return dict(params) if isinstance(params, dict) else {}


def save_sops(sops: list[dict]) -> list[dict]:
    """Atomically persist core SOP definitions and rebuild RAG index caches."""
    ov_file = _active_overrides_file()
    existing_tombstones = {
        str(r.get("id", "")).strip().upper(): r
        for r in _read_overrides(ov_file)
        if r.get("_deleted") and r.get("id")
    }
    active_ids = {str(row.get("id", "")).strip().upper() for row in sops if row.get("id")}
    kept_tombstones = [row for sid, row in existing_tombstones.items() if sid not in active_ids]

    core_only = []
    custom_only = []
    for row in sops:
        sid = str(row.get("id", "")).strip().upper()
        m = _SOP_ID.fullmatch(sid)
        if m and int(m.group(1)) > 18:
            custom_only.append(row)
        else:
            core_only.append(row)

    # Keep deleted core records intact on disk in SOP_FILE so tombstones stay reversible
    tombstoned_core_ids = {
        str(r.get("id", "")).strip().upper()
        for r in kept_tombstones
        if (m := _SOP_ID.fullmatch(str(r.get("id", "")).strip().upper())) and int(m.group(1)) <= 18
    }
    if tombstoned_core_ids:
        disk_base = json.loads(SOP_FILE.read_text(encoding="utf-8")) if SOP_FILE.exists() else copy.deepcopy(_DEFAULT_SOPS_SNAPSHOT)
        core_by_id = {str(r.get("id", "")).strip().upper(): r for r in core_only if r.get("id")}
        for row in disk_base:
            rsid = str(row.get("id", "")).strip().upper()
            if rsid in tombstoned_core_ids and rsid not in core_by_id:
                core_by_id[rsid] = row
        ordered_core = []
        seen_core = set()
        for row in disk_base:
            rsid = str(row.get("id", "")).strip().upper()
            if rsid in core_by_id and rsid not in seen_core:
                ordered_core.append(core_by_id[rsid])
                seen_core.add(rsid)
        for row in core_only:
            rsid = str(row.get("id", "")).strip().upper()
            if rsid not in seen_core:
                ordered_core.append(row)
                seen_core.add(rsid)
        core_only = ordered_core

    _write_json_atomic(SOP_FILE, core_only)
    overrides_out = custom_only + kept_tombstones
    if overrides_out:
        _write_json_atomic(ov_file, overrides_out)
    elif ov_file.exists():
        ov_file.unlink()
    return reload_sops()


def update_sop(
    sop_id: str,
    *,
    title: str | None = None,
    category: str | None = None,
    section: str | None = None,
    rule_text: str | None = None,
    parameters: dict | None = None,
) -> dict:
    """Update an SOP's metadata, rule_text, and/or parameters, persist atomically, and re-index."""
    sid = (sop_id or "").strip().upper()
    current = copy.deepcopy(load_sops())
    target: dict | None = None
    for row in current:
        if str(row.get("id", "")).strip().upper() == sid:
            target = row
            break
    if target is None:
        raise KeyError(f"Unknown SOP: {sop_id}")
    if title is not None:
        t_clean = str(title).strip()
        if not t_clean:
            raise ValueError("title must not be empty")
        target["title"] = t_clean
    if category is not None:
        c_clean = str(category).strip()
        if not c_clean:
            raise ValueError("category must not be empty")
        target["category"] = c_clean
    if section is not None:
        s_clean = str(section).strip()
        if not s_clean:
            raise ValueError("section must not be empty")
        target["section"] = s_clean
    if rule_text is not None:
        cleaned = str(rule_text).strip()
        if not cleaned:
            raise ValueError("rule_text must not be empty")
        target["rule_text"] = cleaned
    if parameters is not None:
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be an object")
        merged = dict(target.get("parameters") or {})
        merged.update(parameters)
        target["parameters"] = merged
    save_sops(current)
    updated = get_sop(sid)
    return copy.deepcopy(updated or target)


def create_sop(payload: dict) -> dict:
    """Create a custom institutional SOP (SOP-19+) and persist to policies_overrides.json."""
    if not isinstance(payload, dict):
        raise ValueError("Policy payload must be a JSON object")

    current = copy.deepcopy(load_sops())
    existing_ids = {str(s.get("id", "")).strip().upper() for s in current if s.get("id")}

    raw_id = str(payload.get("id") or "").strip().upper()
    if raw_id:
        m = _SOP_ID.fullmatch(raw_id)
        if not m:
            raise ValueError("SOP id must match SOP-XX (e.g. SOP-19)")
        sid = f"SOP-{int(m.group(1)):02d}"
        if sid in existing_ids:
            raise ValueError(f"SOP id '{sid}' already exists")
    else:
        nums = [18]
        for eid in existing_ids:
            m = _SOP_ID.fullmatch(eid)
            if m:
                nums.append(int(m.group(1)))
        sid = f"SOP-{max(nums) + 1:02d}"

    title = str(payload.get("title") or "").strip()
    if not title:
        raise ValueError("Policy title is required")

    category = str(payload.get("category") or "General Governance").strip() or "General Governance"
    section = str(payload.get("section") or "Section 5.0: Institutional Rule").strip() or "Section 5.0: Institutional Rule"

    raw_text = payload.get("text") if payload.get("text") is not None else payload.get("rule_text")
    rule_text = str(raw_text or "").strip()
    if len(rule_text) < 15:
        raise ValueError("Policy text must be at least 15 characters")

    raw_params = payload.get("params") if payload.get("params") is not None else payload.get("parameters")
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        raise ValueError("Policy params must be an object")

    clean_params: dict[str, str | int | float | bool] = {}
    for pk, pv in raw_params.items():
        key = str(pk or "").strip()
        if not key:
            raise ValueError("Parameter key cannot be empty")
        if not isinstance(pv, (str, int, float, bool)):
            raise ValueError(f"Parameter '{key}' must be a string, number, or boolean")
        clean_params[key] = pv

    raw_kw = payload.get("keywords")
    if isinstance(raw_kw, list) and raw_kw:
        keywords = [str(k).strip().lower() for k in raw_kw if str(k).strip()]
    else:
        keywords = list(dict.fromkeys(re.findall(r"[a-z0-9]{3,}", f"{title} {category}".lower())))[:10]

    raw_applies = payload.get("applies_to")
    applies_to = (
        [str(a).strip() for a in raw_applies if str(a).strip()]
        if isinstance(raw_applies, list) and raw_applies
        else ["student", "tpo_admin"]
    )

    new_sop = {
        "id": sid,
        "title": title,
        "category": category,
        "section": section,
        "rule_text": rule_text,
        "keywords": keywords,
        "parameters": clean_params,
        "applies_to": applies_to,
        "is_custom": True,
    }

    ov_file = _active_overrides_file()
    overrides = [r for r in _read_overrides(ov_file) if str(r.get("id", "")).strip().upper() != sid]
    overrides.append(new_sop)
    _write_json_atomic(ov_file, overrides)
    reload_sops()
    return copy.deepcopy(get_sop(sid) or new_sop)


def delete_sop(sop_id: str) -> dict:
    """Delete any active SOP (core SOP-01..SOP-18 via tombstone, or custom SOP-19+) and reload caches."""
    sid = (sop_id or "").strip().upper()
    m = _SOP_ID.fullmatch(sid)
    if not m:
        raise ValueError(f"Unknown SOP id: {sop_id}")

    existing = get_sop(sid)
    if existing is None:
        raise ValueError(f"Unknown SOP id: {sop_id}")

    ov_file = _active_overrides_file()
    remaining = [r for r in _read_overrides(ov_file) if str(r.get("id", "")).strip().upper() != sid]
    if int(m.group(1)) <= 18:
        remaining.append({"id": sid, "_deleted": True})
    if remaining:
        _write_json_atomic(ov_file, remaining)
    elif ov_file.exists():
        ov_file.unlink()
    reload_sops()
    return copy.deepcopy(existing)


def reset_sops() -> list[dict]:
    """Restore core SOP-01..SOP-18 definitions to baseline defaults while preserving custom SOPs."""
    baseline = copy.deepcopy(_DEFAULT_SOPS_SNAPSHOT)
    ov_file = _active_overrides_file()
    custom_only = [
        r for r in _read_overrides(ov_file)
        if not r.get("_deleted")
        and (m := _SOP_ID.fullmatch(str(r.get("id", "")).strip().upper()))
        and int(m.group(1)) > 18
    ]
    _write_json_atomic(SOP_FILE, baseline)
    if custom_only:
        _write_json_atomic(ov_file, custom_only)
    elif ov_file.exists():
        ov_file.unlink()
    return reload_sops()


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


def search_sops(query: str, top_k: int = 3, embedder=None, *, min_score: float = 0.14) -> dict:
    """Hybrid retrieval: exact SOP-id pinning > keyword/lexical overlap + dense cosine.

    Applies a relevance floor when no SOP ID or lexical/semantic signal matches so
    off-topic / out-of-scope queries return ``status='out_of_scope'`` instead of
    misleading SOP hits.
    """
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
    # Ignore pure English stop-words for lexical overlap so unrelated questions
    # like "What is the weather?" don't falsely match SOP prose on "the"/"is".
    _stops = {"what", "is", "the", "a", "an", "for", "of", "to", "in", "on", "can", "i", "my", "how", "many", "and", "or"}
    sig_tokens = q_tokens - _stops or q_tokens
    pinned_ids = {f"SOP-{m.group(1).upper()}" for m in _SOP_ID.finditer(query)}

    scored = []
    for item in index:
        sop = item["sop"]
        s_exact = 1.0 if (sop["id"].upper() in pinned_ids) else 0.0
        overlap = sig_tokens & (item["keywords"] | item["title_tokens"] | item["rule_tokens"])
        s_lex = len(overlap) / max(1, len(sig_tokens))
        s_dense = max(0.0, cosine(q_emb, item["embedding"]))
        s_final = max(s_exact, 0.55 * s_lex + 0.45 * s_dense)
        scored.append((s_final, sop, s_lex, s_dense, s_exact))

    scored.sort(key=lambda t: (-t[0], t[1]["id"]))
    results = []
    for s_final, sop, s_lex, s_dense, s_exact in scored[: max(int(top_k), 0)]:
        if s_exact == 0.0 and s_final < min_score and s_lex == 0.0 and s_dense < 0.30:
            continue
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
    if not results:
        return {
            "query": query,
            "results": [],
            "count": 0,
            "top_sop_id": None,
            "status": "out_of_scope",
            "message": "No JKLU placement policy SOP matched this query (out of scope).",
        }
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
        .order_by(Interview.started_at.desc(), Interview.id.desc())
    ).all()
    if not rows:
        return None
    if job_id:
        job_rows = [r for r in rows if r.job_id == job_id and r.score is not None]
        if job_rows:
            # newest-first: a retake supersedes the earlier attempt
            return int(job_rows[0].score)
    scored_rows = [r for r in rows if r.score is not None]
    # Use the newest finished interview score globally rather than best historical score
    return int(scored_rows[0].score) if scored_rows else None


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

    # SOP-03: GPA cutoff + active backlog limits (job override if set, else tier rule)
    sop03 = load_sop_params("SOP-03")
    min_gpa = float(getattr(job, "min_gpa", 0.0) or 0.0)
    gpa_ok = (min_gpa == 0.0) or (float(student.gpa or 0.0) >= min_gpa)
    backlogs = int(getattr(student, "backlogs", 0) or 0)
    job_max_backlogs = getattr(job, "max_backlogs", None)
    default_tier_backlogs = (
        int(sop03.get("super_dream_max_backlogs", 0))
        if tier == "super_dream"
        else int(sop03.get("dream_max_backlogs", 1))
    )
    max_backlogs = int(job_max_backlogs) if job_max_backlogs is not None else default_tier_backlogs
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

    # SOP-04: mock-interview gate for applicable placement tiers
    sop04 = load_sop_params("SOP-04")
    raw_tiers = sop04.get("applicable_tiers", ["dream", "super_dream"])
    applicable_tiers = (
        {str(t).strip().lower() for t in raw_tiers}
        if isinstance(raw_tiers, (list, tuple, set))
        else {"dream", "super_dream"}
    )
    needs_mock = tier in applicable_tiers
    mock_score = _latest_mock_score(session, student_id, job_id)
    min_mock = int(sop04.get("min_mock_score", 65))
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
    verification_statuses = {
        r.skill_id: (getattr(r, "verification_status", "claimed") or "claimed")
        for r in ss_rows
    }
    js = score_job(
        student_skills,
        job,
        sim,
        ctx.taxonomy,
        levels=levels,
        verification_statuses=verification_statuses,
    )
    sop05 = load_sop_params("SOP-05")
    min_coverage_pct = int(sop05.get("min_coverage_pct", 40))
    blocked_band = str(sop05.get("blocked_band", "early")).strip().lower()
    required = len(job.required_skills or [])
    matched = len(js.matched_required)
    coverage = round(100 * matched / required) if required else 100
    skill_ok = coverage >= min_coverage_pct and (not blocked_band or js.band.lower() != blocked_band)
    checks.append({
        "name": "skill_readiness", "sop_id": "SOP-05", "passed": skill_ok,
        "detail": (
            f"Matched {matched}/{required} required skills "
            f"({coverage}% coverage, min {min_coverage_pct}%; {js.band} band)"
        ),
    })

    req_unverified = [
        d["skill"] for d in js.matched_required
        if d.get("verification_status") == "unverified"
    ]
    if req_unverified:
        checks.append({
            "name": "skill_verification",
            "sop_id": "SOP-08",
            "passed": False,
            "detail": (
                f"Required resume skill(s) flagged unverified after assessment: "
                f"{', '.join(req_unverified)}"
            ),
        })

    cited = [c["sop_id"] for c in checks]
    reasons = [c["detail"] for c in checks if not c["passed"]]
    eligible = all(c["passed"] for c in checks)
    suggested = []
    if not mock_ok and needs_mock:
        suggested.append(f"Retake the platform technical mock interview to raise score from "
                         f"{mock_score if mock_score is not None else 'none'} to >= {min_mock} [SOP-04]")
    if req_unverified:
        suggested.append(
            f"Re-verify unverified skill claim(s) ({', '.join(req_unverified)}) via remedial upskilling & interview [SOP-08]"
        )
    if not skill_ok and js.missing_required:
        missing = ", ".join(d["name"] for d in js.missing_required[:3])
        suggested.append(f"Complete recommended courses to close missing gap: {missing} [SOP-05]")
    if not gpa_ok:
        suggested.append(f"Meet the {min_gpa:.2f} CGPA cutoff before applying [SOP-03]")
    if not backlog_ok:
        suggested.append(f"Clear active backlogs to <= {max_backlogs} for {tier} tier eligibility [SOP-03]")
    if not branch_ok:
        suggested.append("Confirm branch eligibility with the TPO [SOP-09]")

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
        "unverified_skills": js.unverified_skills,
        "needs_reverification_skills": js.needs_reverification_skills,
    }
