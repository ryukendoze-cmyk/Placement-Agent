"""Database merge, vector indexing, and sanitation pipeline for scraped job records."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlmodel import Session, select

from app.models import Interview, Job, JobApplication, JobMatch, PlanItem, UpskillPlan
from app.scraper.normalizer import (
    CORE_FALLBACK_SKILLS,
    extract_experience_years,
    is_ai_normalizer_circuit_open,
    is_senior_job,
    normalize_job_with_ai,
    normalize_scraped_job,
)
from app.scraper.service import _clean_key, _clean_url
from app.taxonomy.taxonomy import Taxonomy, load_taxonomy

logger = logging.getLogger(__name__)

#: Placeholder domains that must never persist (ARCHITECTURE §12.3 zero-mock policy).
DUMMY_URL_PATTERNS: tuple[str, ...] = (
    "%example.com%",
    "%rapid.example%",
)


def purge_dummy_jobs(session: Session, ctx: Any | None = None) -> int:
    """Delete legacy dummy/placeholder job records and their vector embeddings.

    Removes every ``Job`` row whose ``job_url`` matches a placeholder pattern
    (``example.com`` or ``rapid.example``) together with every child row that
    references it (``JobMatch``, ``UpskillPlan``/``PlanItem``, ``Interview``),
    so PostgreSQL foreign-key constraints are never violated. When a context
    with a vector store is supplied, the corresponding per-job embeddings are
    dropped as well.

    Args:
        session: Active SQLModel database session.
        ctx: Optional application context exposing ``vector_store`` (and, when
            present, ``job_vectors`` cache) used to clean up embeddings.

    Returns:
        Number of purged job records.
    """
    jobs = list(session.exec(
        select(Job).where(
            Job.job_url.like(DUMMY_URL_PATTERNS[0])
            | Job.job_url.like(DUMMY_URL_PATTERNS[1])
        )
    ).all())
    purged_ids = [j.id for j in jobs if hasattr(j, "id")]
    if not purged_ids:
        return 0

    # Delete children before parents (FK-safe order): PlanItem -> UpskillPlan
    # -> Interview -> JobApplication -> JobMatch -> Job, all scoped to the purged job ids.
    sub_plans = select(UpskillPlan.id).where(UpskillPlan.job_id.in_(purged_ids))
    session.execute(delete(PlanItem).where(PlanItem.plan_id.in_(sub_plans)))
    session.execute(delete(UpskillPlan).where(UpskillPlan.job_id.in_(purged_ids)))
    session.execute(delete(Interview).where(Interview.job_id.in_(purged_ids)))
    session.execute(delete(JobApplication).where(JobApplication.job_id.in_(purged_ids)))
    session.execute(delete(JobMatch).where(JobMatch.job_id.in_(purged_ids)))
    session.execute(delete(Job).where(Job.id.in_(purged_ids)))
    session.commit()

    # Best-effort vector cleanup for the purged ids.
    vector_store = getattr(ctx, "vector_store", None) if ctx is not None else None
    job_vectors = getattr(ctx, "job_vectors", None) if ctx is not None else None
    if vector_store is not None:
        for job_id in purged_ids:
            try:
                remove = getattr(vector_store, "delete", None) or getattr(
                    vector_store, "remove", None
                )
                if callable(remove):
                    remove("job", job_id)
            except Exception as exc:  # noqa: BLE001 - vector cleanup must not block purge
                logger.warning("Failed to drop vector for purged job %s: %s", job_id, exc)
    if isinstance(job_vectors, dict):
        for job_id in purged_ids:
            job_vectors.pop(job_id, None)

    logger.info("Purged %d dummy job record(s) from the database", len(purged_ids))
    return len(purged_ids)

def purge_live_jobs(
    session: Session,
    ctx: Any | None = None,
    clear_output: bool = True,
) -> int:
    """Delete all live scraped jobs, related records, and vector embeddings (ARCHITECTURE §18.4).

    Institutional on-campus jobs (Job.source == "local") and student profiles/courses
    are protected and remain untouched. Only live/external jobs (Job.source != "local"
    or Job.id starting with "job-ext-") are removed.

    Cascades FK deletions safely in child-to-parent order:
    PlanItem -> UpskillPlan -> Interview -> JobMatch -> Job.

    When clear_output is True, output/raw_scraped_jobs.json and
    output/normalized_jobs.json are reset to empty JSON arrays.

    Args:
        session: Active SQLModel database session.
        ctx: Optional application context with vector_store and job_vectors.
        clear_output: Whether to reset output/ JSON cache files to '[]'.

    Returns:
        Number of purged live job records.
    """
    jobs = list(session.exec(
        select(Job).where(
            (Job.source != "local") | (Job.id.like("job-ext-%"))
        )
    ).all())
    purged_ids = [j.id for j in jobs if hasattr(j, "id")]
    if purged_ids:
        sub_plans = select(UpskillPlan.id).where(UpskillPlan.job_id.in_(purged_ids))
        session.execute(delete(PlanItem).where(PlanItem.plan_id.in_(sub_plans)))
        session.execute(delete(UpskillPlan).where(UpskillPlan.job_id.in_(purged_ids)))
        session.execute(delete(Interview).where(Interview.job_id.in_(purged_ids)))
        session.execute(delete(JobApplication).where(JobApplication.job_id.in_(purged_ids)))
        session.execute(delete(JobMatch).where(JobMatch.job_id.in_(purged_ids)))
        session.execute(delete(Job).where(Job.id.in_(purged_ids)))
        session.commit()

        # Best-effort vector store cleanup
        vector_store = getattr(ctx, "vector_store", None) if ctx is not None else None
        job_vectors = getattr(ctx, "job_vectors", None) if ctx is not None else None
        if vector_store is not None:
            for job_id in purged_ids:
                try:
                    remove = getattr(vector_store, "delete", None) or getattr(
                        vector_store, "remove", None
                    )
                    if callable(remove):
                        remove("job", job_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to drop vector for purged job %s: %s", job_id, exc)
        if isinstance(job_vectors, dict):
            for job_id in purged_ids:
                job_vectors.pop(job_id, None)

        logger.info("Purged %d live scraped job record(s) from the database", len(purged_ids))

    if clear_output:
        for out_file in (
            Path("output/raw_scraped_jobs.json"),
            Path("output/normalized_jobs.json"),
        ):
            try:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text("[]\n", encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to reset output cache %s: %s", out_file, exc)

    return len(purged_ids)


def sanitize_legacy_salaries(session: Session) -> int:
    """Reset legacy auto-filled [6, 12] rows to salary_disclosed=False and salary_lpa=[0, 0].

    Scans for jobs that have salary_lpa == [6, 12] without explicit salary disclosure,
    resetting them to [0, 0] and salary_disclosed=False so synthetic salary ranges
    are never surfaced to students (Milestone 8 Salary Integrity).

    Args:
        session: Active SQLModel database session.

    Returns:
        Number of sanitized job records.
    """
    try:
        jobs = list(session.exec(select(Job)).all())
        updated = 0
        for job in jobs:
            salary = getattr(job, "salary_lpa", None)
            disclosed = bool(getattr(job, "salary_disclosed", False))
            if salary == [6, 12] and not disclosed:
                job.salary_lpa = [0, 0]
                job.salary_disclosed = False
                session.add(job)
                updated += 1
            elif (not salary or salary == [0, 0]) and disclosed:
                job.salary_disclosed = False
                session.add(job)
                updated += 1
            elif (
                salary
                and len(salary) == 2
                and 0 < salary[0] <= 5
                and job.description
                and ("$" in job.description or "USD" in job.description)
            ):
                import re
                usd_match = re.search(
                    r"\$\s*([0-9,]+(?:\.[0-9]+)?)\s*(k)?\s*(?:-|–|—|to)\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\s*(k)?",
                    job.description,
                    re.IGNORECASE,
                )
                if usd_match:
                    def _pv(val: str, is_k: bool) -> float:
                        v = float(val.replace(",", ""))
                        return v * 1000 if (is_k or v < 1000) else v
                    lo_val = _pv(usd_match.group(1), bool(usd_match.group(2)))
                    hi_val = _pv(usd_match.group(3), bool(usd_match.group(4)))
                    if lo_val >= 20_000:
                        job.salary_lpa = [round(lo_val * 86.5 / 100_000), round(hi_val * 86.5 / 100_000)]
                        job.salary_disclosed = True
                        session.add(job)
                        updated += 1
        if updated > 0:
            session.commit()
            logger.info("Sanitized %d legacy salary records to undisclosed [0, 0]", updated)
        return updated
    except Exception as exc:  # noqa: BLE001 - sanitation must not crash caller
        logger.warning("Legacy salary sanitation failed: %s", exc)
        return 0


def sanitize_legacy_experience(session: Session, ctx: Any = None) -> int:
    """Repair legacy ``job-ext-%`` rows whose ``experience_years`` was left at 0 (ARCHITECTURE §20.2.4).

    Earlier static ingestion defaulted ``experience_years = 0`` without parsing the
    description, so historical jobs carrying "1-3 years" (etc.) in their text were
    mis-scored. This scans every job, extracts the true floor via
    ``extract_experience_years(title + description + job_url)``, enriches short
    LinkedIn postings via fast public detail fetches, and updates the field when > 0,
    then re-triggers match re-ranking for affected students so scores/eligibility
    reflect the corrected experience.

    Args:
        session: Active SQLModel database session.
        ctx: Application context (used by the post-update match re-ranking).

    Returns:
        Count of jobs whose ``experience_years`` was updated.
    """
    updated_ids: list[str] = []
    try:
        from app.matching.engine import FRESHER_TITLE_RE, MID_LEVEL_TITLE_RE
        from app.scraper.normalizer import fetch_external_job_description

        jobs = session.exec(select(Job)).all()
        for job in jobs:
            clean_body = (job.description or "").split("Apply URL:")[0].strip()
            if len(clean_body) < 60 and job.job_url and "linkedin.com/jobs/view/" in job.job_url:
                fetched = fetch_external_job_description(job.job_url, timeout=3.0)
                if fetched:
                    source_tag = getattr(job, "source", "linkedin") or "linkedin"
                    engine_val = getattr(job, "scraper_engine", "jobspy") or "jobspy"
                    job.description = (
                        f"{fetched}\n\nApply URL: {job.job_url}\n"
                        f"Source: {source_tag} (Scraped via {engine_val})"
                    ).strip()

            combined = f"{getattr(job, 'title', '') or ''} {getattr(job, 'description', '') or ''} {getattr(job, 'job_url', '') or ''}"
            extracted = extract_experience_years(combined)
            years = max(getattr(job, "experience_years", 0) or 0, extracted)
            if years <= 0 and is_senior_job(job):
                title_low = (getattr(job, 'title', '') or '').lower()
                if "iii" in title_low or " 3" in title_low or " iv" in title_low or " 4" in title_low or "specialist" in title_low:
                    years = 5
                else:
                    years = 3
            elif years <= 0 and MID_LEVEL_TITLE_RE.search(getattr(job, "title", "") or ""):
                years = 2
            if years <= 0 and (getattr(job, "job_url", "") or getattr(job, "source", "") != "local"):
                title_str = getattr(job, "title", "") or ""
                clean_len = len((getattr(job, "description", "") or "").split("Apply URL:")[0].strip())
                if clean_len < 50 and not FRESHER_TITLE_RE.search(title_str) and getattr(job, "source", "") not in ("internshala", "unstop"):
                    years = 2
                elif not FRESHER_TITLE_RE.search(title_str):
                    import re
                    m_rng = re.search(r'(?i)\b(\d+)[- ]*to[- ]*(\d+)[- ]*(?:years?|yrs?|yoe)\b', combined)
                    if m_rng:
                        try:
                            h = int(m_rng.group(2))
                            if h >= 2:
                                years = max(years, max(1, h // 2))
                        except (ValueError, TypeError):
                            pass

            if years > (job.experience_years or 0):
                job.experience_years = years
                session.add(job)
                updated_ids.append(job.id)
        if updated_ids:
            session.commit()
            logger.info(
                "Sanitized experience for %d legacy job-ext rows: %s",
                len(updated_ids), ", ".join(updated_ids[:10]),
            )
    except Exception as exc:  # noqa: BLE001 - sanitation must not crash caller
        logger.warning("Legacy experience sanitation failed: %s", exc)
        session.rollback()
        return 0

    # Re-rank affected students so corrected experience flows into scores/eligibility.
    if updated_ids and ctx is not None:
        try:
            from app.api.routes.students import _rank_and_persist
            from app.models import Student

            for student in session.exec(select(Student)).all():
                _rank_and_persist(session, ctx, student.id, student, include_scraped=True)
        except Exception as rank_err:  # noqa: BLE001 - sanitation must not crash caller
            logger.warning("Post-sanitation match re-ranking failed: %s", rank_err)
    return len(updated_ids)


def sanitize_legacy_skills(session: Session, ctx: Any | None = None) -> int:
    """Repair legacy jobs whose skills were bloated with taxonomy dumps or ungrounded skills.

    Scans every job, verifying skills against the taxonomy and job text:
    - Leaves hand-curated local jobs (source == 'local') alone if skills count <= 6.
    - required_skills: valid taxonomy skills, capped at 6, minimum 2 (using CORE_FALLBACK_SKILLS).
    - preferred_skills: valid taxonomy skills mentioned in job text, excluding required, capped at 6.

    Args:
        session: Active SQLModel database session.
        ctx: Application context (used for taxonomy and post-update match re-ranking).

    Returns:
        Count of sanitized jobs.
    """
    taxonomy = getattr(ctx, "taxonomy", None) or load_taxonomy()
    updated_ids: list[str] = []

    try:
        jobs = list(session.exec(select(Job)).all())
        for job in jobs:
            # Leave curated institutional jobs untouched if not bloated
            is_local = getattr(job, "source", "local") == "local"
            orig_req = list(job.required_skills or [])
            orig_pref = list(job.preferred_skills or [])

            if is_local and len(orig_req) <= 6 and len(orig_pref) <= 6:
                continue

            clean_body = (job.description or "").split("Apply URL:")[0].strip()
            corpus = f"{job.title or ''} {clean_body}".lower()

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

            # Sanitize required_skills
            valid_required: list[str] = []
            for s in orig_req:
                s_norm = str(s).strip().lower()
                resolved = taxonomy.resolve(s_norm) or (s_norm if taxonomy.get_opt(s_norm) else None)
                if resolved and resolved not in valid_required:
                    valid_required.append(resolved)

            valid_required = valid_required[:6]
            if len(valid_required) < 2:
                for fallback in CORE_FALLBACK_SKILLS:
                    if fallback not in valid_required:
                        valid_required.append(fallback)
                    if len(valid_required) >= 2:
                        break

            # Sanitize preferred_skills: must be in taxonomy, in text, not in required, capped at 6
            valid_preferred: list[str] = []
            for s in orig_pref:
                s_norm = str(s).strip().lower()
                resolved = taxonomy.resolve(s_norm) or (s_norm if taxonomy.get_opt(s_norm) else None)
                if (
                    resolved
                    and resolved not in valid_preferred
                    and resolved not in valid_required
                    and _skill_in_text(resolved)
                ):
                    valid_preferred.append(resolved)

            valid_preferred = valid_preferred[:6]

            if job.required_skills != valid_required or job.preferred_skills != valid_preferred:
                job.required_skills = valid_required
                job.preferred_skills = valid_preferred
                session.add(job)
                updated_ids.append(job.id)

        if updated_ids:
            session.commit()
            logger.info("Sanitized skills for %d legacy job rows", len(updated_ids))
    except Exception as exc:  # noqa: BLE001 - sanitation must not crash caller
        logger.warning("Legacy skill sanitation failed: %s", exc)
        session.rollback()
        return 0

    if updated_ids and ctx is not None:
        try:
            from app.api.routes.students import _rank_and_persist
            from app.models import Student

            for student in session.exec(select(Student)).all():
                _rank_and_persist(session, ctx, student.id, student, include_scraped=True)
        except Exception as rank_err:  # noqa: BLE001 - sanitation must not crash caller
            logger.warning("Post-sanitation skill match re-ranking failed: %s", rank_err)

    return len(updated_ids)


def index_job_vectors(jobs: list, vector_store, embedder,
                      job_vectors: dict | None = None) -> int:
    """Embed each job's description and index it into the vector store.

    Prefers a single batched ``upsert_many`` (M21) when the store exposes it,
    falling back to per-row ``upsert`` for legacy stores. Embedding failures are
    isolated per job so one bad description never aborts the batch. Returns the
    number of jobs successfully indexed.
    """
    items: list[tuple[str, str, list[float]]] = []
    for job in jobs:
        try:
            vec = embedder.embed(job.description)
            items.append(("job", job.id, vec))
            if isinstance(job_vectors, dict):
                job_vectors[job.id] = vec
        except Exception as exc:
            logger.warning("Failed to embed vector for job %s: %s", job.id, exc)

    if not items:
        return 0
    if callable(getattr(vector_store, "upsert_many", None)):
        try:
            vector_store.upsert_many(items)
        except Exception as exc:
            logger.warning("Batch vector upsert failed, falling back to rows: %s", exc)
            for kind, jid, vec in items:
                try:
                    vector_store.upsert(kind, jid, vec)
                except Exception as row_exc:  # noqa: BLE001
                    logger.warning("Failed to index dense vector for job %s: %s", jid, row_exc)
            return len(items)
        return len(items)

    for kind, jid, vec in items:
        try:
            vector_store.upsert(kind, jid, vec)
        except Exception as exc:
            logger.warning("Failed to index dense vector for job %s: %s", jid, exc)
    return len(items)


def ingest_scraped_jobs(
    raw_jobs: list[dict[str, Any]],
    session: Session,
    ctx: Any,
    taxonomy: Taxonomy | None = None,
) -> dict[str, Any]:
    """Normalize raw scraped jobs, merge into database idempotently, and index in vector store.

    Normalization runs concurrently (ARCHITECTURE §18.2.B.1): when ``ctx.llm``
    is configured the batch is normalized through ``normalize_job_with_ai`` on a
    ``ThreadPoolExecutor`` of ``min(10, len(raw_jobs))`` workers, otherwise the
    static ``normalize_scraped_job`` is used inline. Only normalization is
    parallel — every DB write happens on the calling thread, so the session is
    never touched concurrently and the batch stays transactional.

    Deduplicates against existing database rows and across batch records by:
    1. Exact canonical application URL (tracking parameters stripped).
    2. Normalized (title, company) tuple.

    A record that fails to normalize is logged and skipped; it never aborts the
    batch.

    Args:
        raw_jobs: List of raw scraped job dictionaries.
        session: Active SQLModel database session.
        ctx: Application context containing taxonomy, vector_store, and embedder
            (and optionally ``llm`` for AI normalization).
        taxonomy: Optional pre-loaded Taxonomy instance.

    Returns:
        dict containing 'ingested_count', 'job_ids', and 'jobs'.
    """
    tax = taxonomy or getattr(ctx, "taxonomy", None) or load_taxonomy()

    # One-time sanitation pass: drop legacy placeholder-domain records before
    # merging new postings so dummy links never resurface (ARCHITECTURE §12.3).
    # Skipped for in-memory/mock sessions (no persistent dummy rows to clean).
    if type(session).__module__.split(".")[0] != "unittest":
        purge_dummy_jobs(session, ctx)
        sanitize_legacy_salaries(session)
        sanitize_legacy_experience(session, ctx)
        sanitize_legacy_skills(session, ctx)

    # Pre-index existing database records for cross-provider dedup
    existing_jobs = list(session.exec(select(Job)).all())
    url_lookup: dict[str, Job] = {
        _clean_url(j.job_url): j for j in existing_jobs if j.job_url
    }
    key_lookup: dict[tuple[str, str], Job] = {
        _clean_key(j.title, j.company): j
        for j in existing_jobs
        if _clean_key(j.title, j.company) != ("", "")
    }

    ingested_by_id: dict[str, Job] = {}

    # §18.2.B.1 — concurrent batch normalization.
    # With an LLM configured, normalize 20-50 records in parallel (LLM calls are
    # I/O-bound, so threads give near-linear speedup); without one, the static
    # normalizer is cheap and runs inline. Either way, DB writes stay on this
    # thread: the pool returns Job objects only, so the session is never touched
    # concurrently and transactional integrity is preserved.
    valid_raw = [r for r in raw_jobs if r and isinstance(r, dict)]
    llm = getattr(ctx, "llm", None) if ctx is not None else None
    is_mock = type(llm).__name__ == "MockLLM" if llm is not None else False
    if llm is not None and valid_raw and (is_mock or not is_ai_normalizer_circuit_open()):
        max_workers = min(5, max(1, len(valid_raw)))
        normalized: list[Job] = []
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = []
                for raw in valid_raw:
                    if not is_mock and is_ai_normalizer_circuit_open():
                        normalized.append(normalize_scraped_job(raw, tax))
                    else:
                        futures.append(pool.submit(normalize_job_with_ai, raw, tax, llm))
                for f in futures:
                    normalized.append(f.result())
        except Exception as exc:  # noqa: BLE001 - never drop the batch on pool failure
            logger.warning("Concurrent AI normalization failed (%s); normalizing inline", exc)
            normalized = [normalize_scraped_job(raw, tax) for raw in valid_raw]
    else:
        normalized = [normalize_scraped_job(raw, tax) for raw in valid_raw]

    for job in normalized:
        raw = next(
            (r for r in valid_raw
             if str(r.get("job_url") or r.get("url") or "").strip()
             == str(getattr(job, "job_url", "") or "").strip()),
            None,
        )
        try:
            canon_url = _clean_url(job.job_url) if job.job_url else ""
            if canon_url:
                job.job_url = canon_url
            key = _clean_key(job.title, job.company)

            matched_existing: Job | None = None
            if canon_url and canon_url in url_lookup:
                matched_existing = url_lookup[canon_url]
            elif key != ("", "") and key in key_lookup:
                matched_existing = key_lookup[key]

            if matched_existing is not None:
                # Inherit existing primary key to update record in-place
                job.id = matched_existing.id
                # Keep richer description if existing is longer
                if len(matched_existing.description) > len(job.description):
                    job.description = matched_existing.description
                if matched_existing.job_url:
                    job.job_url = matched_existing.job_url

            merged_job = session.merge(job)
            job_id = merged_job.id if isinstance(getattr(merged_job, "id", None), str) else job.id
            ingested_by_id[job_id] = merged_job
            if canon_url:
                url_lookup[canon_url] = merged_job
            if key != ("", ""):
                key_lookup[key] = merged_job
        except Exception as exc:
            label = (raw or {}).get("title") if raw else None
            logger.warning("Failed to merge job record %r: %s", label, exc)
            continue

    # Commit merged SQLModel entities
    session.commit()
    # Sanitize experience on newly ingested rows and re-rank matches immediately
    if type(session).__module__.split(".")[0] != "unittest":
        sanitize_legacy_experience(session, ctx)
    jobs = list(ingested_by_id.values())
    if jobs:
        try:
            from app.security.audit import log_audit
            log_audit(session, action="jobs.scraped", entity_type="job",
                      entity_id=",".join(j.id for j in jobs[:20]),
                      actor_role="system",
                      detail={"count": len(jobs)},
                      commit=True)
        except Exception:  # pragma: no cover - audit must never break ingestion
            logger.warning("failed to write jobs.scraped audit row", exc_info=True)

    # Index into vector store if available
    vector_store = getattr(ctx, "vector_store", None)
    embedder = getattr(ctx, "embedder", None)
    job_vectors = getattr(ctx, "job_vectors", None)

    if vector_store and embedder:
        index_job_vectors(jobs, vector_store, embedder, job_vectors)

    return {
        "ingested_count": len(jobs),
        "job_ids": [j.id for j in jobs],
        "jobs": jobs,
    }


def ingest_from_output_folder(
    output_dir: str | Path = "output",
    session: Session | None = None,
    ctx: Any | None = None,
    taxonomy: Taxonomy | None = None,
) -> dict[str, Any]:
    """Batch-ingest stealth-harvester output files into the database and vector index.

    Reads ``<output_dir>/normalized_jobs.json`` (preferred) or
    ``<output_dir>/raw_scraped_jobs.json`` (fallback) and feeds every record
    through :func:`ingest_scraped_jobs`, so records are deduplicated against
    existing database rows (canonical ``job_url`` first, normalized
    ``(title, company)`` second) and merged idempotently — re-ingesting the
    same folder never duplicates rows.

    Args:
        output_dir: Directory written by ``job_scraper.pipeline.run_pipeline``.
        session: Active SQLModel database session (required for ingestion).
        ctx: Application context with taxonomy, vector_store, and embedder.
        taxonomy: Optional pre-loaded Taxonomy instance.

    Returns:
        The :func:`ingest_scraped_jobs` result dict plus ``source`` (the file
        that was read, or ``None`` when no output file exists or ``session``
        is missing) and ``output_dir``. Never raises for a missing folder or
        unreadable JSON.
    """
    out = Path(output_dir)
    normalized_path = out / "normalized_jobs.json"
    raw_path = out / "raw_scraped_jobs.json"
    if normalized_path.is_file():
        path = normalized_path
    elif raw_path.is_file():
        path = raw_path
    else:
        logger.warning("No stealth harvester output found under %s", out)
        return {
            "ingested_count": 0,
            "job_ids": [],
            "jobs": [],
            "source": None,
            "output_dir": str(out),
        }

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read stealth harvester output %s: %s", path, exc)
        return {
            "ingested_count": 0,
            "job_ids": [],
            "jobs": [],
            "source": None,
            "output_dir": str(out),
        }

    records = [r for r in (data if isinstance(data, list) else []) if isinstance(r, dict)]
    if session is None:
        logger.warning("ingest_from_output_folder called without a session; ingesting 0 of %d records", len(records))
        return {
            "ingested_count": 0,
            "job_ids": [],
            "jobs": [],
            "source": str(path),
            "output_dir": str(out),
        }

    result = ingest_scraped_jobs(records, session, ctx, taxonomy)
    result["source"] = str(path)
    result["output_dir"] = str(out)
    return result
