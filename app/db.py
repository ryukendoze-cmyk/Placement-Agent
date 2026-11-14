import hashlib
import json
import logging
import re

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

logger = logging.getLogger(__name__)


def make_engine(url: str | None = None) -> Engine:
    if url is None:
        from .config import settings
        url = settings.database_url
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)


def init_db(engine: Engine) -> None:
    from . import models  # noqa: F401  (register tables)
    SQLModel.metadata.create_all(engine)
    _ensure_columns(engine)
    _normalize_legacy_branches(engine)
    _ensure_indexes(engine)
    _purge_dummy_jobs(engine)
    _sanitize_legacy_salaries(engine)
    _sanitize_legacy_experience(engine)
    _sanitize_legacy_skills(engine)
    _recalibrate_legacy_progress_events(engine)
    _scrub_legacy_audit_emails(engine)
    install_audit_guard(engine)


# Canonical SQLAlchemy index names (ix_<table>_<column>) for high-traffic FK /
# filter columns. Declared via index=True on the models for fresh DBs (create_all)
# and re-asserted here with CREATE INDEX IF NOT EXISTS so already-seeded
# databases gain the indexes without dropping/re-creating tables (M21).
_ENSURE_INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS ix_jobmatch_student_id ON jobmatch (student_id)",
    "CREATE INDEX IF NOT EXISTS ix_jobmatch_job_id ON jobmatch (job_id)",
    "CREATE INDEX IF NOT EXISTS ix_upskillplan_student_id ON upskillplan (student_id)",
    "CREATE INDEX IF NOT EXISTS ix_upskillplan_job_id ON upskillplan (job_id)",
    "CREATE INDEX IF NOT EXISTS ix_planitem_plan_id ON planitem (plan_id)",
    "CREATE INDEX IF NOT EXISTS ix_planitem_skill_id ON planitem (skill_id)",
    "CREATE INDEX IF NOT EXISTS ix_progressevent_student_id ON progressevent (student_id)",
    "CREATE INDEX IF NOT EXISTS ix_interview_student_id ON interview (student_id)",
    "CREATE INDEX IF NOT EXISTS ix_interview_job_id ON interview (job_id)",
    "CREATE INDEX IF NOT EXISTS ix_approvalrequest_status ON approvalrequest (status)",
    "CREATE INDEX IF NOT EXISTS ix_auditlog_action ON auditlog (action)",
    "CREATE INDEX IF NOT EXISTS ix_auditlog_created_at ON auditlog (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_job_owner_student_id ON job (owner_student_id)",
    "CREATE INDEX IF NOT EXISTS ix_job_is_live ON job (is_live)",
    "CREATE INDEX IF NOT EXISTS ix_jobmatch_is_favorite ON jobmatch (is_favorite)",
    "CREATE INDEX IF NOT EXISTS ix_jobapplication_is_favorite ON jobapplication (is_favorite)",
]


def _ensure_indexes(engine: Engine) -> None:
    """Create high-traffic indexes idempotently (SQLite + PostgreSQL).

    Uses ``CREATE INDEX IF NOT EXISTS`` (supported by both dialects) with the
    exact names SQLAlchemy auto-generates for ``index=True`` columns, so a fresh
    database gets each index exactly once and a legacy database gains them on
    startup.
    """
    try:
        with engine.begin() as conn:
            for stmt in _ENSURE_INDEX_STATEMENTS:
                conn.execute(text(stmt))
    except Exception as exc:  # noqa: BLE001 - index DDL must never block boot
        logger.warning("Index assurance failed: %s", exc)


def _purge_dummy_jobs(engine: Engine) -> int:
    """One-time sanitation: delete legacy placeholder-domain job rows (§12.3).

    Cascades through child tables in FK-safe order (``planitem`` ->
    ``upskillplan`` -> ``interview`` -> ``jobmatch`` -> ``job``) so PostgreSQL
    foreign-key constraints are never violated, then deletes the dummy ``job``
    rows themselves. Runs as raw SQL on any engine (SQLite in tests, PostgreSQL
    in production) and is safe to call on every startup (idempotent — returns 0
    once the table is clean).
    """
    dummy_job_ids = (
        "SELECT id FROM job WHERE job_url LIKE '%example.com%' "
        "OR job_url LIKE '%rapid.example%'"
    )
    statements = [
        text(
            "DELETE FROM planitem WHERE plan_id IN "
            f"(SELECT id FROM upskillplan WHERE job_id IN ({dummy_job_ids}))"
        ),
        text(
            "DELETE FROM upskillplan WHERE job_id IN "
            f"({dummy_job_ids})"
        ),
        text(
            "DELETE FROM interview WHERE job_id IN "
            f"({dummy_job_ids})"
        ),
        text(
            "DELETE FROM jobmatch WHERE job_id IN "
            f"({dummy_job_ids})"
        ),
        text(
            "DELETE FROM jobapplication WHERE job_id IN "
            f"({dummy_job_ids})"
        ),
        text(
            "DELETE FROM job WHERE job_url LIKE '%example.com%' "
            "OR job_url LIKE '%rapid.example%'"
        ),
    ]
    try:
        purged = 0
        with engine.begin() as conn:
            for stmt in statements:
                result = conn.execute(stmt)
                if stmt is statements[-1]:
                    purged = int(result.rowcount or 0)
        return purged
    except Exception as exc:  # noqa: BLE001 - sanitation must never block boot
        logger.warning("Dummy job purge failed: %s", exc)
        return 0


def _ensure_columns(engine: Engine) -> None:
    """create_all never ALTERs existing tables; add new columns on Postgres and SQLite
    so an already-seeded DB picks up schema additions without re-creating."""
    from sqlalchemy import text

    if engine.dialect.name == "postgresql":
        statements = [
            "ALTER TABLE student ADD COLUMN IF NOT EXISTS resume_text TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE student ADD COLUMN IF NOT EXISTS backlogs INT NOT NULL DEFAULT 0",
            "ALTER TABLE student ADD COLUMN IF NOT EXISTS pii_redacted_count INT NOT NULL DEFAULT 0",
            "ALTER TABLE student ADD COLUMN IF NOT EXISTS resume_reupload_unlocked BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS min_gpa FLOAT NOT NULL DEFAULT 0.0",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS allowed_branches JSON NOT NULL DEFAULT '[]'::json",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS job_url TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'local'",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS scraper_engine TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS salary_disclosed BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS max_backlogs INT DEFAULT NULL",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS is_live BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS owner_student_id VARCHAR DEFAULT NULL",
            "ALTER TABLE jobmatch ADD COLUMN IF NOT EXISTS is_favorite BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE jobapplication ADD COLUMN IF NOT EXISTS is_favorite BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE agentrun ADD COLUMN IF NOT EXISTS citations JSON NOT NULL DEFAULT '{}'::json",
            "ALTER TABLE agentstep ADD COLUMN IF NOT EXISTS verification_passed BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE agentstep ADD COLUMN IF NOT EXISTS error_recovered TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE planitem ADD COLUMN IF NOT EXISTS week INT NOT NULL DEFAULT 1",
            "ALTER TABLE interview ADD COLUMN IF NOT EXISTS role TEXT DEFAULT NULL",
            "ALTER TABLE interview ADD COLUMN IF NOT EXISTS include_resume_probes BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE interview ADD COLUMN IF NOT EXISTS adaptive_followup BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE approvalrequest ADD COLUMN IF NOT EXISTS endorsed_by TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE approvalrequest ADD COLUMN IF NOT EXISTS endorsed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL",
            "ALTER TABLE studentskill ADD COLUMN IF NOT EXISTS verification_status TEXT NOT NULL DEFAULT 'claimed'",
            "ALTER TABLE studentskill ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5",
            "ALTER TABLE studentskill ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'resume'",
            "ALTER TABLE studentskill ADD COLUMN IF NOT EXISTS last_verified_score INTEGER DEFAULT NULL",
            "ALTER TABLE studentskill ADD COLUMN IF NOT EXISTS last_verified_at TIMESTAMP WITH TIME ZONE DEFAULT NULL",
        ]
        with engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
    elif engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            try:
                cols = [
                    row[1] for row in conn.execute(text("PRAGMA table_info(job)")).fetchall()
                ]
                if cols and "salary_disclosed" not in cols:
                    conn.execute(
                        text("ALTER TABLE job ADD COLUMN salary_disclosed BOOLEAN DEFAULT 0")
                    )
                if cols and "max_backlogs" not in cols:
                    conn.execute(
                        text("ALTER TABLE job ADD COLUMN max_backlogs INTEGER DEFAULT NULL")
                    )
                if cols and "scraper_engine" not in cols:
                    conn.execute(
                        text("ALTER TABLE job ADD COLUMN scraper_engine TEXT NOT NULL DEFAULT ''")
                    )
                if cols and "is_live" not in cols:
                    conn.execute(
                        text("ALTER TABLE job ADD COLUMN is_live BOOLEAN NOT NULL DEFAULT 0")
                    )
                if cols and "owner_student_id" not in cols:
                    conn.execute(
                        text("ALTER TABLE job ADD COLUMN owner_student_id TEXT DEFAULT NULL")
                    )
                jm_cols = [
                    row[1] for row in conn.execute(text("PRAGMA table_info(jobmatch)")).fetchall()
                ]
                if jm_cols and "is_favorite" not in jm_cols:
                    conn.execute(
                        text("ALTER TABLE jobmatch ADD COLUMN is_favorite BOOLEAN NOT NULL DEFAULT 0")
                    )
                ja_cols = [
                    row[1] for row in conn.execute(text("PRAGMA table_info(jobapplication)")).fetchall()
                ]
                if ja_cols and "is_favorite" not in ja_cols:
                    conn.execute(
                        text("ALTER TABLE jobapplication ADD COLUMN is_favorite BOOLEAN NOT NULL DEFAULT 0")
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("SQLite column migration failed: %s", exc)
            # M18.1/18.2: interview feature flags on legacy SQLite DBs.
            try:
                iv_cols = [
                    row[1] for row in
                    conn.execute(text("PRAGMA table_info(interview)")).fetchall()
                ]
                if iv_cols and "role" not in iv_cols:
                    conn.execute(text("ALTER TABLE interview ADD COLUMN role TEXT DEFAULT NULL"))
                if iv_cols and "include_resume_probes" not in iv_cols:
                    conn.execute(text(
                        "ALTER TABLE interview ADD COLUMN include_resume_probes BOOLEAN DEFAULT 0"))
                if iv_cols and "adaptive_followup" not in iv_cols:
                    conn.execute(text(
                        "ALTER TABLE interview ADD COLUMN adaptive_followup BOOLEAN DEFAULT 0"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("SQLite interview column migration failed: %s", exc)
            # Skill verification columns on legacy SQLite DBs.
            try:
                ss_cols = [
                    row[1] for row in
                    conn.execute(text("PRAGMA table_info(studentskill)")).fetchall()
                ]
                if ss_cols and "verification_status" not in ss_cols:
                    conn.execute(text(
                        "ALTER TABLE studentskill ADD COLUMN verification_status TEXT NOT NULL DEFAULT 'claimed'"
                    ))
                if ss_cols and "confidence" not in ss_cols:
                    conn.execute(text(
                        "ALTER TABLE studentskill ADD COLUMN confidence REAL NOT NULL DEFAULT 0.5"
                    ))
                if ss_cols and "source" not in ss_cols:
                    conn.execute(text(
                        "ALTER TABLE studentskill ADD COLUMN source TEXT NOT NULL DEFAULT 'resume'"
                    ))
                if ss_cols and "last_verified_score" not in ss_cols:
                    conn.execute(text(
                        "ALTER TABLE studentskill ADD COLUMN last_verified_score INTEGER DEFAULT NULL"
                    ))
                if ss_cols and "last_verified_at" not in ss_cols:
                    conn.execute(text(
                        "ALTER TABLE studentskill ADD COLUMN last_verified_at TIMESTAMP DEFAULT NULL"
                    ))
            except Exception as exc:  # noqa: BLE001
                logger.warning("SQLite studentskill column migration failed: %s", exc)
            # Student resume re-upload unlock flag on legacy SQLite DBs.
            try:
                stu_cols = [
                    row[1] for row in
                    conn.execute(text("PRAGMA table_info(student)")).fetchall()
                ]
                if stu_cols and "resume_reupload_unlocked" not in stu_cols:
                    conn.execute(text(
                        "ALTER TABLE student ADD COLUMN resume_reupload_unlocked BOOLEAN NOT NULL DEFAULT 0"
                    ))
            except Exception as exc:  # noqa: BLE001
                logger.warning("SQLite student column migration failed: %s", exc)
            # M19: approvalrequest endorsement columns on legacy SQLite DBs.
            try:
                ap_cols = [
                    row[1] for row in
                    conn.execute(text("PRAGMA table_info(approvalrequest)")).fetchall()
                ]
                if ap_cols and "endorsed_by" not in ap_cols:
                    conn.execute(text(
                        "ALTER TABLE approvalrequest ADD COLUMN endorsed_by TEXT NOT NULL DEFAULT ''"))
                if ap_cols and "endorsed_at" not in ap_cols:
                    conn.execute(text(
                        "ALTER TABLE approvalrequest ADD COLUMN endorsed_at TIMESTAMP"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("SQLite approvalrequest column migration failed: %s", exc)


def _normalize_legacy_branches(engine: Engine) -> int:
    """Idempotently normalize legacy Student.branch strings into canonical JKLU codes."""
    try:
        from sqlmodel import Session, select

        from .models import Student
        from .taxonomy.branches import normalize_branch

        updated = 0
        with Session(engine) as session:
            students = session.exec(select(Student)).all()
            for stu in students:
                raw_branch = getattr(stu, "branch", "") or ""
                norm = normalize_branch(raw_branch)
                if norm != raw_branch:
                    stu.branch = norm
                    session.add(stu)
                    updated += 1
            if updated > 0:
                session.commit()
                logger.info("Normalized canonical branches for %d legacy student rows", updated)
        return updated
    except Exception as exc:  # noqa: BLE001 - normalization must never block boot
        logger.warning("Legacy branch normalization failed: %s", exc)
        return 0


def _sanitize_legacy_salaries(engine: Engine) -> int:
    """Reset legacy [6, 12] auto-filled salaries and repair compressed USD salaries."""
    try:
        from sqlmodel import Session

        from app.scraper.pipeline import sanitize_legacy_salaries as _pipeline_sanitize
        with Session(engine) as s:
            return _pipeline_sanitize(s)
    except Exception as exc:  # noqa: BLE001 - sanitation must never block boot
        logger.warning("Legacy salary sanitation failed: %s", exc)
        return 0


def sanitize_legacy_salaries(session_or_engine) -> int:
    """Reset legacy auto-filled [6, 12] rows to salary_disclosed=False and salary_lpa=[0, 0].
    Accepts either an active SQLModel Session or a SQLAlchemy Engine."""
    from sqlmodel import Session
    if isinstance(session_or_engine, Session):
        from app.scraper.pipeline import sanitize_legacy_salaries as _pipeline_sanitize
        return _pipeline_sanitize(session_or_engine)
    return _sanitize_legacy_salaries(session_or_engine)




def _sanitize_legacy_experience(engine: Engine) -> int:
    """Sanitize legacy jobs whose experience_years was left at 0 but title/desc is senior or experienced."""
    try:
        from sqlmodel import Session, select

        from app.matching.engine import (
            FRESHER_TITLE_RE,
            MID_LEVEL_TITLE_RE,
            is_senior_job,
        )
        from app.models import Job
        from app.scraper.normalizer import (
            extract_experience_years,
            fetch_external_job_description,
        )

        updated = 0
        with Session(engine) as session:
            jobs = session.exec(select(Job)).all()
            for job in jobs:
                # If short LinkedIn description, attempt one-time fast enrichment
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

                combined = f"{job.title} {job.description} {job.job_url}"
                extracted = extract_experience_years(combined)
                years = max(getattr(job, "experience_years", 0) or 0, extracted)
                if years <= 0 and is_senior_job(job):
                    title_low = job.title.lower()
                    if "iii" in title_low or " 3" in title_low or " iv" in title_low or " 4" in title_low or "specialist" in title_low:
                        years = 5
                    else:
                        years = 3
                elif years <= 0 and MID_LEVEL_TITLE_RE.search(job.title):
                    years = 2
                if years <= 0 and (job.job_url or getattr(job, "source", "") != "local"):
                    title_str = getattr(job, "title", "") or ""
                    clean_len = len((job.description or "").split("Apply URL:")[0].strip())
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
                    updated += 1
            if updated > 0:
                session.commit()
                logger.info("Sanitized experience for %d legacy job rows", updated)
        return updated
    except Exception as exc:  # noqa: BLE001 - sanitation must never block boot
        logger.warning("Legacy experience sanitation failed: %s", exc)
        return 0


def _sanitize_legacy_skills(engine: Engine) -> int:
    """Sanitize legacy jobs with bloated or ungrounded skills."""
    try:
        from sqlmodel import Session

        from app.scraper.pipeline import sanitize_legacy_skills as _pipeline_sanitize
        with Session(engine) as s:
            return _pipeline_sanitize(s)
    except Exception as exc:  # noqa: BLE001 - sanitation must never block boot
        logger.warning("Legacy skills sanitation failed: %s", exc)
        return 0


def sanitize_legacy_skills(session_or_engine, ctx=None) -> int:
    """Sanitize legacy jobs with bloated or ungrounded skills.
    Accepts either an active SQLModel Session or a SQLAlchemy Engine."""
    from sqlmodel import Session
    if isinstance(session_or_engine, Session):
        from app.scraper.pipeline import sanitize_legacy_skills as _pipeline_sanitize
        return _pipeline_sanitize(session_or_engine, ctx)
    return _sanitize_legacy_skills(session_or_engine)


def _recalibrate_legacy_progress_events(engine: Engine) -> int:
    """Recalibrate legacy [20, 34, 45, 58, 67, 75] or 4h-compressed ProgressEvent clusters."""
    try:
        from datetime import datetime, timedelta, timezone

        from sqlmodel import Session, select

        from .analytics.student_progress import (
            _SYNTHETIC_LEGACY_LADDER,
            _as_utc,
            _is_job_visible_to_student,
            compute_student_readiness,
        )
        from .models import Job, JobMatch, PlanItem, ProgressEvent, UpskillPlan

        recalibrated = 0
        now = datetime.now(timezone.utc)
        with Session(engine) as session:
            all_events = list(session.exec(select(ProgressEvent)).all())
            by_student: dict[str, list[ProgressEvent]] = {}
            for ev in all_events:
                by_student.setdefault(ev.student_id, []).append(ev)

            if not by_student:
                return 0

            jobs_by_id = {j.id: j for j in session.exec(select(Job)).all()}
            for sid, evs in by_student.items():
                ordered = sorted(evs, key=lambda e: (_as_utc(e.at), str(e.id)))
                readiness_seq = [int(e.readiness) for e in ordered]
                legacy_n = len(_SYNTHETIC_LEGACY_LADDER)
                has_legacy_prefix = (
                    len(readiness_seq) >= legacy_n
                    and readiness_seq[:legacy_n] == _SYNTHETIC_LEGACY_LADDER
                )
                span_sec = (
                    (_as_utc(ordered[-1].at) - _as_utc(ordered[0].at)).total_seconds()
                    if len(ordered) >= 2
                    else 0.0
                )
                is_compressed = len(ordered) >= 4 and span_sec <= 4 * 3600
                has_compressed_burst = (
                    len(ordered) >= 4
                    and (_as_utc(ordered[3].at) - _as_utc(ordered[0].at)).total_seconds() <= 4 * 3600
                    and all(e.ref_id is None for e in ordered[:4])
                )
                if not (has_legacy_prefix or is_compressed or has_compressed_burst):
                    continue

                matches = [
                    m for m in session.exec(select(JobMatch).where(JobMatch.student_id == sid)).all()
                    if getattr(m, "band", None) != "dismissed"
                    and int(getattr(m, "score", 0) or 0) >= 0
                    and not (isinstance(m.breakdown, dict) and m.breakdown.get("dismissed"))
                    and _is_job_visible_to_student(jobs_by_id.get(m.job_id), sid)
                ]
                best_match = max((int(m.score) for m in matches), default=0)

                plans = list(
                    session.exec(
                        select(UpskillPlan)
                        .where(UpskillPlan.student_id == sid)
                        .order_by(UpskillPlan.created_at.desc())
                    ).all()
                )
                items: list[PlanItem] = []
                if plans:
                    items = list(session.exec(select(PlanItem).where(PlanItem.plan_id == plans[0].id)).all())
                done_items = sum(1 for it in items if str(it.status or "").lower() == "done")
                target = max(0, min(98, compute_student_readiness(best_match, done_items, len(items))))
                baseline = min(target, round(target * 0.58)) if target > 0 else 0
                if baseline == target and target > 1:
                    baseline = max(0, target - 6)

                n = len(ordered)
                # Preserve real multi-day timestamps after a synthetic prefix; spread only the synthetic part.
                if is_compressed or len(ordered) == legacy_n and has_legacy_prefix:
                    prefix_len = n
                elif has_legacy_prefix:
                    prefix_len = legacy_n
                else:
                    prefix_len = 0
                    for idx_b, ev_b in enumerate(ordered):
                        if ev_b.ref_id is None and (_as_utc(ev_b.at) - _as_utc(ordered[0].at)).total_seconds() <= 4 * 3600:
                            prefix_len = idx_b + 1
                        else:
                            break
                    prefix_len = max(4, prefix_len)

                anchor_end = (
                    _as_utc(ordered[prefix_len].at) - timedelta(days=1)
                    if prefix_len < n
                    else now
                )
                prev_val = baseline
                for idx, ev in enumerate(ordered):
                    if idx < prefix_len:
                        offset_days = round(14 * (prefix_len - 1 - idx) / max(1, prefix_len - 1))
                        ev.at = anchor_end - timedelta(days=offset_days)
                    if idx == n - 1:
                        val = target
                    else:
                        val = max(prev_val, round(baseline + (target - baseline) * idx / max(1, n - 1)))
                    prev_val = val
                    ev.readiness = int(val)
                    session.add(ev)
                recalibrated += 1

            if recalibrated > 0:
                session.commit()
                logger.info("Recalibrated legacy progress trajectories for %d students", recalibrated)
        return recalibrated
    except Exception as exc:  # noqa: BLE001 - boot recalibration must never block startup
        logger.warning("Legacy progress recalibration failed: %s", exc)
        return 0


_AUDIT_EMAIL_RE = re.compile(r"[A-Za-z0-9._%*+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _pseudonymize_email_token(email: str) -> str:
    norm = email.strip().lower()
    if not norm:
        return "anonymous"
    return f"id-{hashlib.sha256(norm.encode('utf-8')).hexdigest()[:8]}"


def _scrub_audit_string(value: str) -> str:
    return _AUDIT_EMAIL_RE.sub(lambda m: _pseudonymize_email_token(m.group(0)), value)


def _scrub_audit_obj(value):
    if isinstance(value, str):
        return _scrub_audit_string(value)
    if isinstance(value, dict):
        return {str(k): _scrub_audit_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_audit_obj(v) for v in value]
    return value


def _scrub_legacy_audit_emails(engine: Engine) -> int:
    """Idempotently replace legacy raw email addresses in auditlog with pseudonymous id-XXXXXXXX tokens."""
    try:
        scrubbed = 0
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(text("DROP TRIGGER IF EXISTS auditlog_no_update ON auditlog;"))
            rows = conn.execute(
                text(
                    "SELECT id, entity_id, detail FROM auditlog "
                    "WHERE entity_id LIKE '%@%' OR CAST(detail AS TEXT) LIKE '%@%'"
                )
            ).fetchall()
            for row_id, raw_eid, raw_detail in rows:
                new_eid = _scrub_audit_string(str(raw_eid or ""))
                if isinstance(raw_detail, (dict, list)):
                    new_detail_str = json.dumps(_scrub_audit_obj(raw_detail))
                elif isinstance(raw_detail, str):
                    try:
                        parsed = json.loads(raw_detail)
                        new_detail_str = json.dumps(_scrub_audit_obj(parsed))
                    except (ValueError, TypeError):
                        new_detail_str = _scrub_audit_string(raw_detail)
                elif raw_detail is None:
                    new_detail_str = "{}"
                else:
                    new_detail_str = _scrub_audit_string(str(raw_detail))

                orig_detail_cmp = (
                    json.dumps(raw_detail) if isinstance(raw_detail, (dict, list)) else str(raw_detail or "")
                )
                if new_eid != str(raw_eid or "") or new_detail_str != orig_detail_cmp:
                    update_sql = (
                        "UPDATE auditlog SET entity_id = :eid, detail = CAST(:det AS JSON) WHERE id = :id"
                        if engine.dialect.name == "postgresql"
                        else "UPDATE auditlog SET entity_id = :eid, detail = :det WHERE id = :id"
                    )
                    conn.execute(
                        text(update_sql),
                        {"eid": new_eid, "det": new_detail_str, "id": row_id},
                    )
                    scrubbed += 1
        if scrubbed > 0:
            logger.info("Scrubbed legacy email literals from %d auditlog rows", scrubbed)
        return scrubbed
    except Exception as exc:  # noqa: BLE001 - boot scrub must never block startup
        logger.warning("Legacy audit email scrub failed: %s", exc)
        return 0


def install_audit_guard(engine: Engine) -> None:
    """Make AuditLog rows immutable: refuse UPDATE and DELETE at the ORM layer.
    Call once per engine (idempotent)."""
    from sqlalchemy import event

    from .models import AuditLog

    if getattr(engine, "_audit_guard_installed", False):
        return

    @event.listens_for(AuditLog, "before_update", retval=True)
    def _block_update(mapper, connection, target):  # pragma: no cover - guard
        raise RuntimeError("AuditLog rows are immutable; append a new row instead")

    @event.listens_for(AuditLog, "before_delete", retval=True)
    def _block_delete(mapper, connection, target):  # pragma: no cover - guard
        raise RuntimeError("AuditLog rows are immutable; append a new row instead")

    engine._audit_guard_installed = True
