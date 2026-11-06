import logging

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
    _ensure_indexes(engine)
    _purge_dummy_jobs(engine)
    _sanitize_legacy_salaries(engine)
    _sanitize_legacy_experience(engine)
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
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS min_gpa FLOAT NOT NULL DEFAULT 0.0",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS allowed_branches JSON NOT NULL DEFAULT '[]'::json",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS job_url TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'local'",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS scraper_engine TEXT NOT NULL DEFAULT 'jobspy'",
            "ALTER TABLE job ADD COLUMN IF NOT EXISTS salary_disclosed BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE agentrun ADD COLUMN IF NOT EXISTS citations JSON NOT NULL DEFAULT '{}'::json",
            "ALTER TABLE agentstep ADD COLUMN IF NOT EXISTS verification_passed BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE agentstep ADD COLUMN IF NOT EXISTS error_recovered TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE planitem ADD COLUMN IF NOT EXISTS week INT NOT NULL DEFAULT 1",
            "ALTER TABLE interview ADD COLUMN IF NOT EXISTS include_resume_probes BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE interview ADD COLUMN IF NOT EXISTS adaptive_followup BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE approvalrequest ADD COLUMN IF NOT EXISTS endorsed_by TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE approvalrequest ADD COLUMN IF NOT EXISTS endorsed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL",
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
            except Exception as exc:  # noqa: BLE001
                logger.warning("SQLite column migration failed: %s", exc)
            # M18.1/18.2: interview feature flags on legacy SQLite DBs.
            try:
                iv_cols = [
                    row[1] for row in
                    conn.execute(text("PRAGMA table_info(interview)")).fetchall()
                ]
                if iv_cols and "include_resume_probes" not in iv_cols:
                    conn.execute(text(
                        "ALTER TABLE interview ADD COLUMN include_resume_probes BOOLEAN DEFAULT 0"))
                if iv_cols and "adaptive_followup" not in iv_cols:
                    conn.execute(text(
                        "ALTER TABLE interview ADD COLUMN adaptive_followup BOOLEAN DEFAULT 0"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("SQLite interview column migration failed: %s", exc)
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


def _sanitize_legacy_salaries(engine: Engine) -> int:
    """Reset legacy [6, 12] auto-filled salaries to undisclosed [0, 0] (§Milestone 8)."""
    if engine.dialect.name == "postgresql":
        stmt = text(
            "UPDATE job SET salary_lpa = '[0, 0]'::json, salary_disclosed = FALSE "
            "WHERE (salary_lpa::text LIKE '[6, 12]%' OR salary_lpa::text LIKE '[6,12]%') "
            "AND (salary_disclosed IS FALSE OR salary_disclosed IS NULL)"
        )
    else:
        stmt = text(
            "UPDATE job SET salary_lpa = '[0, 0]', salary_disclosed = 0 "
            "WHERE (salary_lpa LIKE '[6, 12]%' OR salary_lpa LIKE '[6,12]%') "
            "AND (salary_disclosed = 0 OR salary_disclosed IS NULL)"
        )
    try:
        with engine.begin() as conn:
            res = conn.execute(stmt)
            return int(res.rowcount or 0)
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
