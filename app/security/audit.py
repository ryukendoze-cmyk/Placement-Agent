"""Append-only audit trail helper (Milestone 14).

Every privileged or state-changing side-effect funnels through :func:`log_audit`
so ``AuditLog`` coverage is uniform across routes. Rows are immutable: the
``install_audit_guard`` SQLAlchemy listeners in :mod:`app.db` refuse any UPDATE
or DELETE on ``AuditLog``, so callers must only ever add rows (never modify).
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import Session

from ..models import AuditLog


def log_audit(
    session: Session,
    *,
    action: str,
    entity_type: str,
    entity_id: str,
    actor_role: str = "system",
    detail: dict | None = None,
    commit: bool = False,
) -> AuditLog:
    """Stage an ``AuditLog`` row on ``session`` and return it.

    ``commit=False`` (default) keeps the audit row in the caller's transaction so
    it commits atomically with the state change it describes. Pass
    ``commit=True`` only when the caller has no surrounding commit.
    Never raises for a malformed ``detail`` - it is coerced to a dict.
    """
    row = AuditLog(
        id=f"aud-{uuid4().hex[:8]}",
        action=action,
        entity_type=entity_type or "",
        entity_id=entity_id or "",
        actor_role=actor_role or "system",
        detail=detail if isinstance(detail, dict) else {"value": detail},
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    if commit:
        session.commit()
    return row
