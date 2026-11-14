"""RBAC (Phase 4 -> Milestone 9): roles, session identity, and route guards.

Identity resolution (ARCHITECTURE §17.2.A), in priority order:

1. ``X-Demo-Role`` / ``X-Demo-Student-ID`` headers (automated test suites).
2. ``ps06_session`` cookie — the multi-claim session holding ``role``,
   ``user_id``, ``student_id``, ``department``, ``email`` and ``name``
   (JSON, base64url-encoded by :func:`encode_session`).
3. Legacy ``ps06_role`` cookie (single-value, pre-Milestone-9 clients).
4. :data:`DEFAULT_ROLE` — demo/open mode: nothing declared, nothing enforced.

``get_current_user(request)`` resolves a :class:`CurrentUser` from those
sources and guarantees the ``student_id`` claim is locked to the resolved
identity whenever ``role == "student"``. ``get_current_user_optional``
returns ``None`` in open demo mode (no identity declared at all) so callers
can distinguish "demo default" from "authenticated principal".

`current_role(request)` is intentionally never None, so existing code that
reads the role stays simple; protection is expressed by decorating a route
with `Depends(require_role("tpo_admin"))`.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from typing import Any, Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, field_validator

ROLES = ("student", "faculty_mentor", "tpo_admin")
DEFAULT_ROLE = "tpo_admin"   # demo/open default: no explicit role => everything allowed

_HEADER = "x-demo-role"
_HEADER_STUDENT = "x-demo-student-id"
_COOKIE = "ps06_role"        # legacy single-value role cookie
_SESSION_COOKIE = "ps06_session"   # multi-claim session (§17.2.A)


class CurrentUser(BaseModel):
    """Unified identity model (ARCHITECTURE §17.2.A).

    ``student_id`` is strictly bound: it must be a non-empty value whenever
    ``role == "student"`` (enforced by the validator below), so row-level
    security can trust ``user.student_id`` as the ONLY student record that
    principal may reach.
    """

    user_id: str
    email: str
    name: str
    role: Literal["student", "faculty_mentor", "tpo_admin"]
    student_id: str | None = None
    department: str | None = None

    @field_validator("student_id")
    @classmethod
    def _lock_student_id(cls, v: str | None, info) -> str | None:
        """Strictly bind student_id to the student role (§17.2.A)."""
        role = info.data.get("role")
        if role == "student" and not (v and str(v).strip()):
            raise ValueError("student_id is required and must be non-empty when role == 'student'")
        if role != "student" and v:
            # Non-student roles never carry a student_id claim (keeps RLS checks simple).
            return None
        return v


def encode_session(claims: dict[str, Any]) -> str:
    """Serialize a claim dict to a cryptographically signed ps06_session token.

    Produces ``<base64url_payload>.<hmac_sha256_hex>`` (Milestone 15). ``iat`` and
    ``exp`` (from ``settings.session_ttl_seconds``) are injected when absent so every
    newly issued session is expirable and tamper-evident regardless of ``auth_mode``.
    """
    from ..config import settings

    payload = dict(claims)
    now = time.time()
    payload.setdefault("iat", int(now))
    payload.setdefault("exp", int(now) + int(settings.session_ttl_seconds))
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_b64 = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    sig = hmac.new(settings.session_secret_key.encode("utf-8"),
                   payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def decode_session(value: str | None) -> dict[str, Any] | None:
    """Parse a ps06_session cookie value back to a claim dict.

    Signed tokens (``<payload>.<sig>``) are verified with HMAC-SHA256 using
    ``hmac.compare_digest`` (constant-time) and checked against ``exp``; tampered,
    malformed, or expired tokens return ``None``.

    Legacy unsigned base64url cookies (no ``.``) are accepted only in
    ``settings.auth_mode == "demo"`` (backward compatibility with pre-Milestone-15
    cookies and test helpers); ``auth_mode == "strict"`` rejects them outright so
    only cryptographically signed sessions survive in production.

    Returns ``None`` for absent / malformed / non-dict payloads rather than raising,
    so a corrupted cookie never 500s.
    """
    from ..config import settings

    if not value:
        return None
    # The test client may hand back the cookie wrapped in literal quotes.
    value = value.strip().strip('"')
    if not value:
        return None
    if "." in value:
        payload_b64, _, sig = value.partition(".")
        if not payload_b64 or not sig:
            return None
        try:
            expected = hmac.new(settings.session_secret_key.encode("utf-8"),
                                payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, sig):
                return None
            decoded = base64.urlsafe_b64decode(payload_b64.encode("ascii", "ignore"))
            data = json.loads(decoded.decode("utf-8"))
        except (ValueError, TypeError, binascii.Error, json.JSONDecodeError,
                UnicodeDecodeError, UnicodeEncodeError):
            return None
        if not isinstance(data, dict):
            return None
        exp = data.get("exp")
        if exp is not None:
            try:
                if float(exp) < time.time():
                    return None
            except (TypeError, ValueError):
                return None
        return data
    # Legacy unsigned token (no signature separator).
    if str(getattr(settings, "auth_mode", "demo")) == "strict":
        return None
    try:
        decoded = base64.urlsafe_b64decode(value.encode("ascii", "ignore"))
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def normalize_role(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return v if v in ROLES else None


def _is_strict() -> bool:
    from ..config import settings
    return str(getattr(settings, "auth_mode", "demo")).strip().lower() == "strict"


def _session_claims(request: Request) -> dict[str, Any]:
    return decode_session(request.cookies.get(_SESSION_COOKIE)) or {}


def current_role(request: Request) -> str:
    """Resolve the active role from header, session, legacy cookie, else default.

    In ``auth_mode == "strict"``, headers (``X-Demo-Role``) and unsigned legacy
    cookies (``ps06_role``) are ignored; only a valid signed ``ps06_session``
    cookie sets a role, and an unauthenticated request resolves to
    ``"anonymous"`` (never ``DEFAULT_ROLE``).
    """
    if _is_strict():
        session_role = normalize_role(_session_claims(request).get("role"))
        return session_role or "anonymous"
    hdr = normalize_role(request.headers.get(_HEADER))
    if hdr:
        return hdr
    session_role = normalize_role(_session_claims(request).get("role"))
    if session_role:
        return session_role
    cookie = normalize_role(request.cookies.get(_COOKIE))
    if cookie:
        return cookie
    return DEFAULT_ROLE


def role_explicitly_set(request: Request) -> bool:
    """True only when the caller actually declared a role (session only in strict mode)."""
    if _is_strict():
        return normalize_role(_session_claims(request).get("role")) is not None
    if normalize_role(request.headers.get(_HEADER)) is not None:
        return True
    if normalize_role(_session_claims(request).get("role")) is not None:
        return True
    return normalize_role(request.cookies.get(_COOKIE)) is not None


def get_current_user_optional(request: Request) -> CurrentUser | None:
    """Return the CurrentUser if one is declared, else None (open demo mode)."""
    if not role_explicitly_set(request):
        return None
    return get_current_user(request)


def get_current_user(request: Request) -> CurrentUser:
    """Resolve the CurrentUser identity for this request (§17.2.A)."""
    claims = _session_claims(request)

    if _is_strict():
        role = normalize_role(claims.get("role"))
        if not role:
            raise HTTPException(401, "Authentication required")
        session_sid = str(claims.get("student_id") or "").strip() or None
        student_id = (session_sid or "demo-student") if role == "student" else None
        user_id = str(claims.get("user_id") or "").strip() or (
            student_id if student_id else f"demo-{role}"
        )
        email = str(claims.get("email") or "").strip() or f"{user_id}@jklu.edu.in"
        name = str(claims.get("name") or "").strip() or role.replace("_", " ").title()
        department = str(claims.get("department") or "").strip() or None
        return CurrentUser(
            user_id=user_id,
            email=email,
            name=name,
            role=role,  # type: ignore[arg-type]
            student_id=student_id,
            department=department,
        )

    # Role: header > session > legacy cookie > default
    role = (
        normalize_role(request.headers.get(_HEADER))
        or normalize_role(claims.get("role"))
        or normalize_role(request.cookies.get(_COOKIE))
        or DEFAULT_ROLE
    )
    assert role in ROLES  # normalize_role already guards this

    # student_id: header override > session claim > synthetic default (student only)
    header_sid = str(request.headers.get(_HEADER_STUDENT, "") or "").strip() or None
    session_sid = str(claims.get("student_id") or "").strip() or None

    if role == "student":
        student_id = header_sid or session_sid or "demo-student"
    else:
        student_id = None  # validator enforces non-student roles drop any sid

    user_id = str(claims.get("user_id") or "").strip() or (
        student_id if student_id else f"demo-{role}"
    )
    email = str(claims.get("email") or "").strip() or f"{user_id}@jklu.edu.in"
    name = str(claims.get("name") or "").strip() or role.replace("_", " ").title()
    header_dept = str(request.headers.get("x-demo-department", "") or "").strip() or None
    department = header_dept or str(claims.get("department") or "").strip() or None

    return CurrentUser(
        user_id=user_id,
        email=email,
        name=name,
        role=role,  # type: ignore[arg-type]
        student_id=student_id,
        department=department,
    )


def role_allows(role: str, required: str) -> bool:
    return required in _IMPLIES.get(role, set())


# role -> set of roles allowed (a role implies all lower roles)
_IMPLIES: dict[str, set[str]] = {
    "student": {"student"},
    "faculty_mentor": {"student", "faculty_mentor"},
    "tpo_admin": {"student", "faculty_mentor", "tpo_admin"},
}


def require_role(*allowed: str):
    """FastAPI dependency: 401/403 unless the active role is permitted.

    - In strict mode, missing signed session raises 401.
    - If the caller declared a role (header/session/cookie) it must be allowed (403 otherwise).
    - If no role was declared in demo mode, the request is allowed; the returned
      role is still the resolved default so handlers can log it.
    """
    allowed = tuple(a for a in allowed if a in ROLES) or (DEFAULT_ROLE,)

    def _dep(request: Request) -> str:
        if _is_strict() and not role_explicitly_set(request):
            raise HTTPException(401, "Authentication required")
        role = current_role(request)
        if role_explicitly_set(request) and not any(role_allows(role, a) for a in allowed):
            raise HTTPException(403, f"role '{role}' is not permitted; requires one of {list(allowed)}")
        return role

    return _dep


def enforce_student_scope(request: Request, sid: str) -> None:
    """Row-level security guard (ARCHITECTURE §17.2.B.1).

    Raises 403 when:
    - An explicitly-declared student identity tries to reach a record belonging to
      a different student.
    - An explicitly-declared faculty_mentor with a department claim tries to reach
      a student belonging to a different department.
    TPO admins (and open demo mode with no declared identity) pass through.
    """
    user = get_current_user_optional(request)
    if user is None:
        return
    if user.role == "student" and user.student_id != sid:
        raise HTTPException(
            403,
            "Access denied: Students may only access their own placement records",
        )
    if user.role == "faculty_mentor":
        dept = (user.department or "").strip().lower()
        if dept:
            state = getattr(getattr(request, "app", None), "state", None)
            ctx_obj = getattr(state, "ctx", None) if state is not None else None
            engine = getattr(ctx_obj, "engine", None) if ctx_obj is not None else None
            if engine is not None:
                from sqlmodel import Session as _Session

                from app.models import Student as _Student

                with _Session(engine) as _db:
                    stu = _db.get(_Student, sid)
                if stu is not None:
                    s_branch = (stu.branch or "").strip().lower()
                    if s_branch and s_branch != dept:
                        raise HTTPException(
                            403,
                            f"Access denied: Faculty mentor ({user.department}) may not access {stu.branch} student records",
                        )
