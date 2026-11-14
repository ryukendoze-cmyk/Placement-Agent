"""Institutional auth routes: demo role switcher + JKLU login portal (§17.2.A / §17.2.D).

``POST /api/auth/switch-role`` accepts a role (and optional student_id /
department / name / email profile claims), builds the multi-claim
``ps06_session`` cookie and keeps writing the legacy ``ps06_role`` cookie so
pre-Milestone-9 clients keep resolving roles. ``GET /api/auth/role`` reports
the currently resolved role plus the full CurrentUser (when one is declared).

``GET /login`` renders the branded JKLU institutional portal,
``POST /auth/login`` validates an ``@jklu.edu.in`` account (and PIN against
``data/personas.json``), sets ``ps06_session`` and redirects to ``/``, and
``POST /auth/logout`` clears it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlmodel import Session, select

from app.api.deps import get_session
from app.models import Student, StudentSkill
from app.security.audit import log_audit
from app.security.pii import audit_actor_ref, mask_email
from app.security.rbac import (
    ROLES,
    current_role,
    encode_session,
    get_current_user_optional,
    normalize_role,
)

router = APIRouter()

#: Attributes the demo switcher may carry into the session claims. Anything
#: else in the body is ignored (keeps the cookie shape predictable).
_CLAIM_KEYS = ("user_id", "email", "name", "student_id", "department")

#: Institutional email domain (§17.2.D.2).
_JKLU_DOMAIN = "@jklu.edu.in"

_PERSONAS_PATH = Path(__file__).resolve().parents[3] / "data" / "personas.json"
_ROHIT_RESUME = "Skills: Python, FastAPI, Docker, Git\nFull-stack app."


def _load_personas() -> tuple[str, dict[str, dict[str, Any]]]:
    """Return (demo_pin, email_lower -> persona_dict) from data/personas.json."""
    try:
        raw = json.loads(_PERSONAS_PATH.read_text(encoding="utf-8"))
        demo_pin = str(raw.get("demo_pin") or "2026")
        by_email = {
            str(p["email"]).strip().lower(): p
            for p in raw.get("personas", [])
            if isinstance(p, dict) and p.get("email")
        }
        return demo_pin, by_email
    except Exception:
        return "2026", {}


def _verify_pin(pin: str, pin_hash: str | None, demo_pin: str = "2026") -> bool:
    """Verify a PIN against a pbkdf2_sha256$<iters>$<salt>$<hash> string."""
    if not pin_hash:
        return hmac.compare_digest(pin, demo_pin)
    parts = str(pin_hash).split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    _, iters_s, salt_hex, expected_hex = parts
    try:
        iters = int(iters_s)
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iters).hex()
    return hmac.compare_digest(dk, expected_hex)


def _ensure_rohit_shetty(session: Session, request: Request | None = None) -> Student:
    """Bind to seeded 'Rohit Shetty' student, creating him if missing in an unseeded DB."""
    s = session.exec(select(Student).where(Student.name == "Rohit Shetty")).first()
    if s is not None:
        return s
    sid = f"stu-{uuid4().hex[:8]}"
    s = Student(
        id=sid,
        name="Rohit Shetty",
        branch="CSE",
        year=3,
        gpa=7.9,
        resume_text=_ROHIT_RESUME,
    )
    session.add(s)
    session.commit()
    session.refresh(s)
    ctx = getattr(request.app.state, "ctx", None) if request is not None else None
    if ctx is not None and getattr(ctx, "taxonomy", None) is not None:
        try:
            from app.skills.extract import extract_skills, level_for

            found = extract_skills(_ROHIT_RESUME, ctx.taxonomy)
            for skill_id, lines in found.items():
                session.add(
                    StudentSkill(
                        student_id=sid,
                        skill_id=skill_id,
                        mentions=len(lines),
                        level=level_for(len(lines), lines),
                        evidence=list(lines),
                    )
                )
            session.commit()
            if getattr(ctx, "vector_store", None) and getattr(ctx, "embedder", None):
                ctx.vector_store.upsert("student", sid, ctx.embedder.embed(_ROHIT_RESUME))
        except Exception:
            session.rollback()
    return s


def _build_session_response(role: str, claims: dict[str, Any], *, redirect_to: str | None = None):
    """Set ps06_session (+ legacy ps06_role) on a JSON or redirect response."""
    from app.config import settings as _settings
    if redirect_to is not None:
        resp: JSONResponse | RedirectResponse = RedirectResponse(url=redirect_to, status_code=303)
    else:
        resp = JSONResponse(content={"role": role, "ok": True, "claims": claims})
    # strict mode hardens the session cookie (M15): HttpOnly blocks JS reads;
    # demo mode keeps it readable so demo tooling can inspect the claims.
    httponly = str(getattr(_settings, "auth_mode", "demo")).lower() == "strict"
    resp.set_cookie("ps06_session", encode_session(claims), httponly=httponly, samesite="lax")
    resp.set_cookie("ps06_role", role, httponly=httponly, samesite="lax")
    return resp


def _claims_from(body: dict[str, Any], session: Session | None = None) -> dict[str, Any] | None:
    """Build validated claims from a role payload; None when the role is invalid."""
    role = normalize_role(body.get("role"))
    if role not in ROLES:
        return None

    raw_sid = str(body.get("student_id") or "").strip() or None
    department = str(body.get("department") or "").strip() or None
    user_id = str(body.get("user_id") or "").strip() or None
    explicit_email = str(body.get("email") or "").strip() or None
    email = explicit_email
    name = str(body.get("name") or "").strip() or None

    if role == "student":
        s: Student | None = None
        if session is not None:
            if raw_sid and raw_sid != "demo-student":
                s = session.get(Student, raw_sid)
            if s is None and user_id and user_id != "demo-student":
                s = session.get(Student, user_id)
            is_generic_demo = (
                (not raw_sid or raw_sid == "demo-student")
                and (not user_id or user_id == "demo-student")
                and not explicit_email
            )
            if s is None and is_generic_demo:
                s = session.exec(select(Student).order_by(Student.name.asc())).first()

        if s is not None:
            student_id = s.id
            name = s.name
            department = s.branch
            user_id = s.id
            email = explicit_email or f"{s.id}{_JKLU_DOMAIN}"
        else:
            student_id = None if raw_sid == "demo-student" else raw_sid
            if not user_id or user_id == "demo-student":
                user_id = student_id if student_id else "student"
            if not email:
                email = f"{user_id}{_JKLU_DOMAIN}"
            if not name:
                name = "Student"
    else:
        student_id = None
        if not user_id:
            user_id = f"demo-{role}"
        if not email:
            email = f"{user_id}{_JKLU_DOMAIN}"
        if not name:
            name = role.replace("_", " ").title()

    claims: dict[str, Any] = {
        "role": role,
        "user_id": user_id,
        "email": email,
        "name": name,
        "student_id": student_id,
        "department": department,
    }
    return {k: v for k, v in claims.items() if v is not None}


async def _read_json_body(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()
    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        try:
            form = await request.form()
            return dict(form)
        except Exception:
            return {}
    if request.headers.get("content-length") in (None, "0"):
        return {}
    try:
        parsed = await request.json()
    except Exception:  # noqa: BLE001 - malformed body => empty, caller validates
        return {}
    return parsed if isinstance(parsed, dict) else {}


@router.post("/api/auth/switch-role")
async def switch_role(request: Request, session: Session = Depends(get_session)) -> JSONResponse:
    """Set the multi-claim ps06_session cookie (and legacy ps06_role cookie).

    In ``auth_mode == "strict"``, allowed only when ``demo_role_switch=True``.
    """
    from app.config import settings as _settings

    trace_id = f"tr-{uuid4().hex[:8]}"
    is_strict = str(getattr(_settings, "auth_mode", "demo")).lower() == "strict"
    if is_strict and not bool(getattr(_settings, "demo_role_switch", True)):
        log_audit(
            session,
            action="auth.denied",
            entity_type="auth",
            entity_id="/api/auth/switch-role",
            actor_role=current_role(request),
            detail={"trace_id": trace_id, "path": "/api/auth/switch-role", "reason": "demo_role_switch_disabled"},
            commit=True,
        )
        raise HTTPException(403, "Demo role switching is disabled in strict mode")

    from_role = current_role(request)
    prev_user = get_current_user_optional(request)
    body = await _read_json_body(request)
    claims = _claims_from(body, session=session)
    if claims is None:
        return JSONResponse(status_code=400,
                            content={"error": f"role must be one of {list(ROLES)}"})

    log_audit(
        session,
        action="auth.role_switched",
        entity_type="auth",
        entity_id=str(claims.get("user_id") or claims["role"]),
        actor_role=claims["role"],
        detail={
            "trace_id": trace_id,
            "from_role": from_role,
            "to_role": claims["role"],
            "actor": prev_user.user_id if prev_user else from_role,
            "student_id": claims.get("student_id"),
        },
        commit=True,
    )
    return _build_session_response(claims["role"], claims)


@router.get("/api/auth/role")
def get_role(request: Request) -> dict[str, Any]:
    """Report the active role, available roles, and resolved CurrentUser (if any)."""
    user = get_current_user_optional(request)
    return {
        "role": current_role(request),
        "roles": list(ROLES),
        "user": user.model_dump() if user else None,
    }


# ---------------------------------------------------------------------------
# §17.2.D — Institutional login portal
# ---------------------------------------------------------------------------


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    """Branded JKLU institutional login portal (§17.2.D.1)."""
    from app.config import settings as _settings

    is_strict = str(getattr(_settings, "auth_mode", "demo")).strip().lower() == "strict"
    allow_demo = bool(getattr(_settings, "demo_role_switch", True))
    show_demo_controls = allow_demo and not is_strict
    students = (
        session.exec(select(Student).order_by(Student.name.asc())).all()
        if show_demo_controls
        else []
    )
    demo_pin, personas_map = _load_personas()
    return request.app.state.templates.TemplateResponse(
        request, "auth/login.html",
        {
            "role": current_role(request),
            "students": students,
            "personas": list(personas_map.values()) if show_demo_controls else [],
            "demo_pin": demo_pin if show_demo_controls else "",
            "demo_role_switch": allow_demo,
            "auth_mode": "strict" if is_strict else "demo",
            "show_demo_controls": show_demo_controls,
            "error": "",
        },
    )


@router.post("/auth/login")
async def auth_login(request: Request, session: Session = Depends(get_session)) -> Any:
    """Validate an institutional account (+ PIN), set ps06_session, redirect to / (§17.2.D.2).

    Non-``@jklu.edu.in`` emails are rejected with 403 and receive no session.
    """
    from app.config import settings as _settings

    trace_id = f"tr-{uuid4().hex[:8]}"
    body = await _read_json_body(request)
    email = str(body.get("email") or "").strip().lower()
    if not email or not email.endswith(_JKLU_DOMAIN):
        log_audit(
            session,
            action="auth.denied",
            entity_type="auth",
            entity_id=audit_actor_ref(email) if email else "anonymous",
            actor_role="anonymous",
            detail={
                "trace_id": trace_id,
                "path": "/auth/login",
                "actor_ref": mask_email(email) if email else "[empty]",
                "domain_valid": False,
                "reason": "non_jklu_domain",
            },
            commit=True,
        )
        raise HTTPException(403, f"Institutional sign-in requires a {_JKLU_DOMAIN} account")

    demo_pin, personas_map = _load_personas()
    persona = personas_map.get(email)

    raw_pin = body.get("pin")
    pin_str = str(raw_pin).strip() if raw_pin is not None else ""
    fast_login_raw = body.get("fast_login")
    fast_login = (
        fast_login_raw is True
        or str(fast_login_raw or "").strip().lower() in ("true", "1", "yes")
    )
    is_strict = str(getattr(_settings, "auth_mode", "demo")).lower() == "strict"
    allow_fast = bool(getattr(_settings, "demo_role_switch", True))

    if is_strict and persona is None:
        log_audit(
            session,
            action="auth.denied",
            entity_type="auth",
            entity_id=audit_actor_ref(email),
            actor_role="anonymous",
            detail={
                "trace_id": trace_id,
                "path": "/auth/login",
                "actor_ref": mask_email(email),
                "domain_valid": True,
                "reason": "unregistered_account",
            },
            commit=True,
        )
        raise HTTPException(401, "Unknown institutional account")

    if pin_str:
        expected_hash = persona.get("pin_hash") if persona else None
        if not _verify_pin(pin_str, expected_hash, demo_pin=demo_pin):
            log_audit(
                session,
                action="auth.denied",
                entity_type="auth",
                entity_id=audit_actor_ref(email),
                actor_role="anonymous",
                detail={
                    "trace_id": trace_id,
                    "path": "/auth/login",
                    "actor_ref": mask_email(email),
                    "domain_valid": True,
                    "reason": "invalid_pin",
                },
                commit=True,
            )
            raise HTTPException(401, "Invalid institutional PIN")
    elif is_strict and not (fast_login and allow_fast):
        log_audit(
            session,
            action="auth.denied",
            entity_type="auth",
            entity_id=audit_actor_ref(email),
            actor_role="anonymous",
            detail={
                "trace_id": trace_id,
                "path": "/auth/login",
                "actor_ref": mask_email(email),
                "domain_valid": True,
                "reason": "pin_required",
            },
            commit=True,
        )
        raise HTTPException(401, "Institutional PIN is required")

    local = email[: -len(_JKLU_DOMAIN)]
    if persona is not None:
        role = normalize_role(persona.get("role")) or "student"
        name = str(body.get("name") or "").strip() or str(persona.get("name") or local)
        department = str(body.get("department") or "").strip() or persona.get("department")
        user_id = str(persona.get("user_id") or local)
    else:
        # Non-persona accounts must never self-assert faculty_mentor or tpo_admin roles.
        role = "student"
        name = str(body.get("name") or "").strip() or local
        department = str(body.get("department") or "").strip() or None
        user_id = local

    student_id: str | None = None
    if role == "student":
        if email == "rohit.shetty@jklu.edu.in" or local == "rohit.shetty":
            rohit = _ensure_rohit_shetty(session, request=request)
            student_id = rohit.id
            user_id = rohit.id
            name = rohit.name
            department = rohit.branch
        else:
            student_id = local

    claims = _claims_from(
        {
            "role": role,
            "email": email,
            "user_id": user_id,
            "name": name,
            "student_id": student_id,
            "department": department,
        },
        session=session,
    )
    if claims is None:  # defensive: role already normalized above
        raise HTTPException(422, "invalid role")

    log_audit(
        session,
        action="auth.login",
        entity_type="auth",
        entity_id=str(claims.get("user_id") or audit_actor_ref(email)),
        actor_role=claims["role"],
        detail={
            "trace_id": trace_id,
            "actor_ref": mask_email(email),
            "role": claims["role"],
            "student_id": claims.get("student_id"),
            "fast_login": fast_login,
        },
        commit=True,
    )
    is_json = (
        request.headers.get("content-type", "").startswith("application/json")
        or "application/json" in request.headers.get("accept", "")
    )
    redirect_to = None if is_json else "/"
    return _build_session_response(role, claims, redirect_to=redirect_to)


@router.post("/auth/logout")
def auth_logout(request: Request, session: Session = Depends(get_session)) -> RedirectResponse:
    """Clear the session cookies and redirect to /login (§17.2.D.2)."""
    trace_id = f"tr-{uuid4().hex[:8]}"
    actor = current_role(request)
    user = get_current_user_optional(request)
    log_audit(
        session,
        action="auth.logout",
        entity_type="auth",
        entity_id=user.user_id if user else actor,
        actor_role=actor if actor in ROLES else "anonymous",
        detail={
            "trace_id": trace_id,
            "user_id": user.user_id if user else None,
            "actor_ref": mask_email(user.email) if (user and user.email) else None,
        },
        commit=True,
    )
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("ps06_session", samesite="lax")
    resp.delete_cookie("ps06_role", samesite="lax")
    return resp
