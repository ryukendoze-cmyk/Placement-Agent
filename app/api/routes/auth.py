"""Institutional auth routes: demo role switcher + JKLU login portal (§17.2.A / §17.2.D).

``POST /api/auth/switch-role`` accepts a role (and optional student_id /
department / name / email profile claims), builds the multi-claim
``ps06_session`` cookie and keeps writing the legacy ``ps06_role`` cookie so
pre-Milestone-9 clients keep resolving roles. ``GET /api/auth/role`` reports
the currently resolved role plus the full CurrentUser (when one is declared).

``GET /login`` renders the branded JKLU institutional portal,
``POST /auth/login`` validates an ``@jklu.edu.in`` account, sets the same
``ps06_session`` and redirects to ``/``, and ``POST /auth/logout`` clears it.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlmodel import Session, select

from app.api.deps import get_session
from app.models import Student
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


def _build_session_response(role: str, claims: dict[str, Any], *, redirect_to: str | None = None):
    """Set ps06_session (+ legacy ps06_role) on a JSON or redirect response."""
    from app.config import settings as _settings
    if redirect_to is not None:
        resp: JSONResponse | RedirectResponse = RedirectResponse(url=redirect_to, status_code=303)
    else:
        resp = JSONResponse(content={"role": role, "ok": True, "claims": claims})
    # strict mode hardens the session cookie (M15): HttpOnly blocks JS reads;
    # demo mode keeps it readable so demo tooling can inspect the claims.
    httponly = str(getattr(_settings, "auth_mode", "demo")) == "strict"
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
    email = str(body.get("email") or "").strip() or None
    name = str(body.get("name") or "").strip() or None

    if role == "student":
        s: Student | None = None
        if session is not None:
            if raw_sid and raw_sid != "demo-student":
                s = session.get(Student, raw_sid)
            if s is None and user_id and user_id != "demo-student":
                s = session.get(Student, user_id)
            if s is None:
                s = session.exec(select(Student).order_by(Student.name.asc())).first()

        if s is not None:
            student_id = s.id
            name = s.name
            department = s.branch
            user_id = s.id
            email = f"{s.id}{_JKLU_DOMAIN}"
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

    Body: ``{"role": <ROLES>, "student_id"?: str, "department"?: str,
    "name"?: str, "email"?: str, "user_id"?: str}``. Role is required and
    validated; profile claims default to demo identities for the role.
    """
    body = await _read_json_body(request)
    claims = _claims_from(body, session=session)
    if claims is None:
        return JSONResponse(status_code=400,
                            content={"error": f"role must be one of {list(ROLES)}"})
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
    students = session.exec(select(Student).order_by(Student.name.asc())).all()
    return request.app.state.templates.TemplateResponse(
        request, "auth/login.html",
        {"role": current_role(request), "students": students, "error": ""},
    )


@router.post("/auth/login")
async def auth_login(request: Request, session: Session = Depends(get_session)) -> Any:
    """Validate an institutional account, set ps06_session, redirect to / (§17.2.D.2).

    Non-``@jklu.edu.in`` emails are rejected with 403 and receive no session.
    """
    body = await _read_json_body(request)
    email = str(body.get("email") or "").strip().lower()
    if not email or not email.endswith(_JKLU_DOMAIN):
        raise HTTPException(403, f"Institutional sign-in requires a {_JKLU_DOMAIN} account")

    # Role may be supplied (demo fast-login) or derived from the roll number.
    role = normalize_role(body.get("role")) or "student"
    local = email[: -len(_JKLU_DOMAIN)]

    claims = _claims_from({
        "role": role,
        "email": email,
        "user_id": local,
        "name": str(body.get("name") or "").strip() or local,
        # Roll number (e.g. 2022btech01) maps to the student's id (§17.2.D.2).
        "student_id": str(body.get("student_id") or "").strip() or (local if role == "student" else None),
        "department": str(body.get("department") or "").strip() or None,
    }, session=session)
    if claims is None:  # defensive: role already normalized above
        raise HTTPException(422, "invalid role")
    is_json = request.headers.get("content-type", "").startswith("application/json") or "application/json" in request.headers.get("accept", "")
    redirect_to = None if is_json else "/"
    return _build_session_response(role, claims, redirect_to=redirect_to)


@router.post("/auth/logout")
def auth_logout() -> RedirectResponse:
    """Clear the session cookies and redirect to /login (§17.2.D.2)."""
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("ps06_session", samesite="lax")
    resp.delete_cookie("ps06_role", samesite="lax")
    return resp
