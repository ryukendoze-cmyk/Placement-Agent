"""Campus placement drive routes: Smart JD parsing, HITL submission, approvals, and public JNF intake."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from jinja2 import TemplateNotFound
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.api.routes import approvals as approvals_routes
from app.campus.agent import (
    drive_to_dict,
    parse_campus_jd,
    validate_drive_payload,
)
from app.campus.ingest import MAX_JD_BYTES, extract_jd_text, normalize_jd_text
from app.models import ApprovalRequest, CampusDrive
from app.security.audit import log_audit
from app.security.ratelimit import check_rate_limit
from app.security.rbac import (
    decode_session,
    encode_session,
    get_current_user_optional,
    require_role,
)

router = APIRouter()


def _create_jnf_token(expires_hours: int = 72, created_by: str = "tpo_admin") -> str:
    now = int(time.time())
    hours = max(1, min(720, int(expires_hours)))
    return encode_session({
        "kind": "jnf_intake",
        "jnf_id": f"jnf-{uuid4().hex[:8]}",
        "created_by": created_by,
        "iat": now,
        "exp": now + hours * 3600,
    })


def _verify_jnf_token(token: str | None) -> dict[str, Any]:
    tok = str(token or "").strip()
    if not tok or "." not in tok:
        raise HTTPException(401, "Invalid or expired JNF token")
    claims = decode_session(tok)
    if not claims or claims.get("kind") != "jnf_intake":
        raise HTTPException(401, "Invalid or expired JNF token")
    return claims


def _read_capped_jd(upload: UploadFile) -> bytes:
    data = upload.file.read(MAX_JD_BYTES + 1)
    if len(data) > MAX_JD_BYTES:
        raise HTTPException(413, f"JD file exceeds maximum allowed size ({MAX_JD_BYTES} bytes)")
    return data


def _approval_snapshot(drive: CampusDrive) -> tuple[str, str, dict[str, Any]]:
    tier_label = {
        "super_dream": "Super Dream",
        "dream": "Dream",
        "standard": "Standard",
    }.get(drive.policy_tier or "", "Not classified")
    comp = (
        f"₹{drive.ctc_lpa_max:g} LPA"
        if drive.salary_disclosed and drive.ctc_lpa_max
        else (f"₹{drive.stipend_inr_per_month:,}/mo stipend" if drive.stipend_inr_per_month else "Undisclosed")
    )
    elig = drive.eligibility or {}
    cgpa_str = f"CGPA >= {elig['min_cgpa']}" if elig.get("min_cgpa") else "No CGPA cutoff"
    branches = elig.get("allowed_branches") or []
    branch_str = ", ".join(branches) if branches else "All branches"
    title = f"Campus Drive: {drive.company} — {drive.role_title}"
    summary = f"{tier_label} ({comp}) · {cgpa_str} · {branch_str}"
    d = drive_to_dict(drive)
    payload = {
        "drive_id": drive.id,
        "trace_id": drive.trace_id,
        "company": drive.company,
        "role_title": drive.role_title,
        "location": drive.location,
        "work_mode": drive.work_mode,
        "experience_level": drive.experience_level,
        "ctc_lpa_min": drive.ctc_lpa_min,
        "ctc_lpa_max": drive.ctc_lpa_max,
        "salary_disclosed": bool(drive.salary_disclosed),
        "stipend_inr_per_month": drive.stipend_inr_per_month,
        "policy_tier": drive.policy_tier,
        "eligibility": dict(drive.eligibility or {}),
        "required_skills": list(drive.required_skills or []),
        "preferred_skills": list(drive.preferred_skills or []),
        "origin": drive.origin,
        "parser_metadata": dict(drive.parser_metadata or {}),
        "campus_drive": d,
    }
    return title, summary, payload


def _persist_pending_drive(
    session: Session,
    *,
    drive_id: str,
    trace_id: str,
    validated: dict[str, Any],
    origin: str,
    created_by: str,
    actor_role: str,
    audit_action: str,
    extra_audit_detail: dict[str, Any] | None = None,
) -> tuple[CampusDrive, ApprovalRequest]:
    now = datetime.now(timezone.utc)
    apr_id = f"apr-{uuid4().hex[:8]}"
    drive = CampusDrive(
        id=drive_id,
        trace_id=trace_id,
        company=validated["company"],
        role_title=validated["role_title"],
        location=validated["location"],
        work_mode=validated["work_mode"],
        ctc_lpa_min=validated["ctc_lpa_min"],
        ctc_lpa_max=validated["ctc_lpa_max"],
        salary_disclosed=validated["salary_disclosed"],
        stipend_inr_per_month=validated["stipend_inr_per_month"],
        policy_tier=validated["policy_tier"],
        policy_tier_source=validated["policy_tier_source"],
        experience_level=validated["experience_level"],
        source_portal="On-Campus (Local)",
        source_badge="On-Campus · JKLU",
        origin=origin,
        eligibility=validated["eligibility"],
        required_skills=validated["required_skills"],
        preferred_skills=validated["preferred_skills"],
        unmapped_skill_mentions=validated["unmapped_skill_mentions"],
        raw_jd_text=validated["raw_jd_text"],
        parser_metadata=validated["parser_metadata"],
        approval_status="PENDING_TPO_APPROVAL",
        approval_id=apr_id,
        job_id=None,
        created_by=created_by,
        reviewed_by=None,
        created_at=now,
        updated_at=now,
    )
    title, summary, ap_payload = _approval_snapshot(drive)
    ap = ApprovalRequest(
        id=apr_id,
        kind="campus_drive_publish",
        target_id=drive.id,
        title=title,
        summary=summary,
        payload=ap_payload,
        status="pending_review",
        requested_by=created_by,
        created_at=now,
    )
    session.add(drive)
    session.add(ap)
    detail = {
        "trace_id": trace_id,
        "approval_id": apr_id,
        "origin": origin,
        "policy_tier": drive.policy_tier,
    }
    if extra_audit_detail:
        detail.update(extra_audit_detail)
    log_audit(
        session,
        action=audit_action,
        entity_type="campus_drive",
        entity_id=drive.id,
        actor_role=actor_role,
        detail=detail,
    )
    session.commit()
    session.refresh(drive)
    return drive, ap


def _parse_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return default


def _parse_list_field(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            loaded = json.loads(s)
            if isinstance(loaded, list):
                return [str(x).strip() for x in loaded if str(x).strip()]
        except Exception:
            pass
    return [part.strip() for part in s.split(",") if part.strip()]


@router.get("/api/campus-drives/taxonomy")
def get_campus_taxonomy(
    ctx=Depends(get_ctx),
    _role: str = Depends(require_role("student", "faculty_mentor", "tpo_admin")),
) -> list[dict[str, Any]]:
    """Return canonical taxonomy skills for Smart JD autocomplete."""
    return [
        {"id": s.id, "name": s.name, "category": s.category, "aliases": list(s.aliases)}
        for s in ctx.taxonomy.all()
    ]


@router.post("/api/campus-drives/parse-jd")
async def parse_jd_endpoint(
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
    role: str = Depends(require_role("tpo_admin")),
) -> dict[str, Any]:
    """Parse an uploaded JD file or pasted text into a pre-filled CampusDriveSchema draft."""
    content_type = (request.headers.get("content-type") or "").lower()
    raw_text = ""
    force_parser = "auto"
    simulate_failure = "none"

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        upload = form.get("file")
        text_val = str(form.get("text") or "").strip()
        force_parser = str(form.get("force_parser") or form.get("mode") or "auto").strip()
        simulate_failure = str(form.get("simulate_failure") or "none").strip()
        if upload is not None and hasattr(upload, "file") and getattr(upload, "filename", ""):
            suffix = Path(upload.filename or "").suffix.lower()
            data = _read_capped_jd(upload)
            raw_text = extract_jd_text(data, suffix)
        elif text_val:
            raw_text = normalize_jd_text(text_val)
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        raw_text = normalize_jd_text(str(body.get("text") or body.get("raw_jd_text") or ""))
        force_parser = str(body.get("force_parser") or body.get("mode") or "auto").strip()
        simulate_failure = str(body.get("simulate_failure") or "none").strip()

    if not raw_text.strip():
        raise HTTPException(400, "Either a JD file (.pdf, .docx, .txt) or non-empty text is required")

    result = parse_campus_jd(
        raw_text,
        mode=force_parser,
        simulate_failure=simulate_failure,
        ctx=ctx,
        session=session,
        actor_role=role,
        origin="tpo_smart_jd",
    )
    session.commit()
    return result


@router.post("/api/campus-drives", status_code=201)
def create_campus_drive(
    body: dict[str, Any],
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
    role: str = Depends(require_role("tpo_admin")),
) -> dict[str, Any]:
    """Persist a CampusDrive + ApprovalRequest; optionally publish immediately when confirmed."""
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON body required")
    company = str(body.get("company") or "").strip()
    role_title = str(body.get("role_title") or "").strip()
    if not company or not role_title:
        raise HTTPException(400, "company and role_title are required")

    validated = validate_drive_payload(body, ctx.taxonomy)

    raw_id = str(body.get("id") or "").strip()
    if raw_id.startswith("cd-") and session.get(CampusDrive, raw_id) is None:
        drive_id = raw_id
    else:
        drive_id = f"cd-{uuid4().hex[:8]}"
    trace_id = str(body.get("trace_id") or "").strip() or f"tr-{uuid4().hex[:8]}"
    origin = str(body.get("origin") or "tpo_smart_jd").strip()
    if origin not in ("tpo_smart_jd", "recruiter_jnf"):
        origin = "tpo_smart_jd"

    user = get_current_user_optional(request)
    created_by = user.user_id if user is not None else role

    drive, ap = _persist_pending_drive(
        session,
        drive_id=drive_id,
        trace_id=trace_id,
        validated=validated,
        origin=origin,
        created_by=created_by,
        actor_role=role,
        audit_action="campus_drive.submitted",
    )

    publish_now = _parse_bool(body.get("publish_now"), False)
    confirm = _parse_bool(body.get("confirm"), False)
    if publish_now and confirm:
        ap_res = approvals_routes.approve(
            ap.id,
            payload={"note": "direct TPO sign-off"},
            session=session,
            ctx=ctx,
            role=role,
        )
        session.refresh(drive)
        executed = ap_res.get("executed") or {}
        out = drive_to_dict(drive)
        out["task_id"] = executed.get("task_id")
        if executed.get("cohort_summary"):
            out["cohort_summary"] = executed["cohort_summary"]
        return out

    return drive_to_dict(drive)


@router.get("/api/campus-drives")
def list_campus_drives(
    status: str | None = None,
    session: Session = Depends(get_session),
    _role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
) -> list[dict[str, Any]]:
    q = select(CampusDrive).order_by(CampusDrive.created_at.desc())
    if status and status.strip():
        q = q.where(CampusDrive.approval_status == status.strip().upper())
    rows = session.exec(q).all()
    return [drive_to_dict(d) for d in rows]


@router.post("/api/campus-drives/jnf-links")
def create_jnf_link(
    body: dict[str, Any] | None = None,
    request: Request = None,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("tpo_admin")),
) -> dict[str, Any]:
    """Generate a signed, expirable JNF intake token & URL for external recruiters."""
    payload = body if isinstance(body, dict) else {}
    try:
        expires_hours = int(payload.get("expires_hours") or 72)
    except (TypeError, ValueError):
        expires_hours = 72
    expires_hours = max(1, min(720, expires_hours))
    user = get_current_user_optional(request) if request is not None else None
    creator = user.user_id if user is not None else role

    token = _create_jnf_token(expires_hours=expires_hours, created_by=creator)
    claims = decode_session(token) or {}
    trace_id = f"tr-{uuid4().hex[:8]}"
    log_audit(
        session,
        action="jnf.link_created",
        entity_type="jnf_link",
        entity_id=str(claims.get("jnf_id") or ""),
        actor_role=role,
        detail={
            "trace_id": trace_id,
            "jnf_id": claims.get("jnf_id"),
            "expires_hours": expires_hours,
        },
        commit=True,
    )
    return {
        "token": token,
        "url": f"/jnf/{token}",
        "expires_hours": expires_hours,
        "trace_id": trace_id,
    }


@router.post("/api/campus-drives/public-jnf", status_code=201)
async def submit_public_jnf(
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
    _rl: None = Depends(check_rate_limit("public_jnf", max_requests=20, window_seconds=60)),
) -> dict[str, Any]:
    """Public recruiter JNF submission guarded by signed JNF token. Never publishes directly."""
    content_type = (request.headers.get("content-type") or "").lower()
    form_data: dict[str, Any] = {}
    upload: Any = None

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        upload = form.get("file")
        for key in form.keys():
            if key != "file":
                form_data[key] = form.get(key)
    else:
        try:
            parsed = await request.json()
        except Exception:
            parsed = {}
        if isinstance(parsed, dict):
            form_data = parsed

    token = form_data.get("token") or request.query_params.get("token")
    claims = _verify_jnf_token(token)

    raw_text = ""
    if upload is not None and hasattr(upload, "file") and getattr(upload, "filename", ""):
        suffix = Path(upload.filename or "").suffix.lower()
        data = _read_capped_jd(upload)
        raw_text = extract_jd_text(data, suffix)
    else:
        raw_text = normalize_jd_text(str(form_data.get("text") or form_data.get("raw_jd_text") or ""))

    trace_id = f"tr-{uuid4().hex[:8]}"
    base_doc: dict[str, Any] = {}
    if raw_text.strip():
        base_doc = parse_campus_jd(
            raw_text,
            mode="auto",
            ctx=ctx,
            trace_id=trace_id,
            session=session,
            actor_role="external_recruiter",
            origin="recruiter_jnf",
        )

    # Overlay explicit structured recruiter fields if supplied
    merged = dict(base_doc) if base_doc else {"eligibility": {}}
    elig = dict(merged.get("eligibility") or {})
    for key in ("company", "role_title", "location", "work_mode", "experience_level"):
        val = str(form_data.get(key) or "").strip()
        if val:
            merged[key] = val
    for key in ("ctc_lpa_min", "ctc_lpa_max", "stipend_inr_per_month"):
        if form_data.get(key) not in (None, ""):
            merged[key] = form_data.get(key)
    if "salary_disclosed" in form_data and form_data.get("salary_disclosed") not in (None, ""):
        merged["salary_disclosed"] = _parse_bool(form_data.get("salary_disclosed"))

    # Nested eligibility or flat form keys
    nested_elig = form_data.get("eligibility") if isinstance(form_data.get("eligibility"), dict) else {}
    for ekey in ("min_cgpa", "max_active_backlogs", "graduation_year"):
        val = nested_elig.get(ekey) if ekey in nested_elig else form_data.get(ekey)
        if val not in (None, ""):
            elig[ekey] = val
    branches_val = (
        nested_elig.get("allowed_branches")
        if "allowed_branches" in nested_elig
        else form_data.get("allowed_branches")
    )
    if branches_val not in (None, ""):
        elig["allowed_branches"] = _parse_list_field(branches_val)
    merged["eligibility"] = elig

    if form_data.get("required_skills") not in (None, ""):
        merged["required_skills"] = _parse_list_field(form_data.get("required_skills"))
    if form_data.get("preferred_skills") not in (None, ""):
        merged["preferred_skills"] = _parse_list_field(form_data.get("preferred_skills"))
    if raw_text.strip():
        merged["raw_jd_text"] = raw_text

    if not str(merged.get("company") or "").strip() or not str(merged.get("role_title") or "").strip():
        raise HTTPException(400, "company and role_title (or a JD containing them) are required")

    validated = validate_drive_payload(merged, ctx.taxonomy)
    drive, _ap = _persist_pending_drive(
        session,
        drive_id=base_doc.get("id") or f"cd-{uuid4().hex[:8]}",
        trace_id=trace_id,
        validated=validated,
        origin="recruiter_jnf",
        created_by="external_recruiter",
        actor_role="external_recruiter",
        audit_action="jnf.submitted",
        extra_audit_detail={"jnf_id": claims.get("jnf_id")},
    )
    return drive_to_dict(drive)


@router.get("/api/campus-drives/{drive_id}")
def get_campus_drive(
    drive_id: str,
    session: Session = Depends(get_session),
    _role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
) -> dict[str, Any]:
    drive = session.get(CampusDrive, drive_id)
    if drive is None:
        raise HTTPException(404, "campus drive not found")
    return drive_to_dict(drive)


@router.patch("/api/campus-drives/{drive_id}")
def update_campus_drive(
    drive_id: str,
    body: dict[str, Any],
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
    role: str = Depends(require_role("tpo_admin")),
) -> dict[str, Any]:
    drive = session.get(CampusDrive, drive_id)
    if drive is None:
        raise HTTPException(404, "campus drive not found")
    if drive.approval_status not in ("DRAFT", "PENDING_TPO_APPROVAL"):
        raise HTTPException(409, f"cannot edit drive in status {drive.approval_status}")
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON body required")

    before = drive_to_dict(drive)
    merged = dict(before)
    for k in (
        "company", "role_title", "location", "work_mode", "ctc_lpa_min", "ctc_lpa_max",
        "salary_disclosed", "stipend_inr_per_month", "experience_level",
        "required_skills", "preferred_skills", "unmapped_skill_mentions", "raw_jd_text",
    ):
        if k in body:
            merged[k] = body[k]
    if "eligibility" in body and isinstance(body["eligibility"], dict):
        merged["eligibility"] = {**before["eligibility"], **body["eligibility"]}

    if not str(merged.get("company") or "").strip() or not str(merged.get("role_title") or "").strip():
        raise HTTPException(400, "company and role_title cannot be empty")

    validated = validate_drive_payload(merged, ctx.taxonomy)

    drive.company = validated["company"]
    drive.role_title = validated["role_title"]
    drive.location = validated["location"]
    drive.work_mode = validated["work_mode"]
    drive.ctc_lpa_min = validated["ctc_lpa_min"]
    drive.ctc_lpa_max = validated["ctc_lpa_max"]
    drive.salary_disclosed = validated["salary_disclosed"]
    drive.stipend_inr_per_month = validated["stipend_inr_per_month"]
    drive.policy_tier = validated["policy_tier"]
    drive.policy_tier_source = validated["policy_tier_source"]
    drive.experience_level = validated["experience_level"]
    drive.eligibility = validated["eligibility"]
    drive.required_skills = validated["required_skills"]
    drive.preferred_skills = validated["preferred_skills"]
    drive.unmapped_skill_mentions = validated["unmapped_skill_mentions"]
    drive.raw_jd_text = validated["raw_jd_text"]
    drive.parser_metadata = validated["parser_metadata"]
    drive.updated_at = datetime.now(timezone.utc)
    session.add(drive)

    after = drive_to_dict(drive)
    diff = {
        k: {"from": before.get(k), "to": after.get(k)}
        for k in (
            "company", "role_title", "location", "work_mode", "ctc_lpa_min", "ctc_lpa_max",
            "salary_disclosed", "stipend_inr_per_month", "policy_tier", "experience_level",
            "eligibility", "required_skills", "preferred_skills",
        )
        if before.get(k) != after.get(k)
    }

    if drive.approval_id:
        ap = session.get(ApprovalRequest, drive.approval_id)
        if ap is not None:
            title, summary, ap_payload = _approval_snapshot(drive)
            ap.title = title
            ap.summary = summary
            ap.payload = ap_payload
            session.add(ap)

    log_audit(
        session,
        action="campus_drive.edited",
        entity_type="campus_drive",
        entity_id=drive.id,
        actor_role=role,
        detail={"trace_id": drive.trace_id, "diff": diff},
    )
    session.commit()
    session.refresh(drive)
    return drive_to_dict(drive)


@router.post("/api/campus-drives/{drive_id}/approve")
def approve_campus_drive(
    drive_id: str,
    payload: dict[str, Any] | None = None,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
    role: str = Depends(require_role("tpo_admin")),
) -> dict[str, Any]:
    drive = session.get(CampusDrive, drive_id)
    if drive is None:
        raise HTTPException(404, "campus drive not found")
    if not drive.approval_id:
        raise HTTPException(409, "campus drive has no linked approval request")
    res = approvals_routes.approve(
        drive.approval_id,
        payload=payload,
        session=session,
        ctx=ctx,
        role=role,
    )
    session.refresh(drive)
    return {**res, "campus_drive": drive_to_dict(drive)}


@router.post("/api/campus-drives/{drive_id}/reject")
def reject_campus_drive(
    drive_id: str,
    payload: dict[str, Any] | None = None,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("tpo_admin")),
) -> dict[str, Any]:
    drive = session.get(CampusDrive, drive_id)
    if drive is None:
        raise HTTPException(404, "campus drive not found")
    if not drive.approval_id:
        raise HTTPException(409, "campus drive has no linked approval request")
    res = approvals_routes.reject(
        drive.approval_id,
        payload=payload,
        session=session,
        role=role,
    )
    session.refresh(drive)
    return {**res, "campus_drive": drive_to_dict(drive)}


@router.get("/jnf/{token}", response_class=HTMLResponse)
def public_jnf_page(token: str, request: Request) -> HTMLResponse:
    """Public recruiter JNF intake form guarded by signed expirable token."""
    claims = _verify_jnf_token(token)
    from app.config import settings as _settings
    is_strict = str(getattr(_settings, "auth_mode", "demo")).strip().lower() == "strict"
    allow_demo = bool(getattr(_settings, "demo_role_switch", True))
    try:
        return request.app.state.templates.TemplateResponse(
            request,
            "campus/jnf_public.html",
            {
                "token": token,
                "jnf_id": claims.get("jnf_id", ""),
                "expires_at": claims.get("exp"),
                "demo_role_switch": allow_demo,
                "auth_mode": "strict" if is_strict else "demo",
            },
        )
    except TemplateNotFound:
        html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>JKLU Recruiter JNF Submission</title></head>
<body>
  <h1>JKLU CareerOS — Recruiter Job Notification Form</h1>
  <form method="post" action="/api/campus-drives/public-jnf" enctype="multipart/form-data">
    <input type="hidden" name="token" value="{token}" />
    <label>Company <input name="company" /></label><br/>
    <label>Role Title <input name="role_title" /></label><br/>
    <label>Upload JD (.pdf/.docx/.txt) <input type="file" name="file" /></label><br/>
    <label>Or Paste JD <textarea name="raw_jd_text"></textarea></label><br/>
    <button type="submit">Submit for TPO Review</button>
  </form>
</body>
</html>"""
        return HTMLResponse(content=html, status_code=200)
