import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session

# Imported at module level so the §19.5 re-rank after finalization runs through
# this name (also lets tests spy on the re-rank trigger).
from app.api.routes.students import _rank_and_persist
from app.interview.bank import get_question, load_bank
from app.interview.engine import InterviewSession, role_for_job
from app.interview.feedback_loop import QuestionTelemetryManager
from app.interview.quality import audit_question_bank
from app.interview.speech_analytics import aggregate_speech_delivery, analyze_speech_delivery
from app.models import (
    AuditLog,
    Interview,
    Job,
    JobMatch,
    ProgressEvent,
    Skill,
    Student,
    StudentSkill,
)
from app.security.rbac import enforce_student_scope, require_role

router = APIRouter()
logger = logging.getLogger(__name__)

#: Skill-promotion thresholds for mock-interview verification (ARCHITECTURE §19.5.1).
INTERVIEW_PROMOTE_SCORE = 80
INTERVIEW_REMEDIAL_SCORE = 45


def _session(session: Session, interview: Interview, ctx, enable_followups: bool = False) -> InterviewSession:
    job = session.get(Job, interview.job_id) if interview.job_id else None
    role = role_for_job(job) if job else "backend"

    # Query candidate skills
    candidate_skills: list[StudentSkill] = []
    if interview.student_id:
        candidate_skills = list(
            session.exec(
                select(StudentSkill).where(StudentSkill.student_id == interview.student_id)
            ).all()
        )

    # Sort candidate skills by verified depth: advanced (3) > intermediate (2) > basic (1), mentions desc
    level_weights = {"advanced": 3, "intermediate": 2, "basic": 1}
    sorted_candidate_skills = sorted(
        candidate_skills,
        key=lambda s: (level_weights.get(str(s.level).lower(), 0), s.mentions or 0),
        reverse=True,
    )
    candidate_skill_ids = [s.skill_id for s in sorted_candidate_skills if s.skill_id]

    if job and job.required_skills:
        job_req = list(job.required_skills)
        intersect = [s for s in candidate_skill_ids if s in job_req]
        candidate_rem = [s for s in candidate_skill_ids if s not in intersect]
        job_rem = [s for s in job_req if s not in intersect]
        prefer = intersect + candidate_rem + job_rem
    else:
        prefer = candidate_skill_ids

    # M18.2: resume-grounded probes are regenerated deterministically each turn
    # (idempotent) and served before static bank questions via the session.
    resume_questions = []
    if getattr(interview, "include_resume_probes", False) and interview.student_id:
        from app.interview.resume_probe import generate_resume_project_questions
        student = session.get(Student, interview.student_id)
        if student is not None:
            resume_questions = generate_resume_project_questions(
                student, sorted_candidate_skills, getattr(ctx, "taxonomy", None),
                llm=getattr(ctx, "llm", None))

    # M18.1: adaptive follow-up probing is opt-in (flag stored per-interview when
    # requested at creation, or passed per-answer via ``adaptive_followup``).
    followups = bool(enable_followups) or bool(getattr(interview, "adaptive_followup", False))

    return InterviewSession(role, load_bank(), ctx.llm, prefer, interview.transcript or [],
                            enable_followups=followups,
                            resume_questions=resume_questions)


def _promote_skills_from_summary(session: Session, student_id: str | None, summary) -> None:
    """§19.5.1 — promote verified skill levels from the interview's per_skill scores.

    Score >= INTERVIEW_PROMOTE_SCORE promotes basic -> intermediate -> advanced and
    records a verification quote in StudentSkill.evidence. Score < INTERVIEW_REMEDIAL_SCORE
    records an empirical remedial gap in evidence (never a promotion). Mid-band scores
    leave the level untouched. ``per_skill`` values may arrive as strings on the wire.
    """
    if not student_id or not getattr(summary, "per_skill", None):
        return
    rows = {
        ss.skill_id: ss
        for ss in session.exec(
            select(StudentSkill).where(StudentSkill.student_id == student_id)
        ).all()
    }
    changed = False
    for skill_id, raw_score in summary.per_skill.items():
        try:
            score = round(float(raw_score))
        except (TypeError, ValueError):
            continue
        row = rows.get(skill_id)
        if score >= INTERVIEW_PROMOTE_SCORE:
            quote = f"interview verified ({score}): {str(summary.feedback or '').strip()}"
            if row is None:
                # A promoted skill the student did not previously own: create the
                # row only for a real taxonomy skill (never "generic" or an
                # unseeded id, which would violate the StudentSkill -> Skill FK).
                if skill_id and skill_id != "generic" and session.get(Skill, skill_id) is not None:
                    row = StudentSkill(
                        student_id=student_id, skill_id=skill_id, mentions=1,
                        level="intermediate", evidence=[quote] if quote else [])
                    session.add(row)
                    rows[skill_id] = row  # keep the dict current for later iterations
                    changed = True
                continue
            current = str(row.level or "basic").lower()
            if current == "basic":
                row.level = "intermediate"
            elif current == "intermediate":
                row.level = "advanced"
            # advanced stays advanced; still record the verification.
            if quote and quote not in row.evidence:
                row.evidence = list(row.evidence) + [quote]
            session.add(row)
            changed = True
        elif score < INTERVIEW_REMEDIAL_SCORE:
            if row is None:
                continue
            note = f"interview remedial gap ({score}): skill tested below threshold"
            if note not in row.evidence:
                row.evidence = list(row.evidence) + [note]
            session.add(row)
            changed = True
    if changed:
        session.commit()


def _aggregate_speech(transcript: list) -> dict:
    """Aggregate per-turn ``speech_analytics`` into a session ``speech_summary`` (M18.3)."""
    return aggregate_speech_delivery(transcript)


def _finalize(session: Session, row: Interview, summary, turn=None,
              ctx=None, student=None) -> JSONResponse:
    row.transcript = summary.transcript
    row.score = summary.score
    row.feedback = {"overall": summary.feedback, "per_skill": summary.per_skill,
                    "rubric_overall": summary.rubric_overall}
    # M18.3: aggregate per-turn acoustic telemetry into the finalized feedback.
    speech_summary = _aggregate_speech(summary.transcript)
    if speech_summary:
        row.feedback["speech_summary"] = speech_summary
    row.finished = True
    session.add(row)
    # completion side-effects (single transaction with the finalize write).
    # idempotent: only reached when row.finished was False, and the 409 path
    # above never calls _finalize, so no duplicate event/audit rows.
    session.add(ProgressEvent(
        id=f"pe-{uuid4().hex[:8]}", student_id=row.student_id,
        kind="interview_completed", ref_id=row.id, readiness=int(summary.score),
        at=datetime.now(timezone.utc)))
    session.add(AuditLog(
        id=f"aud-{uuid4().hex[:8]}", action="interview.completed",
        entity_type="interview", entity_id=row.id, actor_role="student",
        detail={"score": int(summary.score), "job_id": row.job_id,
                "student_id": row.student_id, "per_skill": summary.per_skill}))
    session.commit()
    # §19.5.1 — promote verified skills from this interview's scores.
    _promote_skills_from_summary(session, row.student_id, summary)
    # §19.5.2 — re-score matches + refresh the upskilling roadmap now that levels
    # moved. Always attempted (so the trigger is observable); any failure — e.g.
    # a missing ctx — is contained so it never undoes the finalize/promote writes.
    if student is None:
        student = session.get(Student, row.student_id) if row.student_id else None
    try:
        _rank_and_persist(session, ctx, row.student_id, student, include_scraped=True)
    except Exception:
        logger.warning("match re-rank after interview finalization failed", exc_info=True)
    content = {
        "finished": True,
        "summary": {"score": summary.score, "per_skill": summary.per_skill,
                    "feedback": summary.feedback,
                    "rubric_overall": summary.rubric_overall},
    }
    if turn is not None:
        content.update({
            "score": turn.score,
            "feedback": turn.feedback,
            "next_difficulty": turn.next_difficulty,
            "rubric": turn.rubric,
            "matched_keywords": turn.matched_keywords,
            "missed_keywords": turn.missed_keywords,
            "ideal_answer_hint": turn.ideal_answer_hint,
            "followup_probe": turn.followup_probe,
        })
    return JSONResponse(content=content)


@router.post("/api/students/{sid}/interviews")
def start_interview(sid: str, request: Request, payload: dict | None = None,
                    session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    if ctx.llm is None:
        raise HTTPException(503, "LLM not configured")
    payload = payload or {}
    job_id = payload.get("job_id")
    role_arg = payload.get("role")
    include_resume_probes = bool(payload.get("include_resume_probes"))
    adaptive_followup = bool(payload.get("adaptive_followup"))
    job = None
    if job_id:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
    elif role_arg:
        jobs = session.exec(select(Job)).all()
        matching_jobs = [j for j in jobs if role_for_job(j) == role_arg]
        job = matching_jobs[0] if matching_jobs else (jobs[0] if jobs else None)
    else:
        matches = session.exec(select(JobMatch)
                               .where(JobMatch.student_id == sid)
                               .order_by(JobMatch.score.desc())).all()
        if not matches:
            raise HTTPException(400, "no matches computed yet; GET /api/students/{sid}/matches first")
        job = session.get(Job, matches[0].job_id)
        if job is None:
            raise HTTPException(404, "job not found")

    job_id_val = job.id if job else "j-general"
    iid = f"int-{uuid4().hex[:8]}"
    row = Interview(id=iid, student_id=sid, job_id=job_id_val, transcript=[], score=None,
                    feedback={}, finished=False)
    # M18.1/18.2: persist per-interview flags so each reconstructed turn keeps them.
    row.include_resume_probes = include_resume_probes
    row.adaptive_followup = adaptive_followup
    session.add(row)
    session.commit()
    s = _session(session, row, ctx, enable_followups=adaptive_followup)
    if role_arg:
        s.role = role_arg
    q = s.next_question()
    row.transcript = s.transcript
    session.add(row)
    session.commit()
    return JSONResponse(status_code=201, content={
        "id": iid, "interview_id": iid, "job_id": job_id_val, "role": s.role,
        "difficulty": s.current_difficulty,
        "question": {"id": q.id, "question": q.question, "skill_id": q.skill_id,
                     "difficulty": q.difficulty} if q else None,
    })


@router.post("/api/interviews/{iid}/answers")
@router.post("/api/interviews/{iid}/answer")
def answer_interview(iid: str, request: Request, payload: dict,
                     session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    row = session.get(Interview, iid)
    if row is None:
        raise HTTPException(404, "interview not found")
    if row.student_id:
        enforce_student_scope(request, row.student_id)
    if row.finished:
        raise HTTPException(409, "interview already finished")
    if ctx.llm is None:
        raise HTTPException(503, "LLM not configured")
    text = (payload or {}).get("text") or (payload or {}).get("answer", "")
    # M18.1: allow enabling adaptive follow-ups per-answer (also settable at creation).
    adaptive = bool((payload or {}).get("adaptive_followup")) or bool(
        getattr(row, "adaptive_followup", False))
    s = _session(session, row, ctx, enable_followups=adaptive)
    try:
        turn = s.answer(text)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # M18.3: acoustic telemetry for this turn (duration optional; typed => wpm 0).
    raw_duration = (payload or {}).get("duration_seconds")
    speech = analyze_speech_delivery(text, raw_duration)
    # attach to the pending (now answered) transcript turn
    if s.transcript:
        s.transcript[-1]["speech_analytics"] = speech
    if turn.injection_flags:
        # M15: record the neutralized attempt (append-only audit row).
        from app.security.audit import log_audit
        log_audit(session, action="security.prompt_injection_blocked",
                  entity_type="interview", entity_id=iid,
                  actor_role="student",
                  detail={"flags": turn.injection_flags,
                          "student_id": row.student_id,
                          "question_id": turn.question_id},
                  commit=True)
    try:
        q_obj = get_question(load_bank(), turn.question_id)
        # Only record calibration telemetry for STATIC bank questions; transient
        # dynamic ids (q-*-followup, q-resume-*) must never pollute the ledger.
        if q_obj is not None:
            QuestionTelemetryManager().record_answer_turn(
                q_id=turn.question_id,
                score=turn.score,
                matched_kw=turn.matched_keywords,
                expected_kw=list(q_obj.keywords),
                difficulty=q_obj.difficulty,
            )
    except Exception:
        pass
    if turn.finished:
        student = session.get(Student, row.student_id) if row.student_id else None
        return _finalize(session, row, s.finish(), turn=turn, ctx=ctx, student=student)
    nq = s.next_question()
    if nq is None:
        # question bank exhausted before MAX_TURNS: end gracefully instead of
        # leaving the interview in a non-completable, non-finished state
        student = session.get(Student, row.student_id) if row.student_id else None
        return _finalize(session, row, s.finish(), turn=turn, ctx=ctx, student=student)
    row.transcript = s.transcript
    session.add(row)
    session.commit()
    return JSONResponse(content={
        "finished": False,
        "score": turn.score,
        "feedback": turn.feedback,
        "next_difficulty": turn.next_difficulty,
        "rubric": turn.rubric,
        "matched_keywords": turn.matched_keywords,
        "missed_keywords": turn.missed_keywords,
        "ideal_answer_hint": turn.ideal_answer_hint,
        "injection_flags": turn.injection_flags,
        "followup_probe": turn.followup_probe,
        "speech_analytics": speech,
        "question": {"id": nq.id, "question": nq.question, "skill_id": nq.skill_id,
                     "difficulty": nq.difficulty},
    })


@router.get("/api/interviews/{iid}")
def get_interview(iid: str, request: Request, session: Session = Depends(get_session)):
    row = session.get(Interview, iid)
    if row is None:
        raise HTTPException(404, "interview not found")
    if row.student_id:
        enforce_student_scope(request, row.student_id)
    return {
        "interview_id": row.id,
        "student_id": row.student_id,
        "job_id": row.job_id,
        "finished": row.finished,
        "score": row.score,
        "feedback": row.feedback,
        "transcript": row.transcript,
        "started_at": row.started_at.isoformat(),
    }


@router.get("/interviews/{iid}")
def interview_page(iid: str, request: Request, session: Session = Depends(get_session)):
    row = session.get(Interview, iid)
    if row is None:
        raise HTTPException(404, "interview not found")
    if row.student_id:
        enforce_student_scope(request, row.student_id)
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "student/interview.html", {
        "interview_id": iid,
        "sid": row.student_id,
    })


@router.get("/api/interview/questions/quality")
def get_question_quality_audit():
    audit_data = audit_question_bank()
    telemetry_data = QuestionTelemetryManager().analyze_feedback_loop()
    return {
        "audit": audit_data,
        "telemetry": telemetry_data,
        **audit_data,
    }


@router.post("/api/interview/questions/refine")
def refine_questions(role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    telemetry = QuestionTelemetryManager().analyze_feedback_loop()
    calibrated_count = (
        len(telemetry.get("flagged_too_easy", []))
        + len(telemetry.get("flagged_too_hard", []))
        + len(telemetry.get("flagged_keyword_mismatch", []))
    )
    return {
        "status": "calibrated",
        "calibrated_count": calibrated_count,
        "recommendations": telemetry.get("recommendations", []),
        "analysis": telemetry,
        "flags": {
            "too_easy": telemetry.get("flagged_too_easy", []),
            "too_hard": telemetry.get("flagged_too_hard", []),
            "keyword_mismatch": telemetry.get("flagged_keyword_mismatch", []),
        },
    }


@router.post("/api/speech/analyze")
def speech_analyze(payload: dict):
    """Stateless acoustic telemetry for one answer (Milestone 18.3)."""
    text = str((payload or {}).get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    return analyze_speech_delivery(text, (payload or {}).get("duration_seconds"))
