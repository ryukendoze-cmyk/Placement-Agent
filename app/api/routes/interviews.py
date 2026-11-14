import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session

# Imported at module level so the §19.5 re-rank after finalization runs through
# this name (also lets tests spy on the re-rank trigger).
from app.api.routes.students import _rank_and_persist
from app.interview.bank import Question, get_question, load_bank
from app.interview.engine import InterviewSession, role_for_job
from app.interview.feedback_loop import QuestionTelemetryManager
from app.interview.quality import audit_question_bank, load_vetted_bank
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
_ORIG_LOAD_BANK = load_bank

#: Skill-promotion thresholds for mock-interview verification (ARCHITECTURE §19.5.1).
INTERVIEW_PROMOTE_SCORE = 80
INTERVIEW_REVERIFY_SCORE = 70
INTERVIEW_REMEDIAL_SCORE = 45
MAX_AUDIO_BYTES = 15 * 1024 * 1024


def _load_session_bank():
    if load_bank is not _ORIG_LOAD_BANK:
        return load_bank()
    vetted = load_vetted_bank()
    return vetted if vetted else load_bank()


def _deserialize_probes(raw_list) -> list[Question]:
    out: list[Question] = []
    for item in raw_list or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        out.append(
            Question(
                id=str(item["id"]),
                role=str(item.get("role") or "generic"),
                skill_id=str(item.get("skill_id") or "generic"),
                difficulty=int(item.get("difficulty", 2)),
                question=str(item.get("question") or ""),
                keywords=tuple(item.get("keywords") or ()),
                outline=tuple(item.get("outline") or ()),
            )
        )
    return out


def _serialize_probes(questions: list[Question]) -> list[dict]:
    return [
        {
            "id": q.id,
            "role": q.role,
            "skill_id": q.skill_id,
            "difficulty": q.difficulty,
            "question": q.question,
            "keywords": list(q.keywords),
            "outline": list(q.outline),
        }
        for q in (questions or [])
    ]


def _session(session: Session, interview: Interview, ctx, enable_followups: bool = False) -> InterviewSession:
    job = session.get(Job, interview.job_id) if interview.job_id else None
    persisted_role = str(getattr(interview, "role", None) or "").strip()
    role = persisted_role or (role_for_job(job) if job else "backend")

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

    # M18.2: reuse cached resume-grounded probes stored on interview.feedback when
    # available so multi-turn interviews do not repeat probe synthesis calls on every turn.
    resume_questions: list[Question] = []
    if getattr(interview, "include_resume_probes", False) and interview.student_id:
        fb = interview.feedback if isinstance(interview.feedback, dict) else {}
        if "_resume_probes" in fb:
            resume_questions = _deserialize_probes(fb.get("_resume_probes"))
        elif not any(str(t.get("q_id", "")).startswith("q-resume-") for t in (interview.transcript or [])):
            from app.interview.resume_probe import generate_resume_project_questions
            student = session.get(Student, interview.student_id)
            if student is not None:
                resume_questions = generate_resume_project_questions(
                    student, sorted_candidate_skills, getattr(ctx, "taxonomy", None),
                    llm=getattr(ctx, "llm", None))

    # M18.1: adaptive follow-up probing is opt-in (flag stored per-interview when
    # requested at creation, or passed per-answer via ``adaptive_followup``).
    followups = bool(enable_followups) or bool(getattr(interview, "adaptive_followup", False))

    return InterviewSession(role, _load_session_bank(), ctx.llm, prefer, interview.transcript or [],
                            enable_followups=followups,
                            resume_questions=resume_questions)


def _promote_skills_from_summary(session: Session, student_id: str | None, summary) -> list[dict]:
    """Closed-loop skill verification from interview per_skill scores."""
    if not student_id or not getattr(summary, "per_skill", None):
        return []
    rows = {
        ss.skill_id: ss
        for ss in session.exec(
            select(StudentSkill).where(StudentSkill.student_id == student_id)
        ).all()
    }
    now = datetime.now(timezone.utc)
    changed = False
    updates: list[dict] = []
    remedial_skills: list[str] = []

    for skill_id, raw_score in summary.per_skill.items():
        try:
            score = max(0, min(100, round(float(raw_score))))
        except (TypeError, ValueError):
            continue
        row = rows.get(skill_id)
        prev_status = str(getattr(row, "verification_status", "claimed") or "claimed").lower() if row else None
        current_level = str(getattr(row, "level", "basic") or "basic").lower() if row else "basic"

        is_reverify_pass = (
            score >= INTERVIEW_REVERIFY_SCORE
            and row is not None
            and (prev_status in ("unverified", "needs_reverification") or current_level == "basic")
        )
        if score >= INTERVIEW_PROMOTE_SCORE or is_reverify_pass:
            quote = f"interview verified ({score}): {str(summary.feedback or '').strip()}"
            if row is None:
                if score >= INTERVIEW_PROMOTE_SCORE and skill_id and skill_id != "generic" and session.get(Skill, skill_id) is not None:
                    row = StudentSkill(
                        student_id=student_id,
                        skill_id=skill_id,
                        mentions=1,
                        level="intermediate",
                        evidence=[quote] if quote else [],
                        verification_status="verified",
                        confidence=round(max(0.7, min(1.0, score / 100.0)), 2),
                        source="interview",
                        last_verified_score=score,
                        last_verified_at=now,
                    )
                    session.add(row)
                    rows[skill_id] = row
                    changed = True
                    updates.append({
                        "skill_id": skill_id,
                        "score": score,
                        "previous_status": None,
                        "verification_status": "verified",
                        "level": row.level,
                        "confidence": row.confidence,
                    })
                continue

            if score >= INTERVIEW_PROMOTE_SCORE:
                if current_level == "basic":
                    row.level = "intermediate"
                elif current_level == "intermediate":
                    row.level = "advanced"
            row.verification_status = "verified"
            row.confidence = round(max(0.7, min(1.0, score / 100.0)), 2)
            row.source = "interview"
            row.last_verified_score = score
            row.last_verified_at = now
            if quote and quote not in row.evidence:
                row.evidence = list(row.evidence) + [quote]
            session.add(row)
            changed = True
            updates.append({
                "skill_id": skill_id,
                "score": score,
                "previous_status": prev_status,
                "verification_status": "verified",
                "level": row.level,
                "confidence": row.confidence,
            })
        elif score < INTERVIEW_REMEDIAL_SCORE:
            if skill_id and skill_id != "generic":
                remedial_skills.append(skill_id)
            if row is None:
                continue
            row.verification_status = "unverified"
            row.confidence = round(max(0.1, score / 100.0), 2)
            row.source = "interview"
            row.last_verified_score = score
            row.last_verified_at = now
            note = f"interview remedial gap ({score}): skill tested below threshold"
            if note not in row.evidence:
                row.evidence = list(row.evidence) + [note]
            session.add(row)
            changed = True
            updates.append({
                "skill_id": skill_id,
                "score": score,
                "previous_status": prev_status,
                "verification_status": "unverified",
                "level": row.level,
                "confidence": row.confidence,
            })
        else:
            if skill_id and skill_id != "generic":
                remedial_skills.append(skill_id)
            if row is None:
                continue
            row.verification_status = "needs_reverification"
            row.confidence = round(score / 100.0, 2)
            row.source = "interview"
            row.last_verified_score = score
            row.last_verified_at = now
            note = f"interview needs reverification ({score}): verify applied proficiency"
            if note not in row.evidence:
                row.evidence = list(row.evidence) + [note]
            session.add(row)
            changed = True
            updates.append({
                "skill_id": skill_id,
                "score": score,
                "previous_status": prev_status,
                "verification_status": "needs_reverification",
                "level": row.level,
                "confidence": row.confidence,
            })

    if changed:
        if updates:
            session.add(AuditLog(
                id=f"aud-{uuid4().hex[:8]}",
                action="skill.verification_updated",
                entity_type="student",
                entity_id=student_id,
                actor_role="system",
                detail={"updates": updates},
            ))
        session.commit()

    if remedial_skills:
        try:
            from app.plan.planner import ensure_remedial_plan_items
            ensure_remedial_plan_items(session, student_id, remedial_skills)
        except Exception:
            logger.warning("ensure_remedial_plan_items failed for %s", student_id, exc_info=True)

    return updates


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
    skill_updates = _promote_skills_from_summary(session, row.student_id, summary)
    if skill_updates:
        fb = dict(row.feedback or {})
        fb["skill_updates"] = skill_updates
        row.feedback = fb
        session.add(row)
        session.commit()
    # §19.5.2 — re-score matches + refresh the upskilling roadmap now that levels
    # moved. Always attempted (so the trigger is observable); any failure — e.g.
    # a missing ctx — is contained so it never undoes the finalize/promote writes.
    if student is None:
        student = session.get(Student, row.student_id) if row.student_id else None
    try:
        _rank_and_persist(session, ctx, row.student_id, student, include_scraped=True)
        # Closed-loop application sync: update match_score on active JobApplications
        from app.models import JobApplication
        active_apps = session.exec(
            select(JobApplication).where(
                JobApplication.student_id == row.student_id,
                JobApplication.status.in_(
                    ["submitted", "awaiting_mentor_endorsement", "awaiting_tpo_approval"]
                ),
            )
        ).all()
        if active_apps:
            match_by_job = {
                m.job_id: m.score
                for m in session.exec(
                    select(JobMatch).where(JobMatch.student_id == row.student_id)
                ).all()
            }
            for app_row in active_apps:
                if app_row.job_id in match_by_job:
                    app_row.match_score = float(match_by_job[app_row.job_id])
                    session.add(app_row)
            session.commit()
    except Exception:
        logger.warning("match re-rank after interview finalization failed", exc_info=True)
    summary_payload = {
        "score": summary.score,
        "per_skill": summary.per_skill,
        "feedback": summary.feedback,
        "rubric_overall": summary.rubric_overall,
        "skill_updates": skill_updates,
    }
    if speech_summary:
        summary_payload["speech_summary"] = speech_summary
    content = {
        "finished": True,
        "summary": summary_payload,
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
    resolved_role = str(role_arg).strip() if role_arg else (role_for_job(job) if job else "backend")
    iid = f"int-{uuid4().hex[:8]}"
    row = Interview(id=iid, student_id=sid, job_id=job_id_val, transcript=[], score=None,
                    feedback={}, finished=False, role=resolved_role)
    # M18.1/18.2: persist per-interview flags so each reconstructed turn keeps them.
    row.include_resume_probes = include_resume_probes
    row.adaptive_followup = adaptive_followup
    session.add(row)
    session.commit()
    s = _session(session, row, ctx, enable_followups=adaptive_followup)
    s.role = resolved_role
    if s.resume_questions:
        fb = dict(row.feedback or {})
        fb["_resume_probes"] = _serialize_probes(s.resume_questions)
        row.feedback = fb
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
    from app.security.pii import scrub_for_llm
    raw_text = (payload or {}).get("text") or (payload or {}).get("answer", "")
    text, _ = scrub_for_llm(str(raw_text))
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
    # Attach telemetry to the exact turn just answered (before any appended follow-up turn).
    answered_turns = s.answered
    if answered_turns:
        answered_turns[-1]["speech_analytics"] = speech
    elif s.transcript:
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
        "error_recovered": turn.error_recovered,
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
    fb = dict(row.feedback or {}) if isinstance(row.feedback, dict) else {}
    fb.pop("_resume_probes", None)
    return {
        "interview_id": row.id,
        "student_id": row.student_id,
        "job_id": row.job_id,
        "role": getattr(row, "role", None),
        "finished": row.finished,
        "score": row.score,
        "feedback": fb,
        "transcript": row.transcript,
        "speech_summary": fb.get("speech_summary"),
        "skill_updates": fb.get("skill_updates", []),
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
def get_question_quality_audit(
    role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
):
    audit_data = audit_question_bank()
    telemetry_data = QuestionTelemetryManager().analyze_feedback_loop()
    return {
        "audit": audit_data,
        "telemetry": telemetry_data,
        **audit_data,
    }


@router.post("/api/interview/questions/refine")
def refine_questions(role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    mgr = QuestionTelemetryManager()
    result = mgr.apply_calibration()
    telemetry = result["analysis"]
    return {
        "status": "calibrated" if result["calibrated_count"] > 0 else "no_outliers",
        "calibrated_count": result["calibrated_count"],
        "updates": result["updates"],
        "recommendations": telemetry.get("recommendations", []),
        "analysis": telemetry,
        "flags": {
            "too_easy": telemetry.get("flagged_too_easy", []),
            "too_hard": telemetry.get("flagged_too_hard", []),
            "keyword_mismatch": telemetry.get("flagged_keyword_mismatch", []),
        },
    }


@router.post("/api/speech/analyze")
def speech_analyze(
    payload: dict,
    role: str = Depends(require_role("student", "faculty_mentor", "tpo_admin")),
):
    """Stateless acoustic telemetry for one answer (Milestone 18.3)."""
    text = str((payload or {}).get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    return analyze_speech_delivery(text, (payload or {}).get("duration_seconds"))


@router.post("/api/speech/transcribe")
async def speech_transcribe(
    request: Request,
    file: UploadFile | None = File(default=None),
    role: str = Depends(require_role("student", "faculty_mentor", "tpo_admin")),
):
    """Transcribe uploaded audio via OpenAI Whisper / local faster-whisper when available,
    falling back gracefully with structured browser Web Speech guidance when offline."""
    transcript_hint = ""
    duration_seconds = None
    audio_bytes = b""
    content_type = ""
    filename = "recording.webm"

    if file is not None:
        audio_bytes = await file.read(MAX_AUDIO_BYTES + 1)
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(413, f"audio file exceeds {MAX_AUDIO_BYTES} byte limit")
        content_type = file.content_type or "audio/webm"
        filename = file.filename or "recording.webm"
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, dict):
            transcript_hint = str(body.get("transcript_hint") or "").strip()
            duration_seconds = body.get("duration_seconds")

    from app.config import settings
    if audio_bytes and settings.llm_provider == "openai" and settings.openai_api_key:
        try:
            import io

            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            buf = io.BytesIO(audio_bytes)
            buf.name = filename
            resp = client.audio.transcriptions.create(model="whisper-1", file=buf)
            text = str(getattr(resp, "text", "") or "").strip()
            if text:
                return {
                    "ok": True,
                    "engine": "openai_whisper",
                    "fallback": False,
                    "transcript": text,
                    "analytics": analyze_speech_delivery(text, duration_seconds),
                }
        except Exception as exc:
            logger.warning("OpenAI Whisper transcription failed (%s)", exc)

    if audio_bytes:
        try:
            import tempfile

            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1] or ".webm") as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                model = WhisperModel("tiny", device="cpu")
                segments, _ = model.transcribe(tmp.name)
                text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                return {
                    "ok": True,
                    "engine": "faster_whisper",
                    "fallback": False,
                    "transcript": text,
                    "analytics": analyze_speech_delivery(text, duration_seconds),
                }
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Local faster_whisper failed (%s)", exc)

    return {
        "ok": True,
        "engine": "browser_webspeech_fallback",
        "fallback": True,
        "audio_received_bytes": len(audio_bytes),
        "content_type": content_type,
        "transcript": transcript_hint,
        "detail": "Offline demo environment uses browser Web Speech STT + editable review.",
        "analytics": analyze_speech_delivery(transcript_hint, duration_seconds) if transcript_hint else None,
    }


@router.post("/api/speech/tts")
def speech_tts(
    payload: dict,
    role: str = Depends(require_role("student", "faculty_mentor", "tpo_admin")),
):
    """Synthesize prompt audio via OpenAI TTS when configured, or return clean Web Speech guidance."""
    text = str((payload or {}).get("text") or "").strip()[:1500]
    if not text:
        raise HTTPException(400, "text is required")
    voice = str((payload or {}).get("voice") or "alloy")[:32]

    from app.config import settings
    if settings.llm_provider == "openai" and settings.openai_api_key:
        try:
            import base64

            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            resp = client.audio.speech.create(model="tts-1", voice=voice, input=text)
            audio_b64 = base64.b64encode(resp.read()).decode("ascii")
            return {
                "ok": True,
                "engine": "openai_tts",
                "fallback": False,
                "mime_type": "audio/mpeg",
                "audio_b64": audio_b64,
                "text": text,
            }
        except Exception as exc:
            logger.warning("OpenAI TTS failed (%s); falling back to browser SpeechSynthesis", exc)

    return {
        "ok": True,
        "engine": "browser_speech_synthesis",
        "fallback": True,
        "text": text,
        "speech_params": {"rate": 0.98, "pitch": 1.0},
    }
