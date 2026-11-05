from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.models import Student
from app.rag.policy_rag import (
    evaluate_student_eligibility,
    search_sops,
    sop_ids,
    verify_policy_citations,
)
from app.security.ratelimit import check_rate_limit
from app.security.rbac import current_role, enforce_student_scope, get_current_user_optional

router = APIRouter()

QUICK_PROMPTS = [
    "What score do I need for Dream drives?",
    "Am I eligible for Super-Dream offers?",
    "How many backlogs are allowed?",
    "What does the resume-integrity policy say?",
]

COPILOT_SYSTEM = (
    "TASK: copilot_answer\n"
    "You are the JKLU Placement Copilot. Answer only from the provided student context and "
    "retrieved SOP clauses. Cite relevant SOPs as [SOP-XX]. Respond with JSON: "
    '{"answer": str, "cited_sops": list[str]}.'
)


def _student_context(session, student_id: str | None) -> dict:
    if not student_id:
        return {}
    from app.models import Interview, Student
    student = session.get(Student, student_id)
    if student is None:
        return {}
    from sqlmodel import select
    rows = session.exec(select(Interview).where(Interview.student_id == student_id,
                                                Interview.finished == True)).all()  # noqa: E712
    scores = [int(r.score) for r in rows if r.score is not None]
    return {
        "name": student.name,
        "gpa": float(student.gpa or 0.0),
        "branch": student.branch,
        "backlogs": int(getattr(student, "backlogs", 0) or 0),
        "mock_interview_score": max(scores) if scores else None,
        "mock_interview_passed": bool(scores) and max(scores) >= 65,
    }


def _copilot_core(session, ctx, *, question: str, student_id: str | None,
                  job_id: str | None, actor_role: str) -> dict:
    """Shared retrieval + synthesis + citation-guard pipeline for ask endpoints.

    Returns the full answer dict; ``copilot_ask`` returns it directly while the
    streaming endpoint reuses it to build ``retrieval``/``complete`` events.
    """
    # 1) Retrieval
    search = search_sops(question, top_k=3, embedder=ctx.embedder)
    sources = search.get("results", [])

    # 2) Student context (graceful if missing/invalid)
    student_context = _student_context(session, student_id)

    # 3) Optional eligibility summary (if both ids valid)
    eligibility = {}
    if student_id and job_id:
        try:
            eligibility = evaluate_student_eligibility(session, ctx, student_id, job_id)
        except Exception:
            eligibility = {}

    # 4) Synthesis (MockLLM deterministic, or real LLM via get_llm)
    user = {"question": question, "student": student_context,
            "sources": sources, "eligibility": eligibility}
    import json as _json
    raw = ""
    if getattr(ctx, "llm", None) is not None:
        raw = ctx.llm.complete(system=COPILOT_SYSTEM, user=_json.dumps(user), json_mode=True)
    try:
        parsed = _json.loads(raw) if raw else {}
    except Exception:
        parsed = {}
    answer = str(parsed.get("answer") or "").strip()
    if not answer:
        answer = (f"Based on JKLU placement policy, your inquiry '{question}' is governed by the "
                  f"retrieved sections: {', '.join(s.get('sop_id','') for s in sources)}.")
    cited = [str(c) for c in (parsed.get("cited_sops") or [])]

    # 5) Citation guard (annotate unknown, keep known)
    from app.rag.policy_rag import sop_ids
    known = sop_ids()
    verify = verify_policy_citations(answer, known)
    # ensure cited_sops are all known; drop unknown ones from the trusted list
    cited = [c.upper() for c in cited if c.upper() in known]

    # Build a mini trace
    trace = [{
        "thought": "Queried institutional placement policy for grounded clauses.",
        "tool": "search_placement_policy_rag",
        "top_sop": search.get("top_sop_id"),
    }]

    return {
        "question": question,
        "student_id": student_id,
        "job_id": job_id,
        "answer": verify["annotated_text"],
        "citations": {
            "verified": sorted(set(verify["verified"]) | set(cited)),
            "unverified": verify["unverified"],
            "grounding_score": verify["grounding_score"],
        },
        "sources": sources,
        "student_context": student_context,
        "eligibility": eligibility or None,
        "trace": trace,
        "actor_role": actor_role,
    }


@router.post("/api/copilot/ask/stream")
def copilot_ask_stream(payload: dict, request: Request,
                       session: Session = Depends(get_session), ctx=Depends(get_ctx),
                       _rl: None = Depends(check_rate_limit("copilot_ask"))):
    """Stream the copilot answer as SSE (Milestone 17).

    Emits ``retrieval`` (sources + student context + eligibility + trace), then
    ``token`` events carrying the answer synthesized by the streaming LLM path,
    then a final ``complete`` with the citation-verified annotated text.
    """
    from app.api.routes.jobs import SSE_HEADERS, _sse_chunk

    question = str((payload or {}).get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    student_id = (payload or {}).get("student_id") or None
    job_id = (payload or {}).get("job_id") or None
    actor_role = (payload or {}).get("actor_role", "student")
    user = get_current_user_optional(request)
    if user is not None and user.role == "student":
        student_id = user.student_id
        actor_role = "student"

    def event_stream():
        try:
            # Shared core computes retrieval + citations (authoritative answer).
            result = _copilot_core(session, ctx, question=question,
                                   student_id=student_id, job_id=job_id,
                                   actor_role=actor_role)
            yield _sse_chunk("retrieval", {
                "sources": result["sources"],
                "student_context": result["student_context"],
                "eligibility": result["eligibility"],
                "trace": result["trace"],
            })
            # Stream the answer text via the LLM streaming path.
            llm = getattr(ctx, "llm", None)
            if llm is not None and hasattr(llm, "stream"):
                user_payload = {
                    "question": question,
                    "student": result["student_context"],
                    "sources": result["sources"],
                    "eligibility": result["eligibility"] or {},
                }
                import json as _json
                chunks = llm.stream(system=COPILOT_SYSTEM,
                                    user=_json.dumps(user_payload), json_mode=True)
                try:
                    for delta in chunks:
                        yield _sse_chunk("token", {"delta": delta})
                except Exception:
                    # fallback: emit the authoritative (verified) answer as one delta
                    yield _sse_chunk("token", {"delta": result["answer"]})
            yield _sse_chunk("complete", {
                "answer": result["answer"],
                "citations": result["citations"],
                "sources": result["sources"],
                "student_context": result["student_context"],
                "eligibility": result["eligibility"],
                "trace": result["trace"],
                "question": result["question"],
                "student_id": result["student_id"],
                "job_id": result["job_id"],
                "actor_role": result["actor_role"],
            })
        except Exception as e:  # pragma: no cover - never break the SSE frame
            yield _sse_chunk("error", {"message": str(e)})
        finally:
            session.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers=SSE_HEADERS)


@router.post("/api/copilot/ask")
def copilot_ask(payload: dict, request: Request,
                session: Session = Depends(get_session), ctx=Depends(get_ctx),
                _rl: None = Depends(check_rate_limit("copilot_ask"))):
    question = str((payload or {}).get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    student_id = (payload or {}).get("student_id") or None
    job_id = (payload or {}).get("job_id") or None
    actor_role = (payload or {}).get("actor_role", "student")

    # Row-level security (§17.2.B.2): a student's query is forcibly bound to
    # their own identity, so they cannot request another student's metrics,
    # GPA or transcript through the Copilot.
    user = get_current_user_optional(request)
    if user is not None and user.role == "student":
        student_id = user.student_id
        actor_role = "student"

    return _copilot_core(session, ctx, question=question, student_id=student_id,
                         job_id=job_id, actor_role=actor_role)


@router.get("/copilot")
def copilot_page(request: Request, session: Session = Depends(get_session)):
    role = current_role(request)
    user = get_current_user_optional(request)
    dept = user.department if user else None

    student_name = ""
    if role == "student":
        sid = user.student_id if user and user.student_id and user.student_id != "demo-student" else None
        st = session.get(Student, sid) if sid else None
        if st is None:
            st = session.exec(select(Student)).first()
        if st is not None:
            students = [st]
            student_name = st.name
            if user:
                user.name = st.name
                user.department = st.branch
                user.student_id = st.id
        else:
            students = []
    elif role == "faculty_mentor" and dept:
        all_s = session.exec(select(Student).order_by(Student.created_at.desc())).all()
        students = [s for s in all_s if s.branch and s.branch.upper() == dept.upper()]
    else:
        students = session.exec(select(Student).order_by(Student.created_at.desc())).all()

    return request.app.state.templates.TemplateResponse(
        request, "copilot.html",
        {"students": students, "sid": (user.student_id if user and role == "student" else ""), "active": "copilot",
         "role": role, "user": user, "current_user": user,
         "student_name": student_name,
         "department": dept,
         "user_name": (user.name if user else ""),
         "user_department": dept,
         "quick_prompts": QUICK_PROMPTS,
         "sop_ids": sorted(sop_ids()), "error": ""})


@router.get("/api/students/{sid}/eligibility/{job_id}")
def student_job_eligibility(sid: str, job_id: str, request: Request,
                            session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    from app.models import Job, Student
    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    if session.get(Job, job_id) is None:
        raise HTTPException(404, "job not found")
    try:
        result = evaluate_student_eligibility(session, ctx, sid, job_id)
    except Exception as e:
        raise HTTPException(400, str(e))
    return result
