import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlmodel import Session, select

from .llm.mock import MockLLM
from .matching.rank import rank_jobs
from .models import (
    AgentRun,
    AgentStep,
    ApprovalRequest,
    AuditLog,
    Course,
    Interview,
    Job,
    JobApplication,
    JobMatch,
    ProgressEvent,
    Skill,
    Student,
    StudentSkill,
)
from .plan.planner import build_plan
from .skills.extract import extract_skills, level_for
from .taxonomy.taxonomy import DATA_DIR, Taxonomy


def _json(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def seed(db: Session, taxonomy: Taxonomy, vector_store=None, embedder=None) -> None:
    for s in taxonomy.all():
        db.merge(Skill(id=s.id, name=s.name, category=s.category, aliases=list(s.aliases)))
    for j in _json("jobs.json"):
        db.merge(Job(**j))
    for c in _json("courses.json"):
        db.merge(Course(**c))
    db.commit()
    if vector_store is not None and embedder is not None:
        jobs = db.exec(select(Job)).all()
        if callable(getattr(vector_store, "upsert_many", None)):
            vector_store.upsert_many(
                [("job", j.id, embedder.embed(j.description)) for j in jobs])
        else:
            for j in jobs:
                vector_store.upsert("job", j.id, embedder.embed(j.description))


_DEMO_COHORT = [
    # (name, branch, year, gpa, resume text) - 12 students across CSE / AI&DS / ECE / IT
    # Resume density is tuned so the cohort spans multiple readiness bands
    # (strong / moderate / developing / early) for the analytics doughnut.
    ("Asha Patel", "CSE", 3, 8.2,
     "Skills: Python, FastAPI, PostgreSQL, Docker, Redis, Kubernetes, AWS, CI/CD, Linux, pytest, Git, SQL, rest_api, communication\n"
     "Backend engineer: built REST APIs with FastAPI and PostgreSQL, containerized with Docker/Kubernetes on AWS with CI/CD, tested with pytest."),
    ("Rohan Gupta", "CSE", 3, 7.8, "Skills: Python, SQL, Git, Linux\nData engineering internship."),
    ("Meera Iyer", "ECE", 3, 8.5,
     "Skills: typescript, react, tailwind, css, html, javascript_web, jest, redux, accessibility, responsive_design, git\n"
     "Frontend engineer: built responsive React apps with TypeScript and Tailwind, tested with Jest, WCAG accessibility."),
    ("Arjun Singh", "AI&DS", 4, 7.5,
     "Skills: sql, pandas, tableau, excel, python, statistics, powerbi, machine_learning, sklearn, data_visualization, ab_testing, etl\n"
     "Data analyst: dashboards in Tableau and Power BI, SQL analytics, A/B testing, ETL pipelines."),
    ("Diya Nair", "CSE", 4, 8.0,
     "Skills: terraform, kubernetes, ci_cd, linux, aws, monitoring, nginx, python, git, docker, ansible, bash\n"
     "DevOps/SRE: infrastructure with Terraform, clusters with Kubernetes, pipelines with CI/CD, Nginx, monitoring, on-call rotations."),
    ("Karan Mehta", "CSE", 3, 7.9, "Skills: Python, FastAPI, Docker, Git\nFull-stack app."),
    ("Ananya Rao", "ECE", 3, 8.1, "Skills: HTML, CSS, JavaScript, React\nResponsive web dev."),
    ("Vikram Shah", "AI&DS", 4, 7.7, "Skills: Python, SQL, deep_learning, pandas\nML capstone."),
    ("Sneha Kulkarni", "CSE", 4, 8.3, "Skills: Python, Git, AWS, Docker\nCloud projects."),
    ("Aditya Verma", "CSE", 3, 7.6, "Skills: React, TypeScript, CSS, HTML, SQL\nMERN stack."),
    ("Ishita Bose", "AI&DS", 3, 7.3, "Skills: Python, pandas, matplotlib, statistics\nData visualization portfolio."),
    ("Tanvi Reddy", "ECE", 4, 6.9, "Skills: HTML, CSS\nLearning web development; wants a frontend role."),
]


def _now(offset_minutes: int = 0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=offset_minutes)


def _seed_progress(db: Session, sid: str) -> None:
    """Multi-point readiness history so the progress chart is alive at startup."""
    ladder = [20, 34, 45, 58, 67, 75]
    kinds = ["resume_parsed", "matches_computed", "plan_created",
             "plan_item_done", "plan_item_done", "interview_completed"]
    for i, (ready, kind) in enumerate(zip(ladder, kinds)):
        db.add(ProgressEvent(id=f"pe-{uuid4().hex[:8]}", student_id=sid, kind=kind,
                             ref_id="", readiness=ready,
                             at=_now(offset_minutes=30 * (len(ladder) - i))))


def _seed_interview(db: Session, sid: str, job_id: str, llm) -> None:
    """A finished interview with a realistic 8-turn rubric transcript.

    Also writes the immutable AuditLog row the live endpoint would have
    written, so /admin/audit is populated at startup too.
    """
    from .interview.bank import load_bank
    from .interview.engine import InterviewSession, role_for_job
    from .interview.speech_analytics import (
        aggregate_speech_delivery,
        analyze_speech_delivery,
    )

    job = db.get(Job, job_id)
    role = role_for_job(job) if job else "backend"
    prefer = list(job.required_skills) if job else []
    s = InterviewSession(role, load_bank(), llm, prefer, transcript=[])
    s.next_question()
    for _ in range(8):
        s.answer("I explain the concept with a concrete example, name the trade-offs, "
                 "and mention lists, dict, joins, index, transactions where relevant.")
        if not s.finished:
            s.next_question()
    summary = s.finish()
    # M25: deterministic synthetic voice timing per turn (125/135/145/155 WPM,
    # all inside the optimal 110-160 band) so the studio shows real pacing,
    # filler and STAR telemetry computed by the live M18 engine.
    for i, turn in enumerate(summary.transcript):
        answer = str(turn.get("a") or "")
        if not answer:
            continue
        target_wpm = 125 + 10 * (i % 4)
        duration = round(max(1, len(answer.split())) / target_wpm * 60.0, 2)
        turn["speech_analytics"] = analyze_speech_delivery(answer, duration)
    feedback = {"overall": summary.feedback, "per_skill": summary.per_skill,
                "rubric_overall": summary.rubric_overall}
    speech_summary = aggregate_speech_delivery(summary.transcript)
    if speech_summary:
        feedback["speech_summary"] = speech_summary
    iid = f"int-{uuid4().hex[:8]}"
    db.add(Interview(id=iid, student_id=sid, job_id=job_id,
                     transcript=summary.transcript, score=summary.score,
                     feedback=feedback,
                     finished=True, started_at=_now(offset_minutes=5)))
    db.add(AuditLog(id=f"aud-{uuid4().hex[:8]}", action="interview.completed",
                    entity_type="interview", entity_id=iid, actor_role="student",
                    detail={"score": int(summary.score), "job_id": job_id,
                            "student_id": sid, "per_skill": summary.per_skill},
                    created_at=_now(offset_minutes=5)))


def _seed_agent_run(db: Session, sid: str, goal: str, tools: list[str]) -> None:
    rid = f"run-{uuid4().hex[:8]}"
    db.add(AgentRun(id=rid, student_id=sid, actor_role="tpo_admin", goal=goal,
                    status="completed",
                    final_answer=f"Completed '{goal}' with verified, evidence-backed steps.",
                    citations={"verified": ["SOP-01", "SOP-04"]},
                    total_latency_ms=412, provider_used="mock",
                    fallback_triggered=False, created_at=_now(offset_minutes=20)))
    for i, tool in enumerate(tools):
        db.add(AgentStep(id=f"ast-{uuid4().hex[:8]}", run_id=rid, step_order=i + 1,
                         thought=f"Decided to invoke {tool} to make progress on the goal.",
                         tool_name=tool, tool_input={"student_id": sid},
                         tool_output={"ok": True},
                         verification_passed=True, latency_ms=50 + i * 15,
                         created_at=_now(offset_minutes=20 - i)))


def _seed_approvals(db: Session, sids: dict[str, str]) -> None:
    """4 pending HITL requests so the approvals center is populated.

    Each request also writes the immutable AuditLog row the live endpoint
    would have written, so /admin/audit shows the queue at startup.
    `sids` maps a student name -> id so each approval binds to the student
    it names in its title.
    """
    asha_sid = sids.get("Asha Patel", "")
    diya_sid = sids.get("Diya Nair", asha_sid)
    rows = [
        ("cohort_bootcamp_publish", "python", "Cohort bridge bootcamp: Python",
         "Launch a bridge bootcamp for the top cohort skill gap.",
         {"skill_id": "python", "course_id": ""}),
        ("job_application_nomination", "job-001", "Drive nomination - Asha Patel",
         "Nominate the student for the top-matched drive.",
         {"student_id": asha_sid, "job_id": "job-001", "readiness": 75}),
        ("proficiency_waiver", "docker", "Proficiency waiver - Diya Nair",
         "Waive docker to 'advanced' with project evidence.",
         {"student_id": diya_sid, "skill_id": "docker", "level": "advanced",
          "new_level": "advanced"}),
        ("cohort_bootcamp_publish", "sql", "Cohort bridge bootcamp: SQL",
         "Second bootcamp for the runner-up cohort gap.",
         {"skill_id": "sql", "course_id": ""}),
    ]
    for kind, target, title, summary, payload in rows:
        aid = f"apr-{uuid4().hex[:8]}"
        db.add(ApprovalRequest(id=aid, kind=kind, target_id=target,
                               title=title, summary=summary, payload=payload,
                               status="pending_review", requested_by="agent",
                               created_at=_now(offset_minutes=15)))
        db.add(AuditLog(id=f"aud-{uuid4().hex[:8]}", action="approval.requested",
                        entity_type="approval", entity_id=aid, actor_role="agent",
                        detail={"kind": kind, "target_id": target, "title": title},
                        created_at=_now(offset_minutes=15)))


def _match_score(db: Session, sid: str, job_id: str, fallback: int) -> int:
    row = db.exec(select(JobMatch).where(JobMatch.student_id == sid,
                                          JobMatch.job_id == job_id)).first()
    return int(row.score) if row is not None else fallback


def _seed_applications(db: Session, sids: dict[str, str]) -> None:
    """M25: three flagship JobApplication rows spanning the M19 state machine.

    - awaiting_tpo_approval : Asha Patel + job-001 (dream), bound to the
      existing job_application_nomination approval (never creates a new one -
      the exactly-4-pending-approvals invariant must hold).
    - nominated             : Meera Iyer + job-007 (standard) - the M19
      auto-nominate branch, so no approval is involved.
    - awaiting_mentor_endorsement: Tanvi Reddy + job-001 with a CGPA waiver
      (eligible_at_apply False), awaiting the faculty mentor's sign-off.
    """
    asha_sid = sids.get("Asha Patel")
    meera_sid = sids.get("Meera Iyer")
    tanvi_sid = sids.get("Tanvi Reddy")
    if not (asha_sid and meera_sid and tanvi_sid):
        return
    nomination = db.exec(select(ApprovalRequest).where(
        ApprovalRequest.kind == "job_application_nomination",
        ApprovalRequest.requested_by == "agent")).first()
    passed_checks = [
        {"name": "gpa_cutoff", "sop_id": "SOP-03", "passed": True,
         "detail": "GPA meets the drive cutoff"},
        {"name": "branch_restriction", "sop_id": "SOP-09", "passed": True,
         "detail": "Branch is in the allowed set"},
        {"name": "tier_mock_gate", "sop_id": "SOP-04", "passed": True,
         "detail": "Mock-interview gate satisfied"},
        {"name": "skill_readiness", "sop_id": "SOP-05", "passed": True,
         "detail": "Required-skill coverage at or above threshold"},
    ]
    rows = [
        (JobApplication(
            id=f"app-{uuid4().hex[:8]}", student_id=asha_sid, job_id="job-001",
            status="awaiting_tpo_approval", tier="dream",
            match_score=_match_score(db, asha_sid, "job-001", 75),
            eligible_at_apply=True,
            eligibility_checks={"checks": list(passed_checks),
                                "cited_sops": ["SOP-03", "SOP-04", "SOP-05", "SOP-09"]},
            approval_id=nomination.id if nomination is not None else None,
            created_at=_now(offset_minutes=10), updated_at=_now(offset_minutes=10)),
         "application.submitted"),
        (JobApplication(
            id=f"app-{uuid4().hex[:8]}", student_id=meera_sid, job_id="job-007",
            status="nominated", tier="standard",
            match_score=_match_score(db, meera_sid, "job-007", 60),
            eligible_at_apply=True,
            eligibility_checks={"checks": list(passed_checks),
                                "cited_sops": ["SOP-03", "SOP-04", "SOP-05", "SOP-09"]},
            created_at=_now(offset_minutes=9), updated_at=_now(offset_minutes=9)),
         "application.nominated"),
        (JobApplication(
            id=f"app-{uuid4().hex[:8]}", student_id=tanvi_sid, job_id="job-001",
            status="awaiting_mentor_endorsement", tier="dream",
            match_score=_match_score(db, tanvi_sid, "job-001", 45),
            eligible_at_apply=False,
            eligibility_checks={"checks": [
                {"name": "gpa_cutoff", "sop_id": "SOP-03", "passed": False,
                 "detail": "GPA 6.90 is below the 7.50 cutoff"},
                {"name": "branch_restriction", "sop_id": "SOP-09", "passed": True,
                 "detail": "Branch is in the allowed set"},
                {"name": "tier_mock_gate", "sop_id": "SOP-04", "passed": True,
                 "detail": "Mock-interview gate satisfied"},
                {"name": "skill_readiness", "sop_id": "SOP-05", "passed": False,
                 "detail": "Required-skill coverage below threshold"}],
                "cited_sops": ["SOP-03", "SOP-04", "SOP-05", "SOP-09"]},
            notes="waiver requested for CGPA cutoff [SOP-03]",
            created_at=_now(offset_minutes=8), updated_at=_now(offset_minutes=8)),
         "application.waiver_requested"),
    ]
    for app, action in rows:
        db.add(app)
        db.add(AuditLog(id=f"aud-{uuid4().hex[:8]}", action=action,
                        entity_type="application", entity_id=app.id,
                        actor_role="agent",
                        detail={"student_id": app.student_id, "job_id": app.job_id,
                                "status": app.status},
                        created_at=_now(offset_minutes=8)))


def _ensure_demo_applications(db: Session, sids: dict[str, str]) -> None:
    """Idempotently seed the flagship application rows (once per database)."""
    if db.exec(select(JobApplication)).first() is None:
        _seed_applications(db, sids)
        db.commit()


def seed_demo_cohort(db, *, taxonomy, embedder, vector_store=None, llm=None) -> int:
    """Create a varied 12-student cohort (CSE / AI&DS / ECE / IT, all bands) with
    matches, a top-match plan, multi-point progress, finished interview
    transcripts, agent traces and 4 pending approvals so every screen is alive.

    Returns the count created. `rank_jobs` embeds job descriptions on the fly
    when `vector_store` is None.
    """
    llm = llm or MockLLM()
    jobs = db.exec(select(Job)).all()
    existing_map = {s.name: s.id for s in db.exec(select(Student)).all()}
    existing = set(existing_map)
    # idempotency: a repeated `seed --demo-cohort` must not duplicate the cohort
    if all(name in existing for name, *_ in _DEMO_COHORT):
        _ensure_demo_applications(db, existing_map)
        return len(_DEMO_COHORT)
    created = 0
    sids: dict[str, str] = dict(existing_map)
    for name, branch, year, gpa, text in _DEMO_COHORT:
        if name in existing:
            continue
        sid = f"stu-{uuid4().hex[:8]}"
        sids[name] = sid
        db.add(Student(id=sid, name=name, branch=branch, year=year, gpa=gpa, resume_text=text))
        found = extract_skills(text, taxonomy)
        for skill_id, lines in found.items():
            db.add(StudentSkill(student_id=sid, skill_id=skill_id, mentions=len(lines),
                                level=level_for(len(lines), lines), evidence=list(lines)))
        db.commit()
        student_vec = embedder.embed(text) if embedder else None
        if vector_store is not None and student_vec is not None:
            vector_store.upsert("student", sid, student_vec)
        ss_rows = db.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
        student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
        levels = {r.skill_id: r.level for r in ss_rows}
        if student_vec is None:
            # offline fallback: rank on skills alone via a zero vector
            student_vec = [0.0] * getattr(embedder, "dim", 384)
        ranked = rank_jobs(jobs, student_vec, student_skills, embedder, taxonomy,
                           vector_store=vector_store, top=10, levels=levels)
        for job, js in ranked:
            db.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=job.id,
                            score=js.score, band=js.band, breakdown=asdict(js)))
        db.commit()
        if ranked:
            try:
                build_plan(db, sid, ranked[0][0], ranked[0][1], llm=llm)
            except Exception:
                db.rollback()
        _seed_progress(db, sid)
        created += 1
        # richer demo artifacts for the first four students (keeps seeding fast)
        if created <= 4 and ranked:
            try:
                _seed_interview(db, sid, ranked[0][0].id, llm)
            except Exception:
                db.rollback()  # a failed sample interview must not abort seeding
        if created == 1:
            _seed_agent_run(db, sid,
                            "Rank jobs and build an upskilling plan for the top match",
                            ["inspect_student_profile", "rank_and_explain_jobs",
                             "simulate_skill_acquisition", "search_course_catalog"])
        elif created == 2:
            _seed_agent_run(db, sid, "Audit resume bullets against the top-matched job",
                            ["inspect_student_profile", "audit_resume_bullets"])
        elif created == 3:
            _seed_agent_run(db, sid, "Diagnose cohort bottlenecks and draft an intervention",
                            ["query_cohort_bottlenecks", "draft_cohort_intervention"])
        db.commit()
    # seed the HITL queue once, after every cohort student exists so each
    # approval can bind to the student it names in its title. Guarded solely by
    # the absence of any agent-requested approval so a partial re-run (resume
    # after a crash) never duplicates the queue or its audit rows.
    have_agent_approval = db.exec(
        select(ApprovalRequest).where(ApprovalRequest.requested_by == "agent")).first()
    if have_agent_approval is None:
        _seed_approvals(db, sids)
        db.commit()
    # M25 flagship applications: after the approvals exist so the dream-tier
    # application can bind to the nomination approval (never creates a new one).
    _ensure_demo_applications(db, sids)
    return created
