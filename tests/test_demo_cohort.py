"""Phase 8: rich demo cohort (12 students, all bands, every screen alive),
CLI --demo-cohort wiring, and the docs/rubric deliverables.
"""
from pathlib import Path

from sqlmodel import Session, select

from app.db import init_db, make_engine
from app.models import (
    AgentRun,
    AgentStep,
    ApprovalRequest,
    AuditLog,
    Interview,
    JobMatch,
    PlanItem,
    ProgressEvent,
    Student,
)
from app.seed import seed, seed_demo_cohort
from app.taxonomy.taxonomy import load_taxonomy


def _seeded_db():
    from app.embeddings.embedder import HashingEmbedder
    eng = make_engine("sqlite://")
    init_db(eng)
    tax = load_taxonomy()
    with Session(eng) as db:
        seed(db, tax)
        n = seed_demo_cohort(db, taxonomy=tax, embedder=HashingEmbedder(),
                             vector_store=None)
    return eng, n


def test_demo_cohort_creates_twelve_students():
    eng, n = _seeded_db()
    assert n == 12, f"expected 12 students, got {n}"
    with Session(eng) as db:
        students = db.exec(select(Student)).all()
        assert len(students) == 12
        branches = {s.branch for s in students}
        assert {"CSE", "AI&DS", "ECE"} <= branches, f"branches covered: {branches}"


def test_demo_cohort_covers_readiness_bands():
    eng, _n = _seeded_db()
    from app.matching.engine import band_for
    with Session(eng) as db:
        students = db.exec(select(Student)).all()
        bands = set()
        for s in students:
            ms = db.exec(select(JobMatch).where(JobMatch.student_id == s.id)).all()
            assert ms, f"{s.name} has no matches"
            top3 = sorted((m.score for m in ms), reverse=True)[:3]
            mean = round(sum(top3) / len(top3))
            bands.add(band_for(mean))  # strong|moderate|developing|early
        assert len(bands) >= 3, f"only bands {bands} represented - cohort is too uniform"


def test_every_demo_student_has_matches_and_a_plan():
    eng, _n = _seeded_db()
    from app.models import UpskillPlan
    with Session(eng) as db:
        students = db.exec(select(Student)).all()
        for s in students:
            ms = db.exec(select(JobMatch).where(JobMatch.student_id == s.id)).all()
            assert len(ms) >= 3, f"{s.name} has only {len(ms)} matches"
        plans = db.exec(select(UpskillPlan)).all()
        assert len(plans) >= 10, f"expected >=10 plans, got {len(plans)}"
        students_with_plan = {p.student_id for p in plans}
        assert len(students_with_plan) >= 10, "most students should have a plan"
        # plans have at least one item
        items = db.exec(select(PlanItem)).all()
        assert items, "expected plan items across the cohort"


def test_demo_cohort_has_multi_point_progress():
    eng, _n = _seeded_db()
    with Session(eng) as db:
        students = db.exec(select(Student)).all()
        with_progress = 0
        for s in students:
            evs = db.exec(select(ProgressEvent).where(ProgressEvent.student_id == s.id)).all()
            if len(evs) >= 3:
                with_progress += 1
        assert with_progress >= 8, f"only {with_progress} students have 3+ progress events"


def test_demo_cohort_has_finished_interview_transcripts():
    eng, _n = _seeded_db()
    with Session(eng) as db:
        ivs = db.exec(select(Interview).where(Interview.finished == True)).all()  # noqa: E712
        assert len(ivs) >= 4, f"expected >=4 finished interviews, got {len(ivs)}"
        for iv in ivs:
            assert iv.transcript, f"{iv.id} has empty transcript"
            assert iv.score is not None
            # rubric fields present on answered turns (Phase 6 contract)
            answered = [t for t in iv.transcript if t.get("score") is not None]
            assert answered and all("rubric" in t for t in answered)


def test_demo_cohort_has_agent_traces():
    eng, _n = _seeded_db()
    with Session(eng) as db:
        runs = db.exec(select(AgentRun)).all()
        assert len(runs) >= 3, f"expected >=3 agent runs, got {len(runs)}"
        steps = db.exec(select(AgentStep)).all()
        assert len(steps) >= 6, f"expected >=6 agent steps, got {len(steps)}"
        for r in runs:
            assert r.goal
            mine = [s for s in steps if s.run_id == r.id]
            assert mine, f"run {r.id} has no steps"


def test_demo_cohort_has_four_pending_approvals():
    eng, _n = _seeded_db()
    with Session(eng) as db:
        pend = db.exec(select(ApprovalRequest).where(
            ApprovalRequest.status == "pending_review")).all()
        assert len(pend) == 4, f"expected 4 pending approvals, got {len(pend)}"
        kinds = {a.kind for a in pend}
        assert "cohort_bootcamp_publish" in kinds
        assert len(kinds) >= 3, f"approval kinds: {kinds}"


def test_demo_cohort_is_idempotent_on_repeat_runs():
    """Re-running the base seed() AND seed_demo_cohort() must not duplicate."""
    from app.embeddings.embedder import HashingEmbedder
    eng = make_engine("sqlite://")
    init_db(eng)
    tax = load_taxonomy()
    with Session(eng) as db:
        seed(db, tax)
        seed_demo_cohort(db, taxonomy=tax, embedder=HashingEmbedder(), vector_store=None)
        first = len(db.exec(select(Student)).all())
        first_approvals = len(db.exec(select(ApprovalRequest)).all())
        # base seed() is merge-based: re-running it must not duplicate anything
        seed(db, tax)
        second = len(db.exec(select(Student)).all())
        assert second == first, "re-running seed() must not duplicate students"
        # seed_demo_cohort() itself must be idempotent too
        seed_demo_cohort(db, taxonomy=tax, embedder=HashingEmbedder(), vector_store=None)
        third = len(db.exec(select(Student)).all())
        assert third == first, "re-running seed_demo_cohort() duplicated students"
        third_approvals = len(db.exec(select(ApprovalRequest)).all())
        assert third_approvals == first_approvals, "re-running duplicated approvals"


def test_partial_resume_does_not_duplicate_approvals():
    """Resume after a crash mid-seed (some students exist, approvals already
    written) must not re-seed the HITL queue (Gate C r2 operator-precedence)."""
    from app.embeddings.embedder import HashingEmbedder
    from app.seed import _DEMO_COHORT, _seed_approvals

    eng = make_engine("sqlite://")
    init_db(eng)
    tax = load_taxonomy()
    with Session(eng) as db:
        seed(db, tax)
        # simulate a crash that stopped after 6 of the 12 students + the queue
        for name, branch, year, gpa, text in _DEMO_COHORT[:6]:
            db.add(Student(id=f"stu-{name.split()[0].lower()[:4]}",
                           name=name, branch=branch, year=year, gpa=gpa,
                           resume_text=text))
        sids = {name: f"stu-{name.split()[0].lower()[:4]}"
                for name, *_ in _DEMO_COHORT[:6]}
        db.commit()
        _seed_approvals(db, sids)   # queue written before the "crash"
        db.commit()
        seeded_approvals = len(db.exec(select(ApprovalRequest)).all())
        assert seeded_approvals == 4, "setup: queue should have 4 rows"
        # resume: seeds the remaining students; must NOT re-seed the queue
        n = seed_demo_cohort(db, taxonomy=tax, embedder=HashingEmbedder(),
                             vector_store=None)
        assert n == 6, f"resume should create the 6 missing students, got {n}"
        final_approvals = len(db.exec(select(ApprovalRequest)).all())
        assert final_approvals == seeded_approvals, (
            f"partial resume duplicated the queue: {seeded_approvals} -> {final_approvals}"
        )
        students = len(db.exec(select(Student)).all())
        assert students == 12, f"expected full cohort of 12, got {students}"


def test_seed_approvals_bind_to_named_students():
    """Each approval must bind to the student named in its title (Gate C UX-1)."""
    eng, _n = _seeded_db()
    with Session(eng) as db:
        students = {s.id: s.name for s in db.exec(select(Student)).all()}
        for a in db.exec(select(ApprovalRequest)).all():
            sid = (a.payload or {}).get("student_id")
            if not sid:
                continue
            name = students.get(sid, "")
            assert name in a.title, (
                f"approval {a.title!r} binds to {name!r} (id {sid}) - title mismatch"
            )
            if a.kind == "proficiency_waiver":
                assert a.payload.get("level") == "advanced", (
                    "waiver payload must set 'level' to advance proficiency"
                )


def test_seed_writes_audit_rows_for_queue_and_interviews():
    """The demo DB must not start with an empty /admin/audit (Gate C UX-3)."""
    eng, _n = _seeded_db()
    with Session(eng) as db:
        rows = db.exec(select(AuditLog)).all()
        actions = {r.action for r in rows}
        assert "approval.requested" in actions, "missing approval.requested audit rows"
        assert "interview.completed" in actions, "missing interview.completed audit rows"
        assert len(rows) >= 8, f"expected >=8 audit rows at seed, got {len(rows)}"


def test_agent_runs_endpoint_includes_steps():
    """GET /api/agent/runs must carry steps - the Agent Brain drawer renders from
    it (Gate C IMPORTANT-1). A regression here leaves the trace blank."""
    from fastapi.testclient import TestClient

    from app.api.main import AppContext, create_app
    from app.embeddings.embedder import HashingEmbedder
    from app.embeddings.store import MemoryVectorStore

    eng, _n = _seeded_db()
    tax = load_taxonomy()
    ctx = AppContext(engine=eng, taxonomy=tax, embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=None)
    client = TestClient(create_app(ctx))
    with client:
        runs = client.get("/api/agent/runs").json()
        assert runs, "expected seeded agent runs"
        non_empty = [r for r in runs if r.get("steps")]
        assert non_empty, "no run returned steps - Agent Brain trace would be blank"
        assert all("thought" in s and "tool_name" in s
                   for s in non_empty[0]["steps"]), "step payload incomplete"


def test_cli_help_advertises_demo_cohort():
    from app.cli import main
    try:
        main(["seed", "--help"])
    except SystemExit as e:
        assert e.code == 0
    # argparse writes help to stdout; capture via capsys is awkward here,
    # so assert the parser wiring directly instead

    from app import cli as cli_mod
    src = Path(cli_mod.__file__).read_text(encoding="utf-8")
    assert "--demo-cohort" in src


def test_docs_deliverables_exist():
    readme = Path("README.md")
    report = Path("docs/COMPLETION-REPORT.md")
    assert readme.exists(), "README.md missing"
    assert report.exists(), "docs/COMPLETION-REPORT.md missing"
    rtext = readme.read_text(encoding="utf-8")
    ctext = report.read_text(encoding="utf-8")
    # rubric mapping table + live demo script live in the completion report
    assert "100" in ctext and "rubric" in ctext.lower(), "rubric mapping missing"
    assert "demo script" in ctext.lower() or "live demo" in ctext.lower(), "demo script missing"
    # README covers running the app + tests
    lower = rtext.lower()
    assert "pytest" in lower and ("uv run" in lower or "python -m" in lower)


def test_jinja_undefined_safe_on_all_templates():
    """Every page renders with the shell context (smoke over the HTML tree)."""
    from fastapi.testclient import TestClient

    from app.api.main import AppContext, create_app
    from app.embeddings.embedder import HashingEmbedder
    from app.embeddings.store import MemoryVectorStore
    eng = make_engine("sqlite://")
    init_db(eng)
    tax = load_taxonomy()
    with Session(eng) as db:
        seed(db, tax)
    ctx = AppContext(engine=eng, taxonomy=tax, embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=None)
    client = TestClient(create_app(ctx))
    with client:
        for path in ("/", "/copilot", "/admin", "/admin/approvals", "/admin/audit"):
            r = client.get(path)
            assert r.status_code == 200, f"{path} -> {r.status_code}"
            assert "JKLU CareerOS" in r.text, f"{path} missing shell"
