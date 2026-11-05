"""M25: flagship demo cohort v3 (applications + speech telemetry) and docs sync."""
from pathlib import Path

from sqlmodel import Session, select

from app.db import init_db, make_engine
from app.models import ApprovalRequest, Interview, JobApplication, Student
from app.seed import seed, seed_demo_cohort
from app.taxonomy.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]


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


# ---------------------------------------------------------------------------
# 25.1 demo cohort v3: JobApplication states
# ---------------------------------------------------------------------------

def test_demo_cohort_seeds_three_application_states():
    eng, _ = _seeded_db()
    with Session(eng) as db:
        apps = db.exec(select(JobApplication)).all()
    assert len(apps) == 3
    statuses = sorted(a.status for a in apps)
    assert statuses == ["awaiting_mentor_endorsement", "awaiting_tpo_approval", "nominated"]


def test_tpo_approval_application_binds_existing_approval():
    eng, _ = _seeded_db()
    with Session(eng) as db:
        apps = db.exec(select(JobApplication)).all()
        approvals = db.exec(select(ApprovalRequest)).all()
        tpo = next(a for a in apps if a.status == "awaiting_tpo_approval")
        assert tpo.approval_id is not None
        bound = next(p for p in approvals if p.id == tpo.approval_id)
        assert bound.kind == "job_application_nomination"
        assert bound.payload["job_id"] == tpo.job_id
        assert bound.status == "pending_review"
        # the approval count invariant from test_demo_cohort must hold
        assert len(approvals) == 4
        # no other application may bind the same approval
        others = [a for a in apps if a.approval_id]
        assert len(others) == 1


def test_mentor_endorsement_application_is_waiver_case():
    eng, _ = _seeded_db()
    with Session(eng) as db:
        tanvi = db.exec(select(Student).where(Student.name == "Tanvi Reddy")).first()
        app = db.exec(select(JobApplication).where(JobApplication.student_id == tanvi.id)).first()
        assert app is not None
        assert app.status == "awaiting_mentor_endorsement"
        assert app.eligible_at_apply is False
        assert app.approval_id is None
        gpa = next(c for c in app.eligibility_checks["checks"] if c["name"] == "gpa_cutoff")
        assert gpa["passed"] is False
        assert "waiver" in app.notes.lower()


def test_nominated_application_is_auto_nominate_standard():
    eng, _ = _seeded_db()
    with Session(eng) as db:
        apps = db.exec(select(JobApplication)).all()
        nom = next(a for a in apps if a.status == "nominated")
        assert nom.approval_id is None
        assert nom.eligible_at_apply is True
        assert nom.tier == "standard"
        assert nom.match_score > 0


def test_demo_cohort_applications_idempotent():
    eng, _ = _seeded_db()
    from app.embeddings.embedder import HashingEmbedder
    tax = load_taxonomy()
    with Session(eng) as db:
        seed_demo_cohort(db, taxonomy=tax, embedder=HashingEmbedder(), vector_store=None)
        n_apps = len(db.exec(select(JobApplication)).all())
        n_appr = len(db.exec(select(ApprovalRequest)).all())
    assert n_apps == 3
    assert n_appr == 4


def test_preexisting_cohort_gains_applications_on_reseed():
    """A DB that already has all 12 cohort students (pre-M25 seed) gets the
    v3 application rows on the next run via the early-return path."""
    from app.embeddings.embedder import HashingEmbedder
    from app.seed import _DEMO_COHORT
    eng = make_engine("sqlite://")
    init_db(eng)
    tax = load_taxonomy()
    with Session(eng) as db:
        seed(db, tax)
        for name, branch, year, gpa, text in _DEMO_COHORT:
            db.add(Student(id=f"stu-{name.split()[0].lower()}", name=name,
                           branch=branch, year=year, gpa=gpa, resume_text=text))
        db.commit()
    with Session(eng) as db:
        n = seed_demo_cohort(db, taxonomy=tax, embedder=HashingEmbedder(),
                             vector_store=None)
        assert n == 12
        assert len(db.exec(select(JobApplication)).all()) == 3


# ---------------------------------------------------------------------------
# 25.1 demo cohort v3: speech delivery telemetry
# ---------------------------------------------------------------------------

def test_seeded_interviews_carry_speech_telemetry():
    eng, _ = _seeded_db()
    with Session(eng) as db:
        ivs = db.exec(select(Interview)).all()
    assert len(ivs) >= 4
    for iv in ivs:
        answered = [t for t in iv.transcript if isinstance(t, dict) and t.get("a")]
        assert answered, f"{iv.id} has no answered turns"
        for t in answered:
            sa = t.get("speech_analytics")
            assert isinstance(sa, dict), "answered turn missing speech_analytics"
            assert sa["wpm"] > 0
            assert "filler_rate_pct" in sa
            assert 0 <= sa["star_score"] <= 100
        ss = iv.feedback.get("speech_summary")
        assert isinstance(ss, dict) and ss, "feedback.speech_summary missing"
        assert ss["avg_wpm"] > 0
        assert ss["turns_measured"] == len(answered)


def test_aggregate_speech_delivery_unit():
    from app.interview.speech_analytics import aggregate_speech_delivery
    assert aggregate_speech_delivery([]) == {}
    assert aggregate_speech_delivery([{"q": "x"}]) == {}
    out = aggregate_speech_delivery([
        {"speech_analytics": {"wpm": 130, "star_score": 80, "delivery_score": 70,
                              "filler_words": ["um"]}},
        {"speech_analytics": {"wpm": 150, "star_score": 60, "delivery_score": 50,
                              "filler_words": ["uh", "like"]}},
        {"q": "no analytics"},
    ])
    assert out["avg_wpm"] == 140
    assert out["total_fillers"] == 3
    assert out["avg_star_score"] == 70
    assert out["turns_measured"] == 2


# ---------------------------------------------------------------------------
# 25.2 documentation sync
# ---------------------------------------------------------------------------

def test_readme_has_docker_quickstart_and_v3_endpoints():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docker compose up --build" in readme
    for ep in ("/api/agent/run/stream", "/api/copilot/ask/stream",
               "/api/students/{sid}/applications", "/api/admin/simulate-drive",
               "/healthz", "/readyz"):
        assert ep in readme, f"README missing endpoint {ep}"
    assert "v3.0.0" in readme


def test_architecture_has_m14_to_m25_sections():
    arch = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for m in range(14, 26):
        assert f"Milestone {m}" in arch, f"ARCHITECTURE.md missing Milestone {m} section"


def test_completion_report_is_v3_with_rubric():
    report = (ROOT / "docs" / "COMPLETION-REPORT.md").read_text(encoding="utf-8")
    low = report.lower()
    assert "v3.0.0" in report
    assert "100" in report and "rubric" in low
    assert "demo script" in low or "live demo" in low
    for m in (14, 17, 19, 21, 22, 23, 24, 25):
        assert f"M{m}" in report or f"Milestone {m}" in report
