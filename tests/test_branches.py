from types import SimpleNamespace

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlmodel import Session, select

from app.db import init_db, make_engine
from app.matching.engine import job_eligibility
from app.models import Student
from app.taxonomy.branches import (
    CANONICAL_BRANCHES,
    JKLU_BRANCHES,
    branch_choices,
    canonical_branches,
    is_branch_eligible,
    normalize_branch,
)


def test_normalize_branch_cse_variants():
    variants = [
        "CSE",
        "cse",
        "CS",
        "Computer Science",
        "Computer Science Engineering",
        "Computer Science and Engineering",
        "Computer Science & Engineering",
        "B.Tech CSE",
        "BTECH CSE",
        "B.Tech in Computer Science",
        "B.Tech in Computer Science and Engineering",
        "Computer Engineering",
    ]
    for v in variants:
        assert normalize_branch(v) == "CSE", f"Failed for variant: {v}"


def test_normalize_branch_other_disciplines():
    assert normalize_branch("Information Technology") == "IT"
    assert normalize_branch("info tech") == "IT"
    assert normalize_branch("Electronics and Communication") == "ECE"
    assert normalize_branch("Electronics & Communication Engineering") == "ECE"
    assert normalize_branch("Mechanical Engineering") == "ME"
    assert normalize_branch("Civil Engineering") == "CE"
    assert normalize_branch("Electrical Engineering") == "EE"
    assert normalize_branch("Artificial Intelligence and Data Science") == "AI&DS"
    assert normalize_branch("AI/DS") == "AI&DS"
    assert normalize_branch("AI&DS") == "AI&DS"


def test_is_branch_eligible_canonical_matching():
    # Student has full form "Computer Science Engineering", job requires "CSE"
    assert is_branch_eligible("Computer Science Engineering", ["CSE", "IT"]) is True
    # Student has "B.Tech in Computer Science & Engineering", job requires "CSE"
    assert is_branch_eligible("B.Tech in Computer Science & Engineering", ["CSE"]) is True
    # Student has "CSE", job requires "Computer Science"
    assert is_branch_eligible("CSE", ["Computer Science", "Information Technology"]) is True
    # Ineligible branch
    assert is_branch_eligible("Mechanical", ["CSE", "IT"]) is False
    assert is_branch_eligible("Civil Engineering", ["CSE", "IT", "ECE"]) is False
    # Empty allowed branches means open to all
    assert is_branch_eligible("Civil Engineering", []) is True
    assert is_branch_eligible("Mechanical", None) is True


def test_job_eligibility_with_full_branch_name():
    student = SimpleNamespace(gpa=8.5, branch="Computer Science Engineering")
    job = SimpleNamespace(min_gpa=7.5, allowed_branches=["CSE", "IT"])

    ok, reasons = job_eligibility(student, job)
    assert ok is True, f"Expected student to be eligible, got reasons: {reasons}"
    assert len(reasons) == 0


def test_job_eligibility_ineligible_branch():
    student = SimpleNamespace(gpa=8.5, branch="Mechanical Engineering")
    job = SimpleNamespace(min_gpa=7.5, allowed_branches=["CSE", "IT"])

    ok, reasons = job_eligibility(student, job)
    assert ok is False
    assert any("not in the allowed branches" in r for r in reasons)


def test_canonical_branches_and_jklu_specializations():
    expected = ["CSE", "AI&DS", "ECE", "IT", "EE", "ME", "CE", "CHE", "BT"]
    assert canonical_branches() == expected
    assert [code for code, _ in CANONICAL_BRANCHES] == expected
    assert JKLU_BRANCHES == CANONICAL_BRANCHES

    choices = branch_choices()
    assert [code for code, _label in choices] == expected
    assert all(" - " in label for _code, label in choices)

    assert normalize_branch("CSE (AI)") == "CSE"
    assert normalize_branch("CSE (Cybersecurity)") == "CSE"
    assert normalize_branch("CSE (Design)") == "CSE"
    assert normalize_branch("M.Tech CSE") == "CSE"
    assert normalize_branch("B.Tech AI&DS") == "AI&DS"
    assert normalize_branch("ECE (VLSI)") == "ECE"
    assert normalize_branch("Chemical Engineering") == "CHE"
    assert normalize_branch("Biotechnology") == "BT"
    assert is_branch_eligible("M.Tech Computer Science", ["CSE (AI)"]) is True


def test_init_db_adds_reupload_lock_and_normalizes_legacy_branches(tmp_path):
    db_file = tmp_path / "legacy_v3.db"
    eng = make_engine(f"sqlite:///{db_file}")

    # Simulate legacy student table without resume_reupload_unlocked and with verbose branch names
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE student ("
                "id TEXT PRIMARY KEY, "
                "name TEXT NOT NULL, "
                "branch TEXT NOT NULL, "
                "year INTEGER NOT NULL, "
                "gpa REAL NOT NULL, "
                "backlogs INTEGER NOT NULL DEFAULT 0, "
                "pii_redacted_count INTEGER NOT NULL DEFAULT 0, "
                "resume_text TEXT NOT NULL, "
                "created_at TIMESTAMP)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO student (id, name, branch, year, gpa, backlogs, pii_redacted_count, resume_text) "
                "VALUES ('stu-1', 'Aarav', 'B.Tech in Computer Science & Engineering', 3, 8.6, 0, 0, 'resume'), "
                "('stu-2', 'Meera', 'Electronics & Communication Engineering', 4, 8.2, 0, 0, '')"
            )
        )

    init_db(eng)

    cols = {c["name"] for c in sa_inspect(eng).get_columns("student")}
    assert "resume_reupload_unlocked" in cols

    with Session(eng) as session:
        s1 = session.exec(select(Student).where(Student.id == "stu-1")).one()
        s2 = session.exec(select(Student).where(Student.id == "stu-2")).one()
        assert s1.branch == "CSE"
        assert s1.resume_reupload_unlocked is False
        assert s2.branch == "ECE"
        assert s2.resume_reupload_unlocked is False
