from types import SimpleNamespace

from app.matching.engine import job_eligibility
from app.taxonomy.branches import is_branch_eligible, normalize_branch


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
