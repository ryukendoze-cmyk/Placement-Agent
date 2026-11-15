"""Regression tests ensuring student names appear in case-insensitive alphabetical order
across Analytics, Placement Copilot, sidebar switchers, admin pages, login personas,
and analytics/simulator endpoints.
"""
import re
from datetime import datetime, timedelta, timezone

from sqlmodel import Session

from app.models import Student


def _extract_select_options(html: str, select_id: str) -> list[str]:
    """Return the visible option labels (excluding the leading placeholder option) for a <select>."""
    m = re.search(
        rf'<select[^>]*id="{re.escape(select_id)}"[^>]*>(.*?)</select>',
        html,
        re.DOTALL,
    )
    assert m is not None, f"Missing <select id={select_id!r}> in rendered HTML"
    labels = re.findall(r"<option[^>]*>(.*?)</option>", m.group(1), re.DOTALL)
    cleaned = [re.sub(r"\s+", " ", lbl).strip() for lbl in labels]
    # Drop the leading placeholder ("All students", "All CSE mentees", "No student context")
    return cleaned[1:] if len(cleaned) > 1 else []


def _seed_unsorted_students(eng) -> None:
    """Insert students whose names are out of alphabetical order and whose created_at timestamps
    are intentionally reversed/scrambled relative to alphabetical order.
    """
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [
        ("stu-ord-z", "Zoya Khan", "CSE", 8.4, base + timedelta(hours=1)),
        ("stu-ord-a", "Aarav Sharma", "CSE", 8.8, base + timedelta(hours=4)),
        ("stu-ord-m", "meera verma", "CSE", 8.1, base + timedelta(hours=3)),
        ("stu-ord-b", "Bhavna Joshi", "ECE", 8.6, base + timedelta(hours=2)),
    ]
    with Session(eng) as db:
        for sid, name, branch, gpa, created_at in samples:
            db.add(
                Student(
                    id=sid,
                    name=name,
                    branch=branch,
                    year=3,
                    gpa=gpa,
                    created_at=created_at,
                )
            )
        db.commit()


def test_sidebar_and_copilot_dropdowns_are_alphabetical(client_engine):
    client, eng = client_engine
    _seed_unsorted_students(eng)
    headers = {"X-Demo-Role": "tpo_admin"}

    # Home, Analytics, Policies, Approvals, Audit, and Copilot sidebar switcher
    for path in ("/", "/copilot", "/admin", "/admin/policies", "/admin/approvals", "/admin/audit"):
        resp = client.get(path, headers=headers)
        assert resp.status_code == 200, f"{path} failed: {resp.status_code}"
        sidebar_names = _extract_select_options(resp.text, "switch-student")
        assert len(sidebar_names) >= 4
        assert [n.lower() for n in sidebar_names] == sorted(n.lower() for n in sidebar_names), (
            f"Sidebar student switcher on {path} is not alphabetical: {sidebar_names}"
        )

    # Copilot student context dropdown ("No student context" selector)
    copilot_html = client.get("/copilot", headers=headers).text
    copilot_labels = _extract_select_options(copilot_html, "copilot-student")
    copilot_names = [lbl.rsplit(" (", 1)[0] for lbl in copilot_labels]
    assert len(copilot_names) >= 4
    assert [n.lower() for n in copilot_names] == sorted(n.lower() for n in copilot_names), (
        f"Copilot student context dropdown is not alphabetical: {copilot_names}"
    )


def test_faculty_mentor_copilot_and_admin_are_alphabetical(client_engine):
    client, eng = client_engine
    _seed_unsorted_students(eng)
    headers = {"X-Demo-Role": "faculty_mentor", "X-Demo-Department": "CSE"}

    for path in ("/copilot", "/admin", "/admin/policies", "/admin/approvals"):
        resp = client.get(path, headers=headers)
        assert resp.status_code == 200
        sidebar_names = _extract_select_options(resp.text, "switch-student")
        assert sidebar_names
        assert [n.lower() for n in sidebar_names] == sorted(n.lower() for n in sidebar_names), (
            f"Faculty sidebar on {path} is not alphabetical: {sidebar_names}"
        )

    copilot_html = client.get("/copilot", headers=headers).text
    copilot_labels = _extract_select_options(copilot_html, "copilot-student")
    copilot_names = [lbl.rsplit(" (", 1)[0] for lbl in copilot_labels]
    assert copilot_names
    assert [n.lower() for n in copilot_names] == sorted(n.lower() for n in copilot_names)


def test_analytics_matrix_drive_simulator_login_and_api_are_alphabetical(client_engine):
    client, eng = client_engine
    _seed_unsorted_students(eng)
    headers = {"X-Demo-Role": "tpo_admin"}

    # Analytics JSON & Student Readiness Matrix
    analytics = client.get("/api/admin/analytics", headers=headers).json()
    matrix_names = [row["name"] for row in analytics["student_matrix"]]
    assert len(matrix_names) >= 4
    assert [n.lower() for n in matrix_names] == sorted(n.lower() for n in matrix_names)

    # Drive simulator eligible_students (standard tier so all GPA-eligible students qualify)
    sim = client.post(
        "/api/admin/simulate-drive",
        headers=headers,
        json={"title": "Test Drive", "salary_lpa": [4, 6], "min_gpa": 6.0, "required_skills": []},
    )
    assert sim.status_code == 200
    sim_names = [s["name"] for s in sim.json()["eligible_students"]]
    assert len(sim_names) >= 4
    assert [n.lower() for n in sim_names] == sorted(n.lower() for n in sim_names)
    assert [n for n in sim_names if n in {"Zoya Khan", "Aarav Sharma", "meera verma", "Bhavna Joshi"}] == [
        "Aarav Sharma",
        "Bhavna Joshi",
        "meera verma",
        "Zoya Khan",
    ]

    # /api/students endpoint
    api_students = client.get("/api/students", headers=headers).json()
    api_names = [s["name"] for s in api_students]
    assert len(api_names) >= 4
    assert [n.lower() for n in api_names] == sorted(n.lower() for n in api_names)

    # /login student personas list
    login_html = client.get("/login").text
    persona_names = re.findall(
        r'<div class="student-persona-card.*?<p class="text-sm font-bold[^"]*">(.*?)</p>',
        login_html,
        re.DOTALL,
    )
    assert len(persona_names) >= 4
    assert [n.strip().lower() for n in persona_names] == sorted(
        n.strip().lower() for n in persona_names
    )
