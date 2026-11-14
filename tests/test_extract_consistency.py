"""Regression tests for JKLU CareerOS V3 Part 1:
- Section-aware proficiency calibration & punctuation alias extraction
- Negation/aspirational guards & profile branch/summary canonicalization
- Deterministic temperature=0.0 + _reconcile_level stability
- Expanded JKLU taxonomy (136 skills), free HTTPS courses, and vetted Q-bank
- Student profile.html resume-lock / pending / unlocked / TPO UI states
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader

from app.interview.quality import audit_question_bank
from app.models import Course
from app.plan.verification import _answers_for_item, check_answers, generate_quiz
from app.skills.extract import _reconcile_level, extract_resume_with_ai
from app.taxonomy.taxonomy import load_taxonomy


class _DeterministicKwargLLM:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.7) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode, "temperature": temperature})
        return json.dumps(self.payload)


class _StrictSignatureLLM:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        self.calls += 1
        return json.dumps(self.payload)


def test_section_aware_calibration_and_punctuation_aliases():
    tax = load_taxonomy()
    resume = (
        "Ananya Sharma, B.Tech ECE (VLSI), Year 4, CGPA 8.9\n"
        "Summary:\n"
        "Final-year JKLU engineering student focused on backend APIs and embedded RTL.\n"
        "Skills: C++, C#, .NET, Node.js, Next.js, CI/CD, Verilog, Docker\n"
        "Coursework: Operating Systems\n"
        "Projects:\n"
        "- Built a REST API portal using FastAPI and PostgreSQL for campus event tracking.\n"
        "Experience:\n"
        "Software Intern (May 2025 - Aug 2025, 4 months): Engineered and scaled Python ETL pipelines, "
        "reducing batch latency by 45% across 10k daily users.\n"
    )
    out = extract_resume_with_ai(resume, tax, llm=None)
    assert out["ai_parsed"] is False
    assert out["profile"]["branch"] == "ECE"
    assert "backend APIs" in out["profile"]["summary"]

    skills = out["skills"]
    # Punctuation-heavy & separator skills match accurately
    for expected in ("cplusplus", "csharp", "nodejs", "nextjs", "ci_cd", "verilog", "rest_api", "operating_systems"):
        assert expected in skills, f"Missing expected skill {expected}"

    # Section calibration: skills-only / coursework-only -> basic; project -> intermediate; quantified experience -> advanced
    assert skills["cplusplus"]["level"] == "basic"
    assert skills["docker"]["level"] == "basic"
    assert skills["operating_systems"]["level"] == "basic"
    assert skills["fastapi"]["level"] == "intermediate"
    assert skills["python"]["level"] == "advanced"


def test_negation_and_aspirational_phrases_excluded():
    tax = load_taxonomy()
    resume = (
        "Rohan Verma, B.Tech CSE (AI), Year 3, CGPA 8.5\n"
        "Skills: Python, FastAPI\n"
        "Note: Willing to learn Rust; familiarity hoped with Kubernetes; not experienced with Go; "
        "no hands-on exposure to Terraform; coursework excluded Scala; interested in Ruby.\n"
    )
    out = extract_resume_with_ai(resume, tax, llm=None)
    assert out["profile"]["branch"] == "CSE"
    skills = out["skills"]
    assert "python" in skills and "fastapi" in skills
    for blocked in ("rust", "kubernetes", "go", "terraform", "scala", "ruby"):
        assert blocked not in skills, f"Negated/aspirational skill {blocked} should be excluded"


def test_temperature_zero_passing_fallback_and_level_reconciliation():
    tax = load_taxonomy()
    resume = (
        "Priya Nair, B.Tech AI&DS, Year 4, CGPA 9.1\n"
        "Skills: Python, SQL, RAG\n"
        "Experience:\n"
        "AI Intern (6 months): Architected a RAG pipeline over 25k documents, improving answer accuracy by 38%.\n"
    )
    # Jittered LLM outputs: LLM A claims basic for RAG (despite quantified experience) and advanced for SQL (skills list only)
    llm_a = _DeterministicKwargLLM(
        {
            "profile": {"name": "Priya Nair", "branch": "B.Tech AI&DS", "year": 4, "gpa": 9.1, "summary": ""},
            "skills": [
                {"skill_id": "sql", "level": "advanced", "mentions": 1, "evidence": ["Skills: Python, SQL, RAG"]},
                {
                    "skill_id": "rag",
                    "level": "basic",
                    "mentions": 2,
                    "evidence": [
                        "AI Intern (6 months): Architected a RAG pipeline over 25k documents, improving answer accuracy by 38%."
                    ],
                },
            ],
        }
    )
    out_a = extract_resume_with_ai(resume, tax, llm=llm_a)
    assert llm_a.calls[0]["temperature"] == 0.0
    assert out_a["profile"]["branch"] == "AI&DS"
    # 2-tier drift blocked both ways by evidence-anchored reconciliation
    assert out_a["skills"]["sql"]["level"] == "basic"
    assert out_a["skills"]["rag"]["level"] == "advanced"

    # Second call on same LLM instance hits per-instance weakref cache without re-invoking LLM
    out_a_repeat = extract_resume_with_ai(resume, tax, llm=llm_a)
    assert len(llm_a.calls) == 1
    assert out_a_repeat == out_a

    # Dropping and GC'ing the instance prevents memory-address reuse collisions
    import gc

    del llm_a
    gc.collect()
    llm_fresh = _DeterministicKwargLLM({"profile": {}, "skills": []})
    extract_resume_with_ai(resume, tax, llm=llm_fresh)
    assert len(llm_fresh.calls) == 1

    # Strict signature LLM without temperature kwarg falls back cleanly without error
    llm_strict = _StrictSignatureLLM({"profile": {}, "skills": []})
    out_strict = extract_resume_with_ai(resume, tax, llm=llm_strict)
    assert llm_strict.calls == 1
    assert out_strict["ai_parsed"] is True
    assert "rag" in out_strict["skills"]

    assert _reconcile_level("advanced", "basic", ["Skills: SQL"], mentions=1) == "basic"
    assert _reconcile_level("basic", "advanced", ["Architected SQL cluster cutting latency 40%"], mentions=1) == "advanced"


def test_expanded_taxonomy_courses_and_question_bank():
    tax = load_taxonomy()
    new_skills = (
        "data_structures",
        "algorithms",
        "object_oriented_programming",
        "operating_systems",
        "computer_networks",
        "system_design",
        "rag",
        "prompt_engineering",
        "vector_databases",
        "embedded_systems",
        "vlsi",
        "verilog",
    )
    assert len(tax.all()) >= 136
    for sk in new_skills:
        assert tax.get_opt(sk) is not None, sk
        assert tax.resolve(sk) == sk

    assert tax.resolve("RAG") == "rag"
    assert tax.resolve("prompt engineering") == "prompt_engineering"
    assert tax.resolve("vector databases") == "vector_databases"

    courses = json.loads(Path("data/courses.json").read_text(encoding="utf-8"))
    assert len(courses) >= 85
    covered = set()
    for c in courses:
        Course(**c)
        assert c["url"].startswith("https://"), c["id"]
        assert c.get("url_verified") is True, c["id"]
        covered.update(c["skill_ids"])
    for sk in new_skills:
        assert sk in covered, f"{sk} missing course coverage"

    audit = audit_question_bank()
    assert audit["rejected_questions"] == 0, audit["rejections"]
    for sk in new_skills:
        qs = generate_quiz(sk)
        assert len(qs) >= 2 and all(q["skill_id"] == sk for q in qs), sk
        ans = _answers_for_item("i1", sk)
        ok, score, _ = check_answers(sk, ans)
        assert ok and score == 100


def _render_profile_html(**kwargs: object) -> str:
    env = Environment(loader=FileSystemLoader("app/templates"))
    tmpl = env.get_template("student/profile.html")
    student = SimpleNamespace(
        id="stu-1",
        name="Ananya Sharma",
        branch="CSE",
        year=4,
        gpa=8.8,
        summary="Final-year CSE candidate.",
        resume_text="Resume body",
        resume_reupload_unlocked=False,
        pii_redacted_count=0,
    )
    defaults = {
        "request": None,
        "sid": "stu-1",
        "student": student,
        "skills": [],
        "taxonomy": {},
        "role": "student",
        "has_resume": True,
        "reupload_unlocked": False,
        "pending_reupload_req": None,
    }
    defaults.update(kwargs)
    return tmpl.render(**defaults)


def test_profile_html_resume_lock_states():
    # 1. Student with locked resume and no pending request -> locked card + TPO request form, NO upload form
    html_locked = _render_profile_html(role="student", has_resume=True, reupload_unlocked=False, pending_reupload_req=None)
    assert "Resume Locked After Initial Verification" in html_locked
    assert "Request TPO Permission" in html_locked
    assert "/api/students/stu-1/resume-reupload-request" in html_locked
    assert 'hx-post="/students/stu-1/resume"' not in html_locked

    # 2. Student with pending TPO approval -> shows Request ID badge, hides both request form and upload form
    html_pending = _render_profile_html(
        role="student",
        has_resume=True,
        reupload_unlocked=False,
        pending_reupload_req=SimpleNamespace(id=77),
    )
    assert "Pending TPO Approval" in html_pending
    assert "Request #77" in html_pending
    assert 'id="reupload-request-form"' not in html_pending
    assert 'hx-post="/students/stu-1/resume"' not in html_pending

    # 3. Student unlocked by TPO -> shows unlocked banner + upload form
    html_unlocked = _render_profile_html(role="student", has_resume=True, reupload_unlocked=True)
    assert "Re-upload Unlocked by TPO" in html_unlocked
    assert 'hx-post="/students/stu-1/resume"' in html_unlocked

    # 4. TPO Admin -> always has upload form
    html_tpo = _render_profile_html(role="tpo_admin", has_resume=True, reupload_unlocked=False)
    assert 'hx-post="/students/stu-1/resume"' in html_tpo
    assert "Resume Locked After Initial Verification" not in html_tpo
