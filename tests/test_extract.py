"""Tests for the dynamic AI resume extraction pipeline (ARCHITECTURE.md 3.1).

Covers: dynamic proficiency levels, candidate profile metadata, taxonomy
grounding / hallucination filtering, LLM-failure fallback, and backward
compatibility of extract_skills / level_for.
"""
import json

import pytest

from app.skills.extract import extract_skills, level_for
from app.taxonomy.taxonomy import Taxonomy

T = Taxonomy([
    {"id": "python", "name": "Python", "category": "languages", "aliases": ["python"]},
    {"id": "fastapi", "name": "FastAPI", "category": "web_backend", "aliases": ["fastapi"]},
    {"id": "c", "name": "C", "category": "languages", "aliases": ["c"]},
])
TEXT = "Built REST APIs with FastAPI in Python\nAdvanced Python scripting\nUsed C in college"


# ---------------------------------------------------------------------------
# Backward compatibility (pre-AI pipeline must keep working unchanged)
# ---------------------------------------------------------------------------
def test_finds_skills_with_evidence():
    out = extract_skills(TEXT, T)
    assert set(out) == {"python", "fastapi", "c"}
    assert "Built REST APIs with FastAPI in Python" in out["fastapi"]
    assert len(out["python"]) == 2


def test_word_boundaries_prevent_false_positives():
    out = extract_skills("I use pythonic style", T)
    assert "python" not in out            # "pythonic" must not match alias "python"


def test_level_heuristic():
    assert level_for(5) == "advanced"
    assert level_for(2) == "intermediate"
    assert level_for(1) == "basic"


# ---------------------------------------------------------------------------
# Fixtures & doubles
# ---------------------------------------------------------------------------
class _StubTaxonomy:
    """Minimal Taxonomy stand-in so grounding tests do not depend on data/."""

    def __init__(self, skills):
        self._skills = skills
        self._by_id = {s["id"]: s for s in skills}
        self.alias_map = {}
        for s in skills:
            self.alias_map[s["id"].lower()] = s["id"]
            for a in s.get("aliases", []):
                self.alias_map[a.lower()] = s["id"]

    def all(self):
        return [type("S", (), s)() for s in self._skills]

    def get_opt(self, skill_id):
        s = self._by_id.get(skill_id)
        return type("S", (), s)() if s else None

    def resolve(self, token):
        return self.alias_map.get(token.lower())

    def get(self, skill_id):
        return type("S", (), self._by_id[skill_id])()


STUB_TAX = _StubTaxonomy([
    {"id": "python", "name": "Python", "category": "languages", "aliases": ["python", "python3"]},
    {"id": "fastapi", "name": "FastAPI", "category": "web_backend", "aliases": ["fastapi"]},
    {"id": "react", "name": "React", "category": "web_frontend", "aliases": ["react", "reactjs"]},
    {"id": "c", "name": "C", "category": "languages", "aliases": ["c"]},
])

# Resume text with NO static level hints ("advanced"/"expert"/"proficient"/
# "intermediate") so any level assignment must come from context, not keywords.
RESUME_AI = (
    "Aarav Mehta, B.Tech CSE, Year 3, CGPA 8.7\n"
    "Summary: Backend engineer who ships production systems.\n"
    "Experience - Summer Intern, Zeta Labs:\n"
    "Architected a Python event pipeline on FastAPI serving 50k requests/day, "
    "cutting p95 latency by 40%.\n"
    "Projects:\n"
    "Built a React dashboard used by 5 teams.\n"
    "Skills: Python, FastAPI, React\n"
)


class _FakeLLM:
    """Canned LLM returning a fixed JSON payload."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def complete(self, system, user, json_mode=False):
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _payload(skills, profile=None):
    return json.dumps({
        "profile": profile or {
            "name": "Aarav Mehta", "branch": "CSE", "year": 3,
            "gpa": 8.7, "summary": "Backend engineer who ships production systems.",
        },
        "skills": skills,
    })


# ---------------------------------------------------------------------------
# 1. Dynamic proficiency levels (contextual evidence, no static keyword hints)
# ---------------------------------------------------------------------------
def test_extract_resume_with_ai_exists_and_is_callable():
    from app.skills.extract import extract_resume_with_ai
    assert callable(extract_resume_with_ai)


def test_extract_resume_with_ai_assigns_dynamic_levels():
    """LLM verdict wins: a skill the LLM calls 'basic' stays basic even when
    resume text contains words like 'advanced' elsewhere; 'advanced' stays
    advanced when no static hint word appears."""
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(_payload([
        {"skill_id": "python", "level": "advanced", "mentions": 3,
         "evidence": ["Architected a Python event pipeline on FastAPI serving 50k requests/day"]},
        {"skill_id": "fastapi", "level": "basic", "mentions": 1,
         "evidence": ["Skills: Python, FastAPI, React"]},
        {"skill_id": "react", "level": "intermediate", "mentions": 2,
         "evidence": ["Built a React dashboard used by 5 teams"]},
    ]))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    skills = out["skills"]
    assert skills["python"]["level"] == "advanced"
    assert skills["fastapi"]["level"] == "basic"
    assert skills["react"]["level"] == "intermediate"
    assert out["ai_parsed"] is True


def test_extract_resume_with_ai_calls_llm_with_extract_resume_task():
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(_payload([]))
    extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    assert llm.calls, "LLM was never invoked"
    call = llm.calls[0]
    assert call["system"].startswith("TASK: extract_resume\n")
    assert call["json_mode"] is True
    sent = json.loads(call["user"])
    assert sent["resume_text"] == RESUME_AI
    assert any(t["id"] == "python" for t in sent["taxonomy"])


def test_extract_resume_with_ai_normalizes_levels():
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(_payload([
        {"skill_id": "python", "level": "EXPERT", "mentions": 2, "evidence": ["Python everywhere"]},
        {"skill_id": "react", "level": "wizard", "mentions": 1, "evidence": ["React dashboard"]},
    ]))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    # invalid levels must be normalized to one of basic|intermediate|advanced
    assert out["skills"]["python"]["level"] in {"basic", "intermediate", "advanced"}
    assert out["skills"]["react"]["level"] in {"basic", "intermediate", "advanced"}


def test_extract_resume_with_ai_mentions_at_least_one():
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(_payload([
        {"skill_id": "python", "level": "basic", "mentions": 0, "evidence": []},
    ]))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    assert out["skills"]["python"]["mentions"] >= 1


def test_empty_text_returns_empty_result():
    from app.skills.extract import extract_resume_with_ai

    out = extract_resume_with_ai("   \n  ", STUB_TAX, None)
    assert out["ai_parsed"] is False
    assert out["skills"] == {}
    assert out["profile"]["name"] == ""
    assert out["profile"]["year"] is None
    assert out["profile"]["gpa"] is None


# ---------------------------------------------------------------------------
# 2. Candidate profile metadata extraction
# ---------------------------------------------------------------------------
def test_profile_metadata_extracted():
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(_payload([], profile={
        "name": "Aarav Mehta", "branch": "CSE", "year": 3,
        "gpa": 8.7, "summary": "Backend engineer who ships production systems.",
    }))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    p = out["profile"]
    assert p["name"] == "Aarav Mehta"
    assert p["branch"] == "CSE"
    assert p["year"] == 3
    assert p["gpa"] == pytest.approx(8.7)
    assert p["summary"] == "Backend engineer who ships production systems."


def test_profile_defaults_when_llm_omits_fields():
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(json.dumps({"profile": {}, "skills": []}))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    p = out["profile"]
    assert set(p) == {"name", "branch", "year", "gpa", "summary"}
    assert p["name"] == "" or isinstance(p["name"], str)
    assert p["year"] is None or isinstance(p["year"], int)
    assert p["gpa"] is None or isinstance(p["gpa"], (int, float))


def test_profile_backfills_remaining_fields_when_llm_returns_partial_profile():
    """Regression (Finding #1): when the LLM returns a name but omits
    branch/year/gpa, the contextual pass must still backfill those missing
    fields. The outer gate `not name and not summary` (extract.py:460) must
    not block backfill of the other keys.

    INVERTED CHECK (charter 6): this currently FAILS because the gate is too
    narrow. It stays red until extract.py:460 changes; when the fix lands, it
    goes green and becomes a permanent regression check. Do NOT delete it -
    the green state is the acceptance signal for the fix.
    """
    from app.skills.extract import extract_resume_with_ai

    text = (
        "Aarav Mehta\n"
        "Summary: Backend engineer who ships production systems.\n"
        "B.Tech CSE Year 3 CGPA 8.7\n"
        "Skills: Python\n"
    )
    llm = _FakeLLM(json.dumps({
        "profile": {"name": "Aarav", "branch": "", "year": None,
                    "gpa": None, "summary": ""},
        "skills": [],
    }))
    out = extract_resume_with_ai(text, STUB_TAX, llm)
    p = out["profile"]
    # name stays as the LLM returned it (not overwritten)
    assert p["name"] == "Aarav"
    # the remaining fields MUST be backfilled from the contextual pass
    assert p["branch"], "branch not backfilled when name was present"
    assert p["gpa"] is not None, "gpa not backfilled when name was present"
    assert p["year"] is not None, "year not backfilled when name was present"


def test_profile_llm_all_empty_gets_full_contextual_backfill():
    """Anti-vacuity: the same probe as the partial-profile test but with an
    all-empty LLM profile proves the contextual backfill path itself works -
    so a failure in the partial case is the gate condition, not a dead path."""
    from app.skills.extract import extract_resume_with_ai

    text = (
        "Aarav Mehta\n"
        "Summary: Backend engineer who ships production systems.\n"
        "B.Tech CSE Year 3 CGPA 8.7\n"
        "Skills: Python\n"
    )
    llm = _FakeLLM(json.dumps({
        "profile": {"name": "", "branch": "", "year": None,
                    "gpa": None, "summary": ""},
        "skills": [],
    }))
    out = extract_resume_with_ai(text, STUB_TAX, llm)
    p = out["profile"]
    assert p["name"] == "Aarav Mehta"
    assert p["branch"] == "CSE"
    assert p["year"] == 3
    assert p["gpa"] == pytest.approx(8.7)
    # this test proves the contextual backfill path is alive; the partial-profile
    # test above failing is therefore the gate condition, not a dead path.


def test_profile_backfills_when_llm_returns_name_and_summary():
    """Regression (Finding #2): the extract.py:482 fix changed the gate from
    `not name and not summary` to `not name or not summary`, which only fixed
    the name-set/summary-empty case. When the LLM returns BOTH a name and a
    summary non-empty, the gate still evaluates False and branch/year/gpa are
    never backfilled — but the contextual pass can produce them.

    The gate should not exist at all; the inner per-field loop (line ~486,
    `if not profile[key] and ctx_profile[key]`) already fills empty keys
    correctly. This pins the remaining narrow-gate miss.

    INVERTED CHECK (charter 6): currently FAILS. Red until extract.py:482
    removes the outer gate; green on fix = permanent regression. Do not delete.
    """
    from app.skills.extract import extract_resume_with_ai

    text = (
        "Aarav Mehta\n"
        "B.Tech CSE Year 3 CGPA 8.7\n"
        "Skills: Python\n"
    )
    llm = _FakeLLM(json.dumps({
        "profile": {"name": "Aarav", "branch": "", "year": None,
                    "gpa": None, "summary": "Summary present"},
        "skills": [],
    }))
    out = extract_resume_with_ai(text, STUB_TAX, llm)
    p = out["profile"]
    assert p["name"] == "Aarav"
    assert p["summary"] == "Summary present"   # LLM value preserved
    # the remaining fields MUST be backfilled despite name+summary both present
    assert p["branch"], "branch not backfilled when name and summary both present"
    assert p["gpa"] is not None, "gpa not backfilled when name and summary both present"
    assert p["year"] is not None, "year not backfilled when name and summary both present"


# ---------------------------------------------------------------------------
# 3. Taxonomy grounding & hallucination filtering
# ---------------------------------------------------------------------------
def test_unknown_skill_ids_rejected():
    """skill_ids not in the taxonomy must be dropped (hallucination guard)."""
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(_payload([
        {"skill_id": "quantum_kubernetes", "level": "advanced", "mentions": 3,
         "evidence": ["I used quantum kubernetes"]},
        {"skill_id": "python", "level": "intermediate", "mentions": 2,
         "evidence": ["Built with Python"]},
    ]))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    assert "quantum_kubernetes" not in out["skills"]
    assert "python" in out["skills"]


def test_ungrounded_skills_filtered():
    """A valid taxonomy skill whose evidence is NOT in the resume and whose
    alias does not appear in the resume must be filtered out."""
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(_payload([
        # 'c' is in taxonomy but NEVER appears in RESUME_AI, and the evidence
        # is fabricated text not present in the resume -> must be dropped.
        {"skill_id": "c", "level": "advanced", "mentions": 5,
         "evidence": ["Rewrote the kernel in C for fun"]},
    ]))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    assert "c" not in out["skills"]


def test_grounded_skill_kept_when_alias_in_text():
    """skill kept when its taxonomy alias appears in the resume text even if
    evidence is weak."""
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(_payload([
        {"skill_id": "python", "level": "advanced", "mentions": 4,
         "evidence": ["skills list"]},
    ]))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    assert "python" in out["skills"]


def test_evidence_deduplicated_and_nonempty():
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(_payload([
        {"skill_id": "python", "level": "intermediate", "mentions": 4,
         "evidence": ["Built with Python", "Built with Python", "  ", "Python scales"]},
    ]))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    ev = out["skills"]["python"]["evidence"]
    assert len(ev) == len(set(ev))
    assert all(e.strip() for e in ev)


# ---------------------------------------------------------------------------
# 4. Fallback when LLM raises or returns malformed JSON
# ---------------------------------------------------------------------------
def test_fallback_when_llm_raises():
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(RuntimeError("LLM exploded"))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    assert out["ai_parsed"] is False
    # contextual fallback must still surface at least one grounded skill
    assert "python" in out["skills"]
    assert out["skills"]["python"]["level"] in {"basic", "intermediate", "advanced"}


def test_fallback_when_llm_returns_malformed_json():
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM("this is {not valid json")
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    assert out["ai_parsed"] is False
    assert "python" in out["skills"]


def test_fallback_when_llm_is_none():
    from app.skills.extract import extract_resume_with_ai

    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, None)
    assert out["ai_parsed"] is False
    assert "python" in out["skills"]
    assert out["skills"]["python"]["level"] in {"basic", "intermediate", "advanced"}


def test_fallback_when_llm_returns_wrong_top_level_type():
    """JSON parses but top level is a list, not the expected object -> fallback."""
    from app.skills.extract import extract_resume_with_ai

    llm = _FakeLLM(json.dumps([1, 2, 3]))
    out = extract_resume_with_ai(RESUME_AI, STUB_TAX, llm)
    assert out["ai_parsed"] is False


# ---------------------------------------------------------------------------
# 5. extract_skills / level_for / extract_skills_with_llm compatibility
# ---------------------------------------------------------------------------
def test_extract_skills_none_llm_signature_accepts_llm_kwarg():
    """extract_skills must still accept an optional llm=None kwarg."""
    out = extract_skills(TEXT, T, None)
    assert set(out) == {"python", "fastapi", "c"}


def test_level_for_backcompat_contract():
    assert level_for(5) == "advanced"
    assert level_for(2) == "intermediate"
    assert level_for(1) == "basic"


def test_level_for_with_evidence_lines_backcompat():
    # old signature: level_for(mentions, evidence_lines)
    assert level_for(1, ["just a skills list"]) == "basic"
    assert level_for(4, ["anything"]) == "advanced"


def test_extract_skills_with_llm_returns_skill_dict():
    from app.skills.extract import extract_skills_with_llm

    llm = _FakeLLM(_payload([
        {"skill_id": "python", "level": "advanced", "mentions": 3,
         "evidence": ["Architected Python pipeline"]},
    ]))
    out = extract_skills_with_llm(RESUME_AI, STUB_TAX, llm)
    assert isinstance(out, dict)
    assert "python" in out
    assert isinstance(out["python"], dict)
    assert out["python"]["skill_id"] == "python"
    assert out["python"]["level"] in {"basic", "intermediate", "advanced"}
    assert isinstance(out["python"]["evidence"], list)


def test_resume_extractor_module_reexports():
    import importlib
    mod = importlib.import_module("app.resume.extractor")
    for name in ("EXTRACT_RESUME_SYSTEM", "extract_resume_with_ai",
                 "extract_skills_with_llm", "extract_skills", "level_for"):
        assert hasattr(mod, name), f"app.resume.extractor missing {name}"
