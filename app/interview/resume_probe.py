"""Resume-grounded project deep-dive question generator (Milestone 18.2).

Synthesizes up to two ``Question`` objects quoted directly from the candidate's
own resume evidence (project bullets, internship achievements, quantified
impact) so the interviewer can probe architecture, failure modes and
performance measurement on work the student actually did. Deterministic for the
offline MockLLM; degrades to a rule-based template without an LLM.

Questions use ids ``q-resume-01`` / ``q-resume-02`` which never collide with the
static ``data/questions.json`` bank (those are ``q-<role>-...``).
"""
from __future__ import annotations

from app.interview.bank import Question

#: Deterministic resume-probe task for the offline MockLLM.
RESUME_PROBE_SYSTEM = (
    "TASK: resume_probe\n"
    "You are a technical interviewer. Given a resume evidence snippet and the skill "
    "used, write ONE deep-dive question on architecture, failure modes and performance "
    "measurement. Respond ONLY with a plain-string question (no JSON)."
)

#: Core systems-design keywords every probe expects to be touched.
_DEFAULT_KEYWORDS = ("architecture", "failure modes", "performance", "trade-offs")


def _skill_name(skill) -> str:
    """Best-effort display name for a StudentSkill row (id fallback)."""
    return str(getattr(skill, "skill_id", "") or "the stack")


def _evidence_lines(student, skills) -> list[tuple[str, str]]:
    """Return ``[(snippet, skill_name), ...]`` grounded in resume + evidence.

    Prefers ``StudentSkill.evidence`` lines (already-scraped resume bullets);
    falls back to meaningful resume sentences when no evidence rows exist.
    Empty result => no probes (never fabricate questions from a blank resume).
    """
    lines: list[tuple[str, str]] = []
    seen: set[str] = set()
    for sk in skills or []:
        for ev in list(getattr(sk, "evidence", []) or []):
            snippet = str(ev).strip()
            if len(snippet) < 15 or snippet.lower() in seen:
                continue
            seen.add(snippet.lower())
            lines.append((snippet, _skill_name(sk)))
    if lines:
        return lines[:2]

    # Fallback: pull meaningful sentences from the raw resume text.
    resume = str(getattr(student, "resume_text", "") or "").strip()
    if not resume:
        return []
    skill_names = [s.name if hasattr(s, "name") else _skill_name(s) for s in (skills or [])]
    for sentence in _split_sentences(resume):
        low = sentence.lower()
        if len(sentence) < 25:
            continue
        if not any(tok in low for tok in ("built", "developed", "implemented", "designed",
                                          "project", "internship", "led", "created")):
            continue
        matched = next((n for n in skill_names if n and n.lower() in low), skill_names[0] if skill_names else "the stack")
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append((sentence, matched or "the stack"))
    return lines[:2]


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    return [p if p.endswith(".") else p + "." for p in parts]


def generate_resume_project_questions(student, skills, taxonomy, llm=None) -> list[Question]:
    """Build up to 2 resume-grounded project deep-dive questions (M18.2).

    ``skills`` is a list of ``StudentSkill`` rows; ``taxonomy`` is used to pull
    canonical skill display names. Returns ``[]`` when the resume/evidence gives
    nothing real to probe (never fabricates).
    """
    pairs = _evidence_lines(student, skills)
    if not pairs:
        return []

    questions: list[Question] = []
    for idx, (snippet, skill_name) in enumerate(pairs[:2], start=1):
        # Prefer the taxonomy display name when available.
        display = skill_name
        if taxonomy is not None:
            try:
                resolved = taxonomy.resolve(skill_name) or taxonomy.resolve(skill_name.lower())
                if resolved:
                    display = taxonomy.get_opt(resolved).name or skill_name
            except Exception:
                display = skill_name
        question_text = _synthesize(snippet, display, llm)
        # expected keywords: skill token + core systems-design concepts
        keywords = [tok for tok in (display.split()[0].lower() if display else "",) if tok]
        keywords += list(_DEFAULT_KEYWORDS)
        outline = [
            "Describe the architecture end to end (components and data flow)",
            "Name concrete failure modes you guarded against and how",
            "Explain how performance was measured (metrics, load, baseline)",
            "Call out the key trade-offs you made and what you would revisit",
        ]
        questions.append(Question(
            id=f"q-resume-{idx:02d}",
            role="generic",
            skill_id=(skills[0].skill_id if skills else "generic"),
            difficulty=2,
            question=question_text,
            keywords=tuple(keywords),
            outline=tuple(outline),
        ))
    return questions


def _synthesize(snippet: str, skill_name: str, llm=None) -> str:
    if llm is not None:
        try:
            import json
            payload = json.dumps({"evidence": snippet, "skill": skill_name})
            raw = str(llm.complete(system=RESUME_PROBE_SYSTEM, user=payload,
                                   json_mode=False) or "").strip()
            if raw:
                return raw if raw.endswith("?") else raw + "?"
        except Exception:
            pass
    return (f"In your resume, you mentioned building '{snippet}' using {skill_name}. "
            f"Walk through the architecture, failure modes, and how you measured its "
            f"performance under load?")
