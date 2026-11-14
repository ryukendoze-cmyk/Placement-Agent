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


def _evidence_entries(student, skills) -> list[tuple[str, str, str]]:
    """Return ``[(snippet, skill_name, skill_id), ...]`` grounded in resume + evidence.

    Prefers distinct skills first (round-robin across skills' evidence lines) so
    multi-skill resumes get diverse probes; falls back to resume sentences.
    """
    entries: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    # Collect per-skill clean evidence lists, then pick round-robin up to 2.
    per_skill: list[list[tuple[str, str, str]]] = []
    for sk in skills or []:
        sid = str(getattr(sk, "skill_id", "") or "generic")
        sname = _skill_name(sk)
        bucket: list[tuple[str, str, str]] = []
        for ev in list(getattr(sk, "evidence", []) or []):
            snippet = str(ev).strip()
            if len(snippet) < 15 or snippet.lower() in seen or snippet.lower().startswith(("[interview ", "interview ")):
                continue
            seen.add(snippet.lower())
            bucket.append((snippet, sname, sid))
        if bucket:
            per_skill.append(bucket)
    depth = 0
    while len(entries) < 2 and per_skill:
        added_this_round = False
        for bucket in per_skill:
            if depth < len(bucket) and len(entries) < 2:
                entries.append(bucket[depth])
                added_this_round = True
        if not added_this_round:
            break
        depth += 1
    if entries:
        return entries[:2]

    # Fallback: pull meaningful sentences from the raw resume text.
    resume = str(getattr(student, "resume_text", "") or "").strip()
    if not resume:
        return []
    skill_pairs = [
        (s.name if hasattr(s, "name") else _skill_name(s), str(getattr(s, "skill_id", "") or "generic"))
        for s in (skills or [])
    ]
    for sentence in _split_sentences(resume):
        low = sentence.lower()
        if len(sentence) < 25:
            continue
        if not any(tok in low for tok in ("built", "developed", "implemented", "designed",
                                          "project", "internship", "led", "created")):
            continue
        matched_name, matched_sid = next(
            ((n, sid) for n, sid in skill_pairs if n and n.lower() in low),
            skill_pairs[0] if skill_pairs else ("the stack", "generic"),
        )
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append((sentence, matched_name or "the stack", matched_sid or "generic"))
    return entries[:2]


def _evidence_lines(student, skills) -> list[tuple[str, str]]:
    """Return ``[(snippet, skill_name), ...]`` grounded in resume + evidence.

    Prefers ``StudentSkill.evidence`` lines (already-scraped resume bullets);
    falls back to meaningful resume sentences when no evidence rows exist.
    Empty result => no probes (never fabricate questions from a blank resume).
    """
    return [(snippet, skill_name) for snippet, skill_name, _ in _evidence_entries(student, skills)]


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    return [p if p.endswith(".") else p + "." for p in parts]


def generate_resume_project_questions(student, skills, taxonomy, llm=None) -> list[Question]:
    """Build up to 2 resume-grounded project deep-dive questions (M18.2).

    ``skills`` is a list of ``StudentSkill`` rows; ``taxonomy`` is used to pull
    canonical skill display names. Returns ``[]`` when the resume/evidence gives
    nothing real to probe (never fabricates).
    """
    entries = _evidence_entries(student, skills)
    if not entries:
        return []

    questions: list[Question] = []
    for idx, (snippet, skill_name, entry_sid) in enumerate(entries[:2], start=1):
        # Prefer the taxonomy display name and canonical skill_id when available.
        display = skill_name
        resolved_sid = entry_sid or "generic"
        if taxonomy is not None:
            try:
                resolved = (
                    taxonomy.resolve(resolved_sid)
                    or taxonomy.resolve(skill_name)
                    or taxonomy.resolve(skill_name.lower())
                )
                if resolved:
                    resolved_sid = resolved
                    opt = taxonomy.get_opt(resolved)
                    if opt and opt.name:
                        display = opt.name
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
            skill_id=resolved_sid,
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
