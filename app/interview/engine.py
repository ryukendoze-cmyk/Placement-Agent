"""Adaptive mock-interview state machine. Pure layer - no FastAPI/SQLModel.

All evolving state lives in the `transcript` list (see plan section 0.1), so a
session is fully reconstructed from an Interview row on each request.
"""
import json
import logging
import re
from dataclasses import dataclass, field

from app.interview.bank import Question, get_question, select
from app.security.guardrails import sanitize_untrusted_text

logger = logging.getLogger(__name__)

MAX_TURNS = 8

EVALUATE_ANSWER_SYSTEM = (
    "TASK: evaluate_answer\n"
    "You are a strict, zero-leniency senior technical interviewer grading an answer for placement readiness.\n"
    "Verify factual correctness strictly against the question, expected keywords, and reference outline.\n"
    "Rules:\n"
    "1. Simply repeating words from the question, dropping buzzwords, writing vague filler "
    "(e.g., 'plays an important role', 'it is core architecture', 'should be efficient'), or inventing false "
    "definitions earns ZERO technical credit.\n"
    "2. Never inflate scores for grammar, length, or confident tone when technical facts are missing or wrong.\n"
    "3. Classify verdict strictly:\n"
    "   - 'wrong_or_baseless' (score 0-18, technical_accuracy 0-12, depth_and_examples 0-10): blank, 'I don't know', "
    "gibberish, fabricated/wrong definitions, word salad, off-topic, or zero accurate technical claims.\n"
    "   - 'vague_superficial' (score 19-36, technical_accuracy 15-35): hand-wavy intuition or copied terms without "
    "explaining the mechanism.\n"
    "   - 'partial' (score 37-62): partially correct mechanism, missing major outline points or containing errors.\n"
    "   - 'solid' (score 63-84): factually accurate explanation covering most outline points with clear reasoning.\n"
    "   - 'strong' (score 85-100): complete, accurate, production-grade answer with concrete examples and trade-offs.\n"
    "Respond ONLY with a valid JSON object containing keys: "
    "\"verdict\" (one of \"wrong_or_baseless\", \"vague_superficial\", \"partial\", \"solid\", \"strong\"), "
    "\"accurate_claims\" (list of accurate technical points actually explained; [] if none), "
    "\"errors_or_bluffs\" (list of false claims, buzzword salad, or vague filler detected), "
    "\"score\" (integer 0-100), \"feedback\" (direct 1-2 sentence evaluation; state clearly if wrong/baseless, never flatter), "
    "\"rubric\" (object with integer keys \"technical_accuracy\", \"depth_and_examples\", \"communication_clarity\"), "
    "\"matched_keywords\" (list of strings actually used accurately), "
    "\"missed_keywords\" (list of strings), and \"ideal_answer_hint\" (one sentence describing a strong answer)."
)

INTERVIEW_FEEDBACK_SYSTEM = (
    "TASK: interview_feedback\n"
    "You are a technical interviewer summarizing a completed mock interview. "
    "Given the overall score (0 to 100) and per-skill breakdown, write a concise 2-3 sentence summary "
    "highlighting strengths, weak areas to revisit, and encouragement. "
    "Respond ONLY with a valid JSON object containing a single key \"feedback\" (string)."
)


def _safe_json(raw: str):
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned or raw)
    except (ValueError, TypeError):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except (ValueError, TypeError):
                return None
        return None


def _coerce_score(val) -> int:
    if isinstance(val, bool) or val is None:
        return 0
    if isinstance(val, (int, float)):
        return round(val)
    if isinstance(val, str):
        try:
            return round(float(val.strip()))
        except ValueError:
            return 0
    return 0


_DEVOPS_RE = re.compile(r"\b(devops|site reliability|cloud|sre|infrastructure|platform)\b")
_DATA_RE = re.compile(r"\b(data|analytics?|machine learning|ml|ai|vision|scientist|bi)\b")
_FRONTEND_RE = re.compile(
    r"\b(frontend|front-end|react|vue|angular|next\.?js|web developer|full[- ]?stack|mern|ui/ux|ui|ux|ios|android|mobile)\b"
)
_NON_ANSWER_RE = re.compile(
    r"^\s*(?:i\s*(?:do\s*not|don'?t)\s*know|idk|no\s*idea|not\s*sure|skip|pass|none|n/?a)\W*$",
    re.IGNORECASE,
)


def _is_non_answer(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    if _NON_ANSWER_RE.match(s):
        return True
    low = s.lower()
    return len(low.split()) <= 6 and any(
        p in low for p in ("don't know", "dont know", "do not know", "no idea", "skip")
    )


_INTERVIEW_STOPWORDS = frozenset({
    "about", "above", "across", "after", "again", "against", "also", "always", "another",
    "because", "before", "being", "below", "between", "both", "could", "does", "doing",
    "done", "during", "each", "either", "every", "first", "from", "have", "having", "here",
    "however", "important", "into", "just", "like", "main", "make", "many", "means", "more",
    "most", "much", "need", "only", "other", "overall", "play", "plays", "process", "role",
    "same", "scenario", "should", "since", "some", "such", "than", "that", "their", "them",
    "then", "there", "these", "they", "thing", "think", "this", "those", "through", "under",
    "used", "using", "very", "what", "when", "where", "which", "while", "with", "would", "your",
})

_NEGATIVE_FEEDBACK_RE = re.compile(
    r"\b(baseless|completely wrong|incorrect|inaccurate|nonsense|irrelevant|off[- ]topic|"
    r"no technical|does not answer|fails to explain|fabricated|meaningless)\b",
    re.IGNORECASE,
)


def _content_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9_+#-]{3,}", (text or "").lower())
        if tok not in _INTERVIEW_STOPWORDS
    }


def _answer_grounding_stats(q: Question, answer_text: str) -> dict:
    """Measure keyword hits, non-question-parroted hits, and outline concept overlap."""
    low = (answer_text or "").lower()
    keywords = [str(k).strip() for k in (q.keywords or ()) if str(k).strip()]
    matched = [k for k in keywords if re.search(rf"\b{re.escape(k.lower())}", low)]
    missed = [k for k in keywords if k not in matched]

    q_tokens = _content_tokens(q.question)
    ans_tokens = _content_tokens(low)
    outline_text = " ".join(str(x) for x in (q.outline or ()))
    outline_tokens = _content_tokens(outline_text) - q_tokens
    outline_hits = sorted(ans_tokens & outline_tokens)
    non_question_kw_hits = [k for k in matched if k.lower() not in q.question.lower()]

    raw_words = re.findall(r"[a-z0-9_+#-]+", low)
    total_words = len(raw_words)
    unique_ratio = (len(set(raw_words)) / total_words) if total_words else 1.0
    max_freq = max((raw_words.count(w) for w in set(raw_words)), default=0) if total_words else 0
    is_stuffed = total_words >= 5 and (
        unique_ratio < 0.45 or (total_words >= 6 and max_freq / total_words > 0.35)
    )

    return {
        "matched": matched,
        "missed": missed,
        "keywords_total": len(keywords),
        "outline_hits": outline_hits,
        "outline_total": len(outline_tokens),
        "non_question_kw_hits": non_question_kw_hits,
        "word_count": total_words,
        "unique_ratio": unique_ratio,
        "is_stuffed": is_stuffed,
    }


def _deterministic_evaluate(q: Question, answer_text: str) -> dict:
    """Strict offline fallback evaluator when the configured LLM fails or emits invalid JSON."""
    text = (answer_text or "").strip()
    stats = _answer_grounding_stats(q, text)
    matched = stats["matched"]
    missed = stats["missed"]
    hint = "; ".join(q.outline[:2]) if q.outline else (
        f"Cover {', '.join(missed[:3])} with a concrete implementation example." if missed else ""
    )

    if _is_non_answer(text):
        return {
            "verdict": "wrong_or_baseless",
            "accurate_claims": [],
            "errors_or_bluffs": ["No technical explanation provided."],
            "score": 10,
            "feedback": "No technical explanation provided. Review core fundamentals and outline a concrete mechanism.",
            "rubric": {
                "technical_accuracy": 5,
                "depth_and_examples": 5,
                "communication_clarity": 20,
            },
            "matched_keywords": [],
            "missed_keywords": list(q.keywords or ()),
            "ideal_answer_hint": hint,
        }

    words = stats["word_count"]
    keywords = [str(k).strip() for k in (q.keywords or ()) if str(k).strip()]
    kw_ratio = len(matched) / max(1, len(keywords)) if keywords else 0.0
    outline_ratio = min(1.0, len(stats["outline_hits"]) / max(2, min(4, stats["outline_total"] or 3)))

    if stats["is_stuffed"] or (len(matched) == 0 and len(stats["outline_hits"]) == 0):
        return {
            "verdict": "wrong_or_baseless",
            "accurate_claims": [],
            "errors_or_bluffs": ["Answer lacks accurate technical concepts or repeats filler."],
            "score": 12 if words >= 4 else 8,
            "feedback": (
                f"Answer did not accurately explain key concepts ({', '.join(missed[:3]) or 'core mechanism'}). "
                "Focus on exact definitions and concrete mechanisms instead of vague phrasing."
            ),
            "rubric": {
                "technical_accuracy": 10,
                "depth_and_examples": 8,
                "communication_clarity": 25 if words >= 6 else 15,
            },
            "matched_keywords": matched,
            "missed_keywords": missed,
            "ideal_answer_hint": hint,
        }

    # If the candidate only parroted a single word from the question with zero outline concepts, keep strictly low
    if len(matched) == 1 and not stats["non_question_kw_hits"] and len(stats["outline_hits"]) == 0:
        return {
            "verdict": "vague_superficial",
            "accurate_claims": [],
            "errors_or_bluffs": ["Merely echoed question terminology without explaining how it works."],
            "score": 20,
            "feedback": (
                f"Merely mentioning '{matched[0]}' without explaining how it works is insufficient. "
                f"Address {', '.join(missed[:3]) or 'the underlying mechanism'}."
            ),
            "rubric": {
                "technical_accuracy": 18,
                "depth_and_examples": 15,
                "communication_clarity": 35,
            },
            "matched_keywords": matched,
            "missed_keywords": missed,
            "ideal_answer_hint": hint,
        }

    low = text.lower()
    depth_signals = sum(
        1
        for tok in ("because", "example", "for instance", "trade-off", "tradeoff", "latency", "throughput", "index", "cache", "test", "scale")
        if tok in low
    )
    evidence_factor = min(1.0, 0.7 * kw_ratio + 0.3 * outline_ratio)
    tech = min(94, max(15, round(10 + 82 * evidence_factor)))
    word_depth = min(22, words // 2) if (kw_ratio > 0 or outline_ratio > 0) else 0
    depth = min(90, max(12, round(10 + 48 * evidence_factor + word_depth + min(12, depth_signals * 4))))
    clarity = min(90, max(25, round(25 + min(30, words) * max(0.35, evidence_factor) + (12 if words >= 12 and evidence_factor >= 0.4 else 0))))
    score = round((tech * 0.55) + (depth * 0.30) + (clarity * 0.15))

    if matched and missed:
        fb = f"Covered {', '.join(matched[:3])}. Deepen your answer on {', '.join(missed[:3])}."
    elif matched:
        fb = f"Solid coverage of {', '.join(matched[:3])} with clear reasoning."
    else:
        fb = f"Partially addressed {', '.join(stats['outline_hits'][:3])}, but missed {', '.join(missed[:3])}."
    verdict = "strong" if score >= 82 else ("solid" if score >= 63 else ("partial" if score >= 37 else "vague_superficial"))
    return {
        "verdict": verdict,
        "accurate_claims": matched[:3],
        "errors_or_bluffs": [],
        "score": max(0, min(100, score)),
        "feedback": fb,
        "rubric": {
            "technical_accuracy": tech,
            "depth_and_examples": depth,
            "communication_clarity": clarity,
        },
        "matched_keywords": matched,
        "missed_keywords": missed,
        "ideal_answer_hint": hint,
    }


def _reconcile_evaluation(q: Question, answer_text: str, parsed: dict) -> dict:
    """Enforce strict technical grading invariants on both live LLM and fallback outputs.

    Prevents lenient LLMs from awarding 45-65/100 to fabricated definitions,
    question-parroting, or vague word salad just because answers are grammatically fluent.
    """
    out = dict(parsed or {})
    stats = _answer_grounding_stats(q, answer_text)
    score = max(0, min(100, _coerce_score(out.get("score"))))
    feedback = str(out.get("feedback") or "").strip()
    raw_rub = out.get("rubric") if isinstance(out.get("rubric"), dict) else {}
    rubric = {
        k: max(0, min(100, _coerce_score(raw_rub.get(k, score))))
        for k in RUBRIC_KEYS
    }
    matched = [str(x) for x in (out.get("matched_keywords") or []) if str(x).strip()]
    missed = [str(x) for x in (out.get("missed_keywords") or []) if str(x).strip()]
    ideal = str(out.get("ideal_answer_hint") or "").strip()
    verdict = str(out.get("verdict") or "").strip().lower()
    raw_claims = out.get("accurate_claims")
    has_claims_field = isinstance(raw_claims, list)
    accurate_claims = [str(x).strip() for x in raw_claims if str(x).strip()] if has_claims_field else []
    raw_errors = out.get("errors_or_bluffs")
    errors_or_bluffs = [str(x).strip() for x in raw_errors if str(x).strip()] if isinstance(raw_errors, list) else []

    tech = rubric["technical_accuracy"]
    depth = rubric["depth_and_examples"]
    comm = rubric["communication_clarity"]

    if _is_non_answer(answer_text) or stats["is_stuffed"]:
        verdict = "wrong_or_baseless"
        tech = min(tech, 10)
        depth = min(depth, 8)
        comm = min(comm, 25)
        score = min(score, 12)
        if not feedback:
            feedback = "No valid technical explanation was provided."
    elif verdict == "wrong_or_baseless" or (has_claims_field and not accurate_claims and (errors_or_bluffs or tech < 60)):
        verdict = "wrong_or_baseless"
        tech = min(tech, 15)
        depth = min(depth, 10)
        score = min(score, 15)
    elif verdict == "vague_superficial":
        tech = min(tech, 32)
        depth = min(depth, 25)
        score = min(score, 32)
    else:
        # Cross-check against baseless bluffing / question parroting when no outline or
        # non-question keywords are supported and accurate_claims is absent/empty.
        zero_grounding = (
            len(stats["matched"]) == 0
            and len(stats["outline_hits"]) == 0
            and not accurate_claims
        )
        parrot_only = (
            len(stats["non_question_kw_hits"]) == 0
            and len(stats["outline_hits"]) == 0
            and len(stats["matched"]) <= 1
            and not accurate_claims
        )
        if zero_grounding and (stats["word_count"] < 4 or _NEGATIVE_FEEDBACK_RE.search(feedback) or tech <= 65):
            verdict = "wrong_or_baseless"
            tech = min(tech, 15)
            depth = min(depth, 10)
            score = min(score, 15)
        elif parrot_only and (_NEGATIVE_FEEDBACK_RE.search(feedback) or tech <= 65):
            verdict = verdict or "vague_superficial"
            tech = min(tech, 22)
            depth = min(depth, 18)
            score = min(score, 22)
        elif _NEGATIVE_FEEDBACK_RE.search(feedback) and tech <= 55:
            tech = min(tech, 28)
            depth = min(depth, 22)
            score = min(score, 28)

    # Enforce technical-first composite weighting so communication_clarity cannot rescue wrong tech
    weighted = round(0.60 * tech + 0.25 * depth + 0.15 * comm)
    score = min(score, weighted + 3)

    if tech <= 15:
        score = min(score, 15)
    elif tech <= 25:
        score = min(score, 25)
    elif tech <= 35:
        score = min(score, 35)
    elif tech <= 45:
        score = min(score, 45)

    rubric = {
        "technical_accuracy": max(0, min(100, tech)),
        "depth_and_examples": max(0, min(100, depth)),
        "communication_clarity": max(0, min(100, comm)),
    }
    return {
        "verdict": verdict or ("strong" if score >= 82 else ("solid" if score >= 63 else ("partial" if score >= 37 else "wrong_or_baseless"))),
        "accurate_claims": accurate_claims,
        "errors_or_bluffs": errors_or_bluffs,
        "score": max(0, min(100, score)),
        "feedback": feedback,
        "rubric": rubric,
        "matched_keywords": matched,
        "missed_keywords": missed,
        "ideal_answer_hint": ideal,
    }


def role_for_job(job) -> str:
    t = (getattr(job, "title", "") or "").lower()
    if _DEVOPS_RE.search(t):
        return "devops"
    if _DATA_RE.search(t):
        return "data"
    if _FRONTEND_RE.search(t):
        return "frontend"
    return "backend"


RUBRIC_KEYS = ("technical_accuracy", "depth_and_examples", "communication_clarity")


@dataclass
class TurnResult:
    score: int
    feedback: str
    next_difficulty: int
    question_id: str
    question: str
    finished: bool
    rubric: dict = field(default_factory=dict)
    matched_keywords: list = field(default_factory=list)
    missed_keywords: list = field(default_factory=list)
    ideal_answer_hint: str = ""
    #: M15: prompt-injection patterns detected (and neutralized) in the answer text.
    injection_flags: list = field(default_factory=list)
    #: M18: contextual follow-up probe generated for this turn ("" when none).
    followup_probe: str = ""
    #: Non-empty when LLM failure/malformed JSON recovered via deterministic fallback.
    error_recovered: str = ""


#: Deterministic follow-up probe task for the offline MockLLM.
FOLLOWUP_PROBE_SYSTEM = (
    "TASK: followup_probe\n"
    "You are a technical interviewer. The candidate's answer missed key concepts. "
    "Write ONE concise contextual follow-up question probing the missed concept on the "
    "same skill. Respond ONLY with a plain-string question (no JSON)."
)


def generate_followup_probe(question: str, answer: str, missed_keywords: list[str],
                            skill_id: str, llm=None) -> str:
    """Produce a single contextual follow-up question for the missed concept (M18.1).

    Deterministic for the offline MockLLM; degrades to a rule-based template when no
    LLM is configured or the LLM returns nothing usable. Never returns an empty string
    when there is at least one missed keyword to probe.
    """
    missed = [str(k).strip() for k in (missed_keywords or []) if str(k).strip()]
    if not missed:
        return ""
    if llm is not None:
        try:
            payload = json.dumps({"question": question, "answer": answer,
                                  "missed_keywords": missed, "skill_id": skill_id})
            raw = llm.complete(system=FOLLOWUP_PROBE_SYSTEM, user=payload, json_mode=False)
            probe = str(raw or "").strip()
            if probe:
                if not probe.endswith("?"):
                    probe += "?"
                return probe
        except Exception:
            pass
    # rule-based fallback: probe the first missed keyword
    kw = missed[0]
    return (f"Could you go deeper on '{kw}'? Walk me through how it applies to "
            f"{skill_id} and the trade-offs you would weigh.")


@dataclass
class InterviewSummary:
    score: int
    per_skill: dict
    feedback: str
    transcript: list = field(default_factory=list)
    rubric_overall: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    assessment_per_skill: dict | None = None


class InterviewSession:
    def __init__(self, role: str, bank, llm, prefer_skills, transcript,
                 enable_followups: bool = False, resume_questions=None, interview_plan=None):
        self.role = role
        self.bank = bank
        self.llm = llm
        self.prefer_skills = list(prefer_skills or [])
        self.transcript = [dict(t) for t in (transcript or [])]
        #: M18.1: opt-in adaptive follow-up probing (default off -> legacy id sequences).
        self.enable_followups = bool(enable_followups)
        #: M18.2: resume-grounded probes served before static bank questions.
        self.resume_questions = list(resume_questions or [])
        # None preserves direct callers and historical interview semantics.
        self.interview_plan = interview_plan

    @property
    def coverage(self):
        targets = (self.interview_plan or {}).get("targets", [])
        result = []
        for target in targets:
            turns = [t for t in self.transcript if t.get("verification_target") == target["skill_id"]]
            answers = [t for t in turns if t.get("score") is not None]
            result.append({"skill_id": target["skill_id"], "name": target.get("name", target["skill_id"]),
                           "original_status": target["verification_status"],
                           "status": "assessed" if answers else "asked" if turns else "pending",
                           "score": round(sum(t["score"] for t in answers) / len(answers)) if answers else None,
                           "assessment_limited": any(t.get("assessment_limited") or t.get("evaluation_source") == "offline_fallback" for t in answers)})
        assessed = sum(r["status"] == "assessed" for r in result)
        return {"targets": result, "total": len(result), "assessed": assessed,
                "remaining": len(result) - assessed, "complete": assessed == len(result),
                "answered_turns": self.turn,
                "base_question_goal": max(MAX_TURNS, len(result)) if self.interview_plan is not None else MAX_TURNS,
                "base_questions_answered": sum(not t.get("is_followup") for t in self.answered)}

    @property
    def answered(self):
        return [t for t in self.transcript if t.get("score") is not None]

    @property
    def turn(self):
        return len(self.answered)

    @property
    def finished(self):
        if self.interview_plan is not None:
            return (self.pending is None and self.coverage["complete"]
                    and self.coverage["base_questions_answered"] >= self.coverage["base_question_goal"])
        return self.turn >= MAX_TURNS

    @property
    def used_ids(self):
        return [t["q_id"] for t in self.transcript]

    @property
    def pending(self):
        if self.transcript and self.transcript[-1].get("score") is None:
            return self.transcript[-1]
        return None

    @property
    def current_difficulty(self):
        if self.pending is not None:
            return self.pending.get("difficulty", 1)
        if self.transcript:
            return self.transcript[-1].get("next_difficulty", 1)
        return 1

    def _dynamic_question_from_pending(self, pend):
        """Rebuild a Question for dynamic ids (M18: ``q-*-followup``,
        ``q-resume-*``) that are NOT in the static bank. Returns None for
        static/ghost ids so the caller keeps raising for unknown questions."""
        if not (pend.get("is_followup") or pend.get("is_resume_probe") or pend.get("question_source")):
            return None
        return Question(
            id=pend["q_id"],
            role=self.role,
            skill_id=pend.get("skill_id", "generic"),
            difficulty=int(pend.get("difficulty", 1)),
            question=pend.get("q", ""),
            keywords=tuple(pend.get("keywords") or ()),
            outline=tuple(pend.get("outline") or ()),
        )

    def next_question(self):
        if self.finished:
            return None
        if self.pending is not None:
            dyn = self._dynamic_question_from_pending(self.pending)
            if dyn is not None:
                return dyn
            return get_question(self.bank, self.pending["q_id"])
        if self.interview_plan is not None:
            return self._next_planned_question()
        # M18.2: serve unasked resume probes before the static bank so
        # lexicographic tie-breaking can never starve them. Keywords/outline are
        # persisted on the transcript turn so the next reconstructed request can
        # rehydrate the probe (the bank never contains q-resume-*).
        for rq in self.resume_questions:
            if rq.id not in self.used_ids:
                d = rq.difficulty if rq.difficulty is not None else self.current_difficulty
                self.transcript.append({"q_id": rq.id, "q": rq.question,
                                        "skill_id": rq.skill_id,
                                        "a": "", "difficulty": d, "score": None,
                                        "next_difficulty": self.current_difficulty,
                                        "keywords": list(rq.keywords),
                                        "outline": list(rq.outline),
                                        "is_resume_probe": True})
                return rq
        q = select(self.bank, self.role, self.current_difficulty, self.used_ids, self.prefer_skills)
        if q is None:
            return None
        d = self.current_difficulty
        self.transcript.append({"q_id": q.id, "q": q.question, "skill_id": q.skill_id,
                                "a": "", "difficulty": d, "score": None, "next_difficulty": d})
        return q

    def _next_planned_question(self):
        from app.interview.generation import author_question
        plan = self.interview_plan
        covered = {t.get("verification_target") for t in self.answered if not t.get("is_followup")}
        target = next((t for t in plan["targets"] if t["skill_id"] not in covered), None)
        verification = target is not None
        if target is None:
            candidates = plan.get("candidate_skills") or [{"skill_id": s, "verification_status": "verified"} for s in self.prefer_skills]
            candidates = candidates or [{"skill_id": "generic", "name": self.role, "verification_status": "verified"}]
            target = candidates[self.coverage["base_questions_answered"] % len(candidates)]
        difficulty = {"basic": 0, "intermediate": 1, "advanced": 2}.get(target.get("level"), self.current_difficulty) if verification else self.current_difficulty
        q, provenance = author_question(llm=self.llm, bank=self.bank, role=self.role, target=target,
            difficulty=difficulty, qid=f"q-ai-{len(self.transcript) + 1}",
            context=self._generation_context(), used_ids=self.used_ids)
        self._append_planned(q, provenance, target["skill_id"] if verification else None)
        return q

    def _generation_context(self):
        return {**self.interview_plan.get("context", {}),
                "recent_turns": [{"question": t.get("q"), "answer": t.get("a"),
                                  "skill_id": t.get("skill_id"), "score": t.get("score")}
                                 for t in self.answered[-4:]]}

    def _append_planned(self, q, provenance, verification_target=None, is_followup=False):
        self.transcript.append({"q_id": q.id, "q": q.question, "skill_id": q.skill_id,
            "a": "", "difficulty": q.difficulty, "score": None, "next_difficulty": q.difficulty,
            "keywords": list(q.keywords), "outline": list(q.outline),
            "verification_target": verification_target, "is_followup": is_followup, **provenance})

    def answer(self, text: str) -> TurnResult:
        pend = self.pending
        if pend is None:
            self.next_question()
            pend = self.pending
            if pend is None:
                raise ValueError("no question to answer")
        q = self._dynamic_question_from_pending(pend)
        if q is None:
            q = get_question(self.bank, pend["q_id"])
        if q is None and self.resume_questions:
            # M18.2: resume probes are session-scoped (regenerated), not in the bank.
            q = get_question(self.resume_questions, pend["q_id"])
        if q is None:
            raise ValueError(f"question '{pend['q_id']}' not found in question bank")
        # M15 indirect prompt-injection gate: neutralize adversarial phrasing
        # BEFORE it reaches the evaluator LLM; the transcript stores the
        # sanitized text so downstream prompts never see the raw payload.
        text, inj_flags = sanitize_untrusted_text(str(text or ""), source="interview_answer")
        payload = json.dumps({"question": q.question, "answer": text,
                              "keywords": list(q.keywords),
                              "outline": list(q.outline)})
        from app.interview.generation import complete_live, is_live
        evaluate = self.llm.complete if self.llm is not None else None
        if self.interview_plan is not None and is_live(self.llm):
            evaluate = lambda **kw: complete_live(self.llm, **kw)
        parsed = None
        error_recovered = ""
        try:
            raw_complete = evaluate(system=EVALUATE_ANSWER_SYSTEM, user=payload, json_mode=True)
            parsed = _safe_json(raw_complete)
            if self.interview_plan is not None and (not isinstance(parsed, dict)
                    or not isinstance(parsed.get("rubric"), dict) or "score" not in parsed):
                parsed = None
        except Exception as exc:
            logger.warning("Interview answer evaluation failed (%s); trying recovery", exc)
            error_recovered = f"llm_error:{type(exc).__name__}"

        if not isinstance(parsed, dict) and not error_recovered:
            # 1 concise JSON-repair retry on malformed string output
            try:
                repair_prompt = (
                    "Your previous output was not valid JSON. Return ONLY valid JSON with keys: "
                    "score, feedback, rubric, matched_keywords, missed_keywords, ideal_answer_hint.\n"
                    + payload
                )
                raw_retry = evaluate(system=EVALUATE_ANSWER_SYSTEM, user=repair_prompt, json_mode=True)
                parsed = _safe_json(raw_retry)
                if self.interview_plan is not None and (not isinstance(parsed, dict)
                        or not isinstance(parsed.get("rubric"), dict) or "score" not in parsed):
                    parsed = None
                if not isinstance(parsed, dict):
                    error_recovered = "invalid_json_fallback"
            except Exception as exc:
                logger.warning("Interview answer JSON-repair retry failed (%s)", exc)
                error_recovered = f"llm_retry_error:{type(exc).__name__}"

        if not isinstance(parsed, dict):
            parsed = _deterministic_evaluate(q, text)
            if not error_recovered:
                error_recovered = "deterministic_rubric_fallback"

        reconciled = _reconcile_evaluation(q, text, parsed)
        score = reconciled["score"]
        feedback = reconciled["feedback"]
        rubric = reconciled["rubric"]
        matched = reconciled["matched_keywords"]
        missed = reconciled["missed_keywords"]
        ideal = reconciled["ideal_answer_hint"]
        verdict = reconciled["verdict"]
        d = pend["difficulty"]
        if score >= 70:
            nd = min(2, d + 1)
        elif score < 40:
            nd = max(0, d - 1)
        else:
            nd = d
        pend["a"] = text
        pend["score"] = score
        pend["feedback"] = feedback
        pend["verdict"] = verdict
        pend["next_difficulty"] = nd
        pend["rubric"] = rubric
        pend["matched_keywords"] = matched
        pend["missed_keywords"] = missed
        pend["ideal_answer_hint"] = ideal
        pend["evaluation_source"] = "offline_fallback" if error_recovered else "live_ai" if is_live(self.llm) else "mock"
        if error_recovered:
            pend["error_recovered"] = error_recovered

        # --- M18.1: adaptive follow-up probing (opt-in) ----------------------
        followup_probe = ""
        if (self.enable_followups
                and not pend.get("is_followup")
                and (self.interview_plan is not None or self.turn < MAX_TURNS)
                and missed
                and not _is_non_answer(text)
                and (score < 75 or rubric.get("depth_and_examples", 100) < 60)):
            if self.interview_plan is not None:
                from app.interview.generation import author_question
                target = next((t for t in self.interview_plan["targets"] if t["skill_id"] == q.skill_id),
                              {"skill_id": q.skill_id, "verification_status": "verified"})
                probe, provenance = author_question(llm=self.llm, bank=self.bank, role=self.role,
                    target=target, difficulty=nd, qid=f"{q.id}-followup",
                    context=self._generation_context(),
                    previous={"question": q.question, "answer": text, "missed_keywords": missed[:2],
                              "errors_or_bluffs": reconciled["errors_or_bluffs"]})
                followup_probe = probe.question
                pend["followup_probe"] = followup_probe
                self._append_planned(probe, provenance, pend.get("verification_target"), is_followup=True)
            else:
                followup_probe = generate_followup_probe(
                question=q.question, answer=text, missed_keywords=missed[:2],
                skill_id=q.skill_id, llm=self.llm)
            if followup_probe and self.interview_plan is None:
                pend["followup_probe"] = followup_probe
                probe_id = f"{q.id}-followup"
                targeted_kws = [k for k in missed if k.lower() in followup_probe.lower()] or missed[:1]
                self.transcript.append({
                    "q_id": probe_id, "q": followup_probe, "skill_id": q.skill_id,
                    "a": "", "difficulty": nd, "score": None, "next_difficulty": nd,
                    "keywords": targeted_kws,
                    "outline": [f"Explain {k} and the trade-offs involved"
                                for k in targeted_kws[:4]],
                    "is_followup": True,
                })

        return TurnResult(score=score, feedback=feedback, next_difficulty=nd,
                          question_id=q.id, question=q.question, finished=self.finished,
                          rubric=rubric, matched_keywords=matched, missed_keywords=missed,
                          ideal_answer_hint=ideal, injection_flags=inj_flags,
                          followup_probe=followup_probe, error_recovered=error_recovered)

    def finish(self) -> InterviewSummary:
        a = self.answered
        overall = round(sum(t["score"] for t in a) / len(a)) if a else 0
        per = {}
        for t in a:
            per.setdefault(t["skill_id"], []).append(t["score"])
        per_skill = {k: round(sum(v) / len(v)) for k, v in per.items()}
        assessment_per_skill = None
        if self.interview_plan is not None:
            # Never promote from an unavailable evaluator or an unsupported-skill
            # template with no factual reference. Keep those claims pending review.
            assessment_per_skill = {
                skill: value for skill, value in per_skill.items()
                if not any(t.get("evaluation_source") == "offline_fallback" or t.get("assessment_limited")
                           for t in a if t["skill_id"] == skill)}
        rubric_overall = {k: 0 for k in RUBRIC_KEYS}
        scored = [t for t in a if isinstance(t.get("rubric"), dict)]
        if scored:
            for k in RUBRIC_KEYS:
                vals = [max(0, min(100, _coerce_score(t["rubric"].get(k, 0)))) for t in scored]
                rubric_overall[k] = round(sum(vals) / len(vals))
        parsed = None
        try:
            raw_complete = self.llm.complete(
                system=INTERVIEW_FEEDBACK_SYSTEM,
                user=json.dumps({"score": overall, "per_skill": per_skill}), json_mode=True)
            parsed = _safe_json(raw_complete)
        except Exception as exc:
            logger.warning("Interview finish summary generation failed (%s); using fallback", exc)
        feedback = str(parsed.get("feedback") or "").strip() if isinstance(parsed, dict) else ""
        if not feedback:
            strengths = [k for k, v in per_skill.items() if v >= 70]
            gaps = [k for k, v in per_skill.items() if v < 55]
            parts = [f"Completed {len(a)} interview turns with an overall score of {overall}/100."]
            if strengths:
                parts.append(f"Demonstrated strength in {', '.join(strengths)}.")
            if gaps:
                parts.append(f"Revisit and re-verify {', '.join(gaps)}.")
            feedback = " ".join(parts)
        return InterviewSummary(score=overall, per_skill=per_skill, feedback=feedback,
                                transcript=self.transcript, rubric_overall=rubric_overall,
                                coverage=self.coverage, assessment_per_skill=assessment_per_skill)
