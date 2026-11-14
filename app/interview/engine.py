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
    "You are a technical interviewer. Evaluate the candidate's answer against the interview question "
    "and expected keywords. Respond ONLY with a valid JSON object containing keys: "
    "\"score\" (integer from 0 to 100), \"feedback\" (concise 1-2 sentence constructive feedback string), "
    "\"rubric\" (object with keys \"technical_accuracy\", \"depth_and_examples\", \"communication_clarity\", "
    "each an integer from 0 to 100), \"matched_keywords\" (list of strings), "
    "\"missed_keywords\" (list of strings), and \"ideal_answer_hint\" (string, one sentence describing a strong answer)."
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


def _deterministic_evaluate(q: Question, answer_text: str) -> dict:
    """Offline fallback evaluator when the configured LLM fails or emits invalid JSON."""
    text = (answer_text or "").strip()
    keywords = [str(k).strip() for k in (q.keywords or ()) if str(k).strip()]
    low = text.lower()
    matched = [k for k in keywords if re.search(rf"\b{re.escape(k.lower())}", low)]
    missed = [k for k in keywords if k not in matched]

    if _is_non_answer(text):
        return {
            "score": 15,
            "feedback": "No technical explanation provided. Review core fundamentals and outline a concrete example.",
            "rubric": {
                "technical_accuracy": 10,
                "depth_and_examples": 10,
                "communication_clarity": 25,
            },
            "matched_keywords": matched,
            "missed_keywords": missed,
            "ideal_answer_hint": "; ".join(q.outline[:2]) if q.outline else "",
        }

    words = re.findall(r"\b\w+\b", low)
    kw_ratio = len(matched) / max(1, len(keywords)) if keywords else 0.5
    depth_signals = sum(
        1
        for tok in ("because", "example", "for instance", "trade-off", "tradeoff", "latency", "throughput", "index", "cache", "test", "scale")
        if tok in low
    )
    tech = min(92, max(20, round(25 + 65 * kw_ratio)))
    depth = min(90, max(20, round(20 + 35 * kw_ratio + min(30, len(words) // 2) + min(15, depth_signals * 5))))
    clarity = min(92, max(30, round(35 + min(35, len(words)) + (15 if len(words) >= 12 else 0))))
    score = round((tech * 0.45) + (depth * 0.35) + (clarity * 0.20))
    hint = "; ".join(q.outline[:2]) if q.outline else (
        f"Cover {', '.join(missed[:3])} with a concrete implementation example." if missed else ""
    )
    if matched and missed:
        fb = f"Covered {', '.join(matched[:3])}. Deepen your answer on {', '.join(missed[:3])}."
    elif matched:
        fb = f"Solid coverage of {', '.join(matched[:3])} with clear reasoning."
    else:
        fb = f"Answer lacked key concepts ({', '.join(missed[:3])}). Focus on concrete mechanisms and trade-offs."
    return {
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


class InterviewSession:
    def __init__(self, role: str, bank, llm, prefer_skills, transcript,
                 enable_followups: bool = False, resume_questions=None):
        self.role = role
        self.bank = bank
        self.llm = llm
        self.prefer_skills = list(prefer_skills or [])
        self.transcript = [dict(t) for t in (transcript or [])]
        #: M18.1: opt-in adaptive follow-up probing (default off -> legacy id sequences).
        self.enable_followups = bool(enable_followups)
        #: M18.2: resume-grounded probes served before static bank questions.
        self.resume_questions = list(resume_questions or [])

    @property
    def answered(self):
        return [t for t in self.transcript if t.get("score") is not None]

    @property
    def turn(self):
        return len(self.answered)

    @property
    def finished(self):
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
        if not (pend.get("is_followup") or pend.get("is_resume_probe")):
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
        parsed = None
        error_recovered = ""
        try:
            raw_complete = self.llm.complete(system=EVALUATE_ANSWER_SYSTEM, user=payload, json_mode=True)
            parsed = _safe_json(raw_complete)
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
                raw_retry = self.llm.complete(system=EVALUATE_ANSWER_SYSTEM, user=repair_prompt, json_mode=True)
                parsed = _safe_json(raw_retry)
                if not isinstance(parsed, dict):
                    error_recovered = "invalid_json_fallback"
            except Exception as exc:
                logger.warning("Interview answer JSON-repair retry failed (%s)", exc)
                error_recovered = f"llm_retry_error:{type(exc).__name__}"

        if not isinstance(parsed, dict):
            parsed = _deterministic_evaluate(q, text)
            if not error_recovered:
                error_recovered = "deterministic_rubric_fallback"

        score = _coerce_score(parsed.get("score"))
        feedback = str(parsed.get("feedback") or "")
        raw_rub = parsed.get("rubric") if isinstance(parsed.get("rubric"), dict) else {}
        rubric = {k: max(0, min(100, _coerce_score(raw_rub.get(k, score)))) for k in RUBRIC_KEYS}
        matched = [str(x) for x in (parsed.get("matched_keywords") or [])]
        missed = [str(x) for x in (parsed.get("missed_keywords") or [])]
        ideal = str(parsed.get("ideal_answer_hint") or "")
        score = max(0, min(100, score))
        d = pend["difficulty"]
        if score >= 70:
            nd = min(2, d + 1)
        elif score < 40:
            nd = max(0, d - 1)
        else:
            nd = d
        pend["a"] = text
        pend["score"] = score
        pend["next_difficulty"] = nd
        pend["rubric"] = rubric
        pend["matched_keywords"] = matched
        pend["missed_keywords"] = missed
        pend["ideal_answer_hint"] = ideal
        if error_recovered:
            pend["error_recovered"] = error_recovered

        # --- M18.1: adaptive follow-up probing (opt-in) ----------------------
        followup_probe = ""
        if (self.enable_followups
                and not pend.get("is_followup")
                and self.turn < MAX_TURNS
                and missed
                and not _is_non_answer(text)
                and score >= 25
                and ((35 <= score < 75)
                     or rubric.get("depth_and_examples", 100) < 60)):
            followup_probe = generate_followup_probe(
                question=q.question, answer=text, missed_keywords=missed[:2],
                skill_id=q.skill_id, llm=self.llm)
            if followup_probe:
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
                                transcript=self.transcript, rubric_overall=rubric_overall)
