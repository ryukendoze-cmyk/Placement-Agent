"""Skill-verification micro-quiz (Phase 5).

Serves 2 short-answer questions per plan item from data/questions.json
(fallback cascade: skill_id exact -> _PARENT_MAP canonical parent -> "generic").
Grading is deterministic keyword matching against the question's `keywords` -
the same evidence contract as the interview evaluator (no LLM, offline-safe).

GET never serializes the grading key (`keywords`) or any `explanation`; POST
grades server-side only and, on pass (>=50% of questions meeting the
keyword threshold), triggers the shared closed-loop completion path.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# canonical parent for a skill when the exact skill has no quiz questions
_PARENT_MAP = {
    "fastapi": "python", "django": "python", "flask": "python", "pandas": "python",
    "numpy": "python", "pytest": "python", "celery": "python",
    "postgresql": "sql", "mysql": "sql", "sqlite": "sql", "dbt": "sql",
    "sql_analytics": "sql", "bigquery": "sql", "etl": "sql", "data_modeling": "sql",
    "css": "html", "javascript_web": "html", "typescript": "javascript_web",
    "react": "javascript_web", "nextjs": "react", "vue": "javascript_web",
    "angular": "typescript", "tailwind": "css", "bootstrap": "css",
    "docker": "linux", "kubernetes": "docker", "bash": "linux", "nginx": "linux",
    "ci_cd": "git", "github": "git", "terraform": "aws", "ansible": "linux",
    "monitoring": "linux",
    "deep_learning": "machine_learning", "pytorch": "deep_learning",
    "tensorflow": "deep_learning", "nlp": "deep_learning", "llms": "nlp",
    "computer_vision": "deep_learning",
    "machine_learning": "statistics", "sklearn": "statistics",
    "ab_testing": "statistics",
    "test_automation": "unittest", "playwright": "test_automation",
    "selenium": "test_automation", "jest": "test_automation",
    "rest_api": "python",
}

PASSING_SCORE = 50          # percent of questions graded correct
QUESTIONS_PER_QUIZ = 2
HIT_RATIO = 0.5             # a question is correct when >=50% of its keywords appear
EXPLANATION_TEMPLATES = {
    "git": "Answer should cover the core git concepts referenced by the prompt.",
    "default": "A strong answer demonstrates mastery of core concepts and "
               "principles with concrete reasoning.",
}


@lru_cache(maxsize=1)
def _load_quiz_bank() -> tuple[dict, ...]:
    raw = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
    return tuple(raw)


def questions_for_skill(skill_id: str) -> list[dict]:
    """All bank questions matching skill_id (exact, then parent, then generic)."""
    bank = _load_quiz_bank()
    exact = [q for q in bank if q.get("skill_id") == skill_id]
    if exact:
        return exact
    parent = _PARENT_MAP.get(skill_id)
    if parent:
        parent_qs = [q for q in bank if q.get("skill_id") == parent]
        if parent_qs:
            return parent_qs
    return [q for q in bank if q.get("skill_id") == "generic"]


def generate_quiz(skill_id: str, n: int = QUESTIONS_PER_QUIZ) -> list[dict]:
    """Deterministically pick n questions (stable by difficulty then id)."""
    pool = questions_for_skill(skill_id)
    if not pool:
        return []
    pool = sorted(pool, key=lambda q: (q.get("difficulty", 0), q.get("id", "")))
    picked: list[dict] = []
    i = 0
    while len(picked) < n and pool:
        picked.append(pool[i % len(pool)])
        i += 1
        if i >= len(pool):  # exhausted pool without hitting n
            break
    return picked


def _answer_key(skill_id: str) -> dict[str, list[str]]:
    """{question_id: keywords} for grading - never leaves the server on GET."""
    return {q["id"]: list(q.get("keywords", [])) for q in generate_quiz(skill_id)}


def _answers_for_item(item_id: str, skill_id: str) -> dict[str, str]:
    """A passing answer per question (for tests/seeded demos): the keywords
    joined into one short-answer string that always meets the hit threshold."""
    return {qid: " ".join(kws) for qid, kws in _answer_key(skill_id).items()}


def _grade_question(answer: str, keywords: list[str]) -> tuple[bool, float]:
    """Deterministic keyword hit ratio >= HIT_RATIO -> correct."""
    if not keywords:
        return True, 1.0
    text = (answer or "").lower()
    hits = sum(1 for k in keywords if str(k).lower() in text)
    ratio = hits / len(keywords)
    return ratio >= HIT_RATIO, ratio


def check_answers(skill_id: str, answers: dict) -> tuple[bool, int, list[dict]]:
    """Grade submitted answers. Returns (passed, score, feedback_rows).

    `answers` maps question_id -> free-text answer.
    """
    quiz = generate_quiz(skill_id)
    if not quiz:
        return True, 100, []  # no questions = vacuously verified
    correct = 0
    rows = []
    for q in quiz:
        qid = q["id"]
        keywords = list(q.get("keywords", []))
        is_correct, ratio = _grade_question(str(answers.get(qid, "")), keywords)
        if is_correct:
            correct += 1
        rows.append({
            "question_id": qid,
            "correct": is_correct,
            "hit_ratio": round(ratio, 2),
            "explanation": EXPLANATION_TEMPLATES.get(skill_id, EXPLANATION_TEMPLATES["default"]),
        })
    total = len(quiz)
    score = round(100 * correct / total)
    passed = score >= PASSING_SCORE
    return passed, score, rows
