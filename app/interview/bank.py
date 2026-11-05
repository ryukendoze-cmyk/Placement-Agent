"""Question bank: load data/questions.json and select the next question deterministically."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Question:
    id: str
    role: str
    skill_id: str
    difficulty: int
    question: str
    keywords: tuple[str, ...]
    outline: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def load_bank() -> tuple[Question, ...]:
    import json
    raw = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
    return tuple(Question(id=q["id"], role=q["role"], skill_id=q["skill_id"],
                          difficulty=q["difficulty"], question=q["question"],
                          keywords=tuple(q["keywords"]),
                          outline=tuple(q.get("outline") or ())) for q in raw)


def get_question(bank, q_id: str):
    for q in bank:
        if q.id == q_id:
            return q
    return None


def select(bank, role: str, difficulty: int, exclude_ids, prefer_skills) -> "Question | None":
    """Pick the best non-excluded question.

    Deterministic total order (lower wins): same role > generic > other role;
    closest difficulty; preferred skill > generic > other skill; then id (stable).
    Difficulty and role are relaxed (never None while questions remain).
    """
    excl = set(exclude_ids or [])
    prefer = set(prefer_skills or [])
    best = None
    best_key = None
    for q in bank:
        if q.id in excl:
            continue
        role_score = 0 if q.role == role else (1 if q.role == "generic" else 2)
        diff_dist = abs(q.difficulty - difficulty)
        if prefer:
            pref_match = 0 if q.skill_id in prefer else 1
            key = (role_score, pref_match, diff_dist, q.id)
        else:
            key = (role_score, diff_dist, 1 if q.skill_id == "generic" else 2, q.id)
        if best_key is None or key < best_key:
            best_key = key
            best = q
    return best
