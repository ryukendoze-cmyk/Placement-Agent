"""Question telemetry and adaptive feedback loop manager.

Tracks per-question telemetry (times asked, average score, keyword hit ratio)
and detects outliers (questions that are too easy, too hard, or have misaligned keywords).
"""
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_TELEMETRY_PATH = DATA_DIR / "question_telemetry.json"

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _get_path_lock(path: Path) -> threading.RLock:
    """Return a shared re-entrant lock keyed by resolved telemetry file path."""
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _default_runtime_telemetry_path() -> Path:
    """Isolate pytest runs from mutating repository data/question_telemetry.json."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return Path(tempfile.gettempdir()) / f"jklu_question_telemetry_{os.getpid()}.json"
    return DEFAULT_TELEMETRY_PATH


class QuestionTelemetryManager:
    """Manages telemetry recording and feedback loop analysis for interview questions."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else _default_runtime_telemetry_path()
        self._lock = _get_path_lock(self.path)
        self._ledger: dict[str, dict[str, Any]] = {}
        self.load_telemetry()

    def record_answer_turn(
        self,
        q_id: str,
        score: int | float,
        matched_kw: list[str] | None = None,
        expected_kw: list[str] | None = None,
        difficulty: int = 1,
        matched_keywords: list[str] | None = None,
        expected_keywords: list[str] | None = None,
    ) -> None:
        """Record an answer turn and update running telemetry statistics."""
        matched = matched_kw if matched_kw is not None else (matched_keywords or [])
        expected = expected_kw if expected_kw is not None else (expected_keywords or [])

        with self._lock:
            if self.path.exists():
                self.load_telemetry()
            rec = self._ledger.setdefault(
                q_id,
                {
                    "q_id": q_id,
                    "id": q_id,
                    "times_asked": 0,
                    "total_score": 0.0,
                    "avg_score": 0.0,
                    "total_keyword_hit_ratio": 0.0,
                    "avg_keyword_hit_ratio": 0.0,
                    "last_difficulty": difficulty,
                },
            )
            rec["times_asked"] += 1
            rec["total_score"] += float(score)
            rec["avg_score"] = round(rec["total_score"] / rec["times_asked"], 1)

            ratio = len(matched) / max(1, len(expected))
            rec["total_keyword_hit_ratio"] += ratio
            rec["avg_keyword_hit_ratio"] = round(
                rec["total_keyword_hit_ratio"] / rec["times_asked"], 3
            )
            rec["last_difficulty"] = difficulty
            self.persist_telemetry()

    def analyze_feedback_loop(self) -> dict[str, Any]:
        """Analyze question bank performance and identify outlier questions."""
        with self._lock:
            snapshot = {k: dict(v) for k, v in self._ledger.items()}
        flagged_too_easy = []
        flagged_too_hard = []
        flagged_kw_mismatch = []
        recommendations = []

        for q_id, rec in snapshot.items():
            times = rec.get("times_asked", 0)
            if times < 3:
                continue

            avg_s = rec.get("avg_score", 0.0)
            ratio = rec.get("avg_keyword_hit_ratio", 0.0)
            diff = rec.get("last_difficulty", 1)

            if avg_s >= 90.0:
                item = {
                    "q_id": q_id,
                    "id": q_id,
                    "times_asked": times,
                    "avg_score": avg_s,
                    "difficulty": diff,
                    "reason": f"High average score ({avg_s} >= 90.0 across {times} attempts)",
                }
                flagged_too_easy.append(item)
                recommendations.append(
                    f"Upgrade difficulty or revise question '{q_id}' (avg score: {avg_s} across {times} sessions)."
                )

            if avg_s <= 35.0:
                item = {
                    "q_id": q_id,
                    "id": q_id,
                    "times_asked": times,
                    "avg_score": avg_s,
                    "difficulty": diff,
                    "reason": f"Low average score ({avg_s} <= 35.0 across {times} attempts)",
                }
                flagged_too_hard.append(item)
                recommendations.append(
                    f"Lower difficulty or provide hints for question '{q_id}' (avg score: {avg_s} across {times} sessions)."
                )

            if ratio < 0.25:
                item = {
                    "q_id": q_id,
                    "id": q_id,
                    "times_asked": times,
                    "avg_keyword_hit_ratio": ratio,
                    "reason": f"Low keyword match ratio ({ratio:.2f} < 0.25 across {times} attempts)",
                }
                flagged_kw_mismatch.append(item)
                recommendations.append(
                    f"Review expected keywords for question '{q_id}' (hit ratio: {ratio:.2f} < 0.25 across {times} sessions)."
                )

        return {
            "total_tracked": len(self._ledger),
            "flagged_too_easy": flagged_too_easy,
            "flagged_too_hard": flagged_too_hard,
            "flagged_keyword_mismatch": flagged_kw_mismatch,
            "recommendations": recommendations,
        }

    def apply_calibration(self) -> dict[str, Any]:
        """Persist concrete difficulty/keyword calibration suggestions on flagged questions."""
        with self._lock:
            if self.path.exists():
                self.load_telemetry()
            analysis = self.analyze_feedback_loop()
            updates: list[dict[str, Any]] = []
            now = datetime.now(timezone.utc).isoformat()

            for item in analysis["flagged_too_easy"]:
                qid = item["q_id"]
                rec = self._ledger.get(qid)
                if not rec:
                    continue
                cur = int(rec.get("calibrated_difficulty", rec.get("last_difficulty", 1)))
                nxt = min(2, cur + 1)
                rec["calibrated_difficulty"] = nxt
                rec["calibration_action"] = "raise_difficulty"
                rec["calibrated_at"] = now
                updates.append({"q_id": qid, "action": "raise_difficulty", "from": cur, "to": nxt})

            for item in analysis["flagged_too_hard"]:
                qid = item["q_id"]
                rec = self._ledger.get(qid)
                if not rec:
                    continue
                cur = int(rec.get("calibrated_difficulty", rec.get("last_difficulty", 1)))
                nxt = max(0, cur - 1)
                rec["calibrated_difficulty"] = nxt
                rec["calibration_action"] = "lower_difficulty"
                rec["calibrated_at"] = now
                updates.append({"q_id": qid, "action": "lower_difficulty", "from": cur, "to": nxt})

            for item in analysis["flagged_keyword_mismatch"]:
                qid = item["q_id"]
                rec = self._ledger.get(qid)
                if not rec:
                    continue
                rec["keyword_review_flagged"] = True
                rec["calibrated_at"] = now
                updates.append({"q_id": qid, "action": "flag_keyword_review"})

            if updates:
                self.persist_telemetry()
            return {"calibrated_count": len(updates), "updates": updates, "analysis": analysis}

    def persist_telemetry(self, path: Path | str | None = None) -> None:
        """Persist in-memory telemetry to JSON file atomically."""
        target_path = Path(path) if path else self.path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = json.dumps(self._ledger, indent=2)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target_path.name}.",
                suffix=".tmp",
                dir=str(target_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_name, target_path)
            except BaseException:
                if os.path.exists(tmp_name):
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                raise

    def load_telemetry(self, path: Path | str | None = None) -> dict[str, Any]:
        """Load telemetry from JSON file."""
        target_path = Path(path) if path else self.path
        with self._lock:
            if not target_path.exists():
                self._ledger = {}
                return self._ledger
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._ledger = data
                    else:
                        self._ledger = {}
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load telemetry from %s: %s", target_path, e)
                self._ledger = {}
            return self._ledger
