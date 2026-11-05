"""Voice & acoustic communication telemetry (Milestone 18.3).

Pure analysis of a spoken/typed interview answer: pacing (WPM), filler-word
rate, STAR structure detection, a composite delivery score, and one concise
coaching tip. No I/O, fully deterministic, safe to call on every turn.

Optimal pacing band is 110-160 WPM. When ``duration_seconds`` is missing (typed
answers) ``wpm`` is 0 and pacing contributes neutral credit so the score is
driven by STAR structure and filler discipline instead.
"""
from __future__ import annotations

import re

#: Multi-word first, then single-word fillers (boundary-anchored to avoid
#: false positives like "likely" / "dislike").
_FILLER_PATTERNS = [
    (re.compile(r"\byou\s+know\b", re.I), "you know"),
    (re.compile(r"\bsort\s+of\b", re.I), "sort of"),
    (re.compile(r"\bum\b", re.I), "um"),
    (re.compile(r"\buh\b", re.I), "uh"),
    (re.compile(r"\ber\b", re.I), "er"),
    (re.compile(r"\blike\b", re.I), "like"),
    (re.compile(r"\bbasically\b", re.I), "basically"),
    (re.compile(r"\bactually\b", re.I), "actually"),
]

_SIT_PAT = re.compile(
    r"\b(situation|context|background|scenario|when|faced|challenge|at my|during)\b", re.I)
_TASK_PAT = re.compile(
    r"\b(task|objective|goal|responsible|needed to|had to|aimed|supposed to)\b", re.I)
_ACT_PAT = re.compile(
    r"\b(action|implemented|built|developed|designed|introduced|refactored|deployed|"
    r"created|fixed|optimized|migrated|led|i (wrote|ran|set up|configured))\b", re.I)
_RES_PAT = re.compile(
    r"\b(result|outcome|impact|metric|reduced|increased|improved|percent|%|ms|throughput|"
    r"latency|saved|doubled|grew|achieved|delivered)\b", re.I)

_OPTIMAL_WPM = (110, 160)


def analyze_speech_delivery(text: str, duration_seconds: float | None = None) -> dict:
    """Compute pacing, filler, STAR and delivery telemetry for one answer.

    Returns a dict with ``word_count``, ``wpm``, ``filler_words``,
    ``filler_rate_pct``, ``star_components``, ``star_score``, ``delivery_score``
    and ``coaching_tip``.
    """
    raw = str(text or "")
    words = raw.split()
    word_count = len(words)

    # --- pacing ------------------------------------------------------------
    wpm = 0
    duration = None
    try:
        if duration_seconds is not None:
            duration = float(duration_seconds)
    except (TypeError, ValueError):
        duration = None
    if duration and duration > 0 and word_count > 0:
        wpm = round(word_count / duration * 60)

    # --- filler words ------------------------------------------------------
    filler_words: list[str] = []
    for pattern, label in _FILLER_PATTERNS:
        filler_words.extend([label] * len(pattern.findall(raw)))
    filler_count = len(filler_words)
    filler_rate_pct = round(100.0 * filler_count / max(1, word_count), 1)

    # --- STAR structure ----------------------------------------------------
    star_components = {
        "situation": bool(_SIT_PAT.search(raw)),
        "task": bool(_TASK_PAT.search(raw)),
        "action": bool(_ACT_PAT.search(raw)),
        "result": bool(_RES_PAT.search(raw)),
    }
    star_score = sum(25 for v in star_components.values() if v)

    # --- delivery score ----------------------------------------------------
    # pacing (30): in-band = full, out-of-band = partial, unknown duration = neutral
    if wpm == 0:
        pacing = 25
    elif _OPTIMAL_WPM[0] <= wpm <= _OPTIMAL_WPM[1]:
        pacing = 30
    else:
        # distance from the band, clamped
        if wpm < _OPTIMAL_WPM[0]:
            dist = _OPTIMAL_WPM[0] - wpm
        else:
            dist = wpm - _OPTIMAL_WPM[1]
        pacing = max(0, 30 - int(dist / 10))
    # filler discipline (30): full credit at 0%, none at >=15%
    filler_credit = max(0, round(30 * (1 - min(filler_rate_pct, 15.0) / 15.0)))
    # STAR (40): proportional to components present
    star_credit = round(40 * star_score / 100)
    delivery_score = max(0, min(100, pacing + filler_credit + star_credit))

    # --- coaching tip (priority: fillers > pacing > STAR > praise) ---------
    coaching_tip = _coaching_tip(filler_words, filler_rate_pct, wpm, star_components, word_count)

    return {
        "word_count": word_count,
        "wpm": wpm,
        "filler_words": filler_words,
        "filler_rate_pct": filler_rate_pct,
        "star_components": star_components,
        "star_score": star_score,
        "delivery_score": delivery_score,
        "coaching_tip": coaching_tip,
    }


def _coaching_tip(filler_words, filler_rate_pct, wpm, star_components, word_count) -> str:
    if filler_count := len(filler_words):
        top = sorted(set(filler_words))
        preview = ", ".join(f"'{f}'" for f in top[:3])
        return (f"Reduce filler words ({preview}) - you used {filler_count} "
                f"({filler_rate_pct}%) to sound more authoritative.")
    if wpm and wpm > _OPTIMAL_WPM[1]:
        return f"Slow down - you spoke at {wpm} WPM; aim for {_OPTIMAL_WPM[0]}-{_OPTIMAL_WPM[1]} WPM."
    if wpm and wpm < _OPTIMAL_WPM[0]:
        return f"Pick up the pace - {wpm} WPM is below the {_OPTIMAL_WPM[0]} WPM floor; add energy."
    missing = [k for k, v in star_components.items() if not v]
    if missing and word_count >= 8:
        label = missing[0].capitalize()
        return (f"Quantify the {label} of your STAR answer (add the metric or concrete "
                f"outcome you achieved).")
    if not star_components["result"]:
        return "Quantify the outcome (Result) with latency or throughput metrics."
    return "Strong structure - keep pairing each answer with a measurable result."


def aggregate_speech_delivery(transcript: list) -> dict:
    """Aggregate per-turn ``speech_analytics`` into a session ``speech_summary``.

    Pure helper (M18.3): used by the live finalize endpoint AND the demo-cohort
    seeder (M25) so seeded transcripts carry the same telemetry shape.
    """
    turns = [t.get("speech_analytics") for t in (transcript or [])
             if isinstance(t, dict) and isinstance(t.get("speech_analytics"), dict)]
    if not turns:
        return {}
    wpms = [t.get("wpm", 0) for t in turns if t.get("wpm")]
    stars = [t.get("star_score", 0) for t in turns]
    deliveries = [t.get("delivery_score", 0) for t in turns]
    fillers = sum(len(t.get("filler_words") or []) for t in turns)
    return {
        "avg_wpm": round(sum(wpms) / len(wpms)) if wpms else 0,
        "total_fillers": fillers,
        "avg_star_score": round(sum(stars) / len(stars)) if stars else 0,
        "avg_delivery_score": round(sum(deliveries) / len(deliveries)) if deliveries else 0,
        "turns_measured": len(turns),
    }
