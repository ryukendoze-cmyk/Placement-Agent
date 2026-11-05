"""Counterfactual "what-if" matching simulator. Pure layer (no FastAPI/SQLModel).

Answers: "if this student learns these missing skills, how much does their match
for a specific job improve, what does it cost in study hours, and does it unlock a
higher readiness band?"
"""
from ..taxonomy.taxonomy import Taxonomy
from .engine import higher_level, job_tier, level_rank, score_job

BAND_RANK = {"early": 0, "developing": 1, "moderate": 2, "strong": 3}


def simulate_counterfactual(
    student_skills: dict[str, list[str]],
    job,
    added_skill_ids: list[str],
    sim: float,
    taxonomy: Taxonomy,
    *,
    levels: dict[str, str] | None = None,
    courses_for_skill: dict[str, dict] | None = None,
) -> dict:
    """Project the (student, job) match if the student acquires `added_skill_ids`,
    assumed to reach at least the 'intermediate' level (never downgrading an
    existing higher level). `courses_for_skill` maps each added skill to its best
    course dict so study hours are summed over *distinct* courses only."""
    levels = levels or {}
    added = list(dict.fromkeys(added_skill_ids or []))  # dedupe, keep order

    current = score_job(student_skills, job, sim, taxonomy, levels=levels)

    proj_skills = dict(student_skills)
    proj_levels = dict(levels)
    for s in added:
        proj_skills.setdefault(s, ["(simulated acquisition)"])
        proj_levels[s] = higher_level(levels.get(s), "intermediate")

    projected = score_job(proj_skills, job, sim, taxonomy, levels=proj_levels)
    delta = projected.score - current.score

    job_relevant = set(job.required_skills) | set(job.preferred_skills)
    required = set(job.required_skills)
    impacted = [
        s for s in added
        if (s in job_relevant and s not in student_skills)
        or (s in required and level_rank(levels.get(s)) < level_rank("intermediate"))
    ]
    neutral = [s for s in added if s not in impacted]

    by_course = {c["id"]: c for c in (courses_for_skill or {}).values() if c}
    total_hours = sum(c.get("hours", 0) for c in by_course.values())
    unlocked = BAND_RANK.get(projected.band, 0) > BAND_RANK.get(current.band, 0)

    return {
        "job_id": job.id,
        "current_score": current.score,
        "current_band": current.band,
        "current_tier": job_tier(job),
        "projected_score": projected.score,
        "projected_band": projected.band,
        "delta": delta,
        "total_hours": total_hours,
        "course_ids": sorted(by_course),
        "added_skills": added,
        "impacted_skills": impacted,
        "neutral_skills": neutral,
        "unlocked_higher_band": unlocked,
        "missing_required_after": [d["skill"] for d in projected.missing_required],
    }
