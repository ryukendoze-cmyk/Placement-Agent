"""Separate a closed practice attempt from a completed readiness assessment."""


def is_completed_assessment(interview) -> bool:
    feedback = interview.feedback or {}
    if feedback.get("completion_reason") == "ended_early":
        return False
    coverage = feedback.get("coverage")
    if coverage and (not coverage.get("complete") or any(
            t.get("assessment_limited") for t in coverage.get("targets", []))):
        return False
    if any(t.get("evaluation_source") == "offline_fallback" for t in (interview.transcript or [])):
        return False
    return bool(interview.finished or interview.score is not None)
