"""Matching engine and seniority classification for JKLU CareerOS."""

from app.matching.engine import (
    FRESHER_TITLE_RE,
    SENIOR_TITLE_RE,
    is_campus_fresher,
    is_fresher_job,
    is_senior_job,
)

__all__ = [
    "FRESHER_TITLE_RE",
    "SENIOR_TITLE_RE",
    "is_campus_fresher",
    "is_fresher_job",
    "is_senior_job",
]
