"""Public resume-extraction API re-exports (module-path compatibility).

Both ``app.skills.extract`` and this module expose the same extraction
contracts; import from either path.
"""
from ..skills.extract import (
    EXTRACT_RESUME_SYSTEM,
    extract_resume_with_ai,
    extract_skills,
    extract_skills_with_llm,
    level_for,
)

__all__ = [
    "EXTRACT_RESUME_SYSTEM",
    "extract_resume_with_ai",
    "extract_skills",
    "extract_skills_with_llm",
    "level_for",
]
