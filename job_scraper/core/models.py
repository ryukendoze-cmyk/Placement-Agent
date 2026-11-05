from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RawScrapedJob(BaseModel):
    """Raw record emitted by every portal extractor before normalization."""

    title: str
    company: str
    location: str = "India"
    description: str = ""
    job_url: str
    site: str
    scraper_engine: str = "stealth_harvester"
    date_posted: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    interval: Optional[Literal["lpa", "monthly", "annual", "hourly"]] = None
    is_remote: bool = False
    experience_years: Optional[float] = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("job_url")
    @classmethod
    def _reject_placeholder_urls(cls, v: str) -> str:
        lowered = v.lower()
        if not lowered.startswith(("http://", "https://")):
            raise ValueError(f"job_url must be an absolute http(s) URL, got: {v!r}")
        if "example.com" in lowered or "rapid.example" in lowered:
            raise ValueError(f"placeholder URL is forbidden: {v!r}")
        return v


class PlacementJob(BaseModel):
    """Normalized entity matching ps06-placement-agent's SQLModel Job schema."""

    id: str
    title: str
    company: str
    description: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    location: str = "India"
    remote: bool = False
    salary_lpa: list[int] = Field(default_factory=lambda: [0, 0])
    salary_disclosed: bool = Field(default=False)
    min_gpa: float = 7.0
    allowed_branches: list[str] = Field(default_factory=lambda: ["CSE", "IT", "ECE"])
    job_url: str
    source: str
    scraper_engine: str = "stealth_harvester"

    @field_validator("salary_lpa")
    @classmethod
    def _validate_salary_pair(cls, v: list[int]) -> list[int]:
        if len(v) not in (0, 2):
            raise ValueError(f"salary_lpa must contain 0 or 2 integers, got {v!r}")
        if len(v) == 2 and (v[0] < 0 or v[1] < 0):
            raise ValueError(f"salary_lpa values must be non-negative, got {v!r}")
        return v

    @field_validator("job_url")
    @classmethod
    def _reject_placeholder_urls(cls, v: str) -> str:
        lowered = v.lower()
        if not lowered.startswith(("http://", "https://")):
            raise ValueError(f"job_url must be an absolute http(s) URL, got: {v!r}")
        if "example.com" in lowered or "rapid.example" in lowered:
            raise ValueError(f"placeholder URL is forbidden: {v!r}")
        return v

    def to_placement_dict(self) -> dict[str, Any]:
        return self.model_dump()
