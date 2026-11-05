"""Live Indian job scraping service and normalization pipeline for JKLU CareerOS."""

from app.scraper.ai_query import suggest_job_queries_for_student
from app.scraper.normalizer import (
    EXP_PLUS_SINGLE_RE,
    EXP_RANGE_RE,
    FRESHER_TITLE_RE,
    JOB_NORMALIZER_SYSTEM,
    SENIOR_TITLE_RE,
    extract_experience_years,
    is_fresher_job,
    is_senior_job,
    normalize_job_with_ai,
    normalize_scraped_job,
    parse_salary_lpa,
)
from app.scraper.pipeline import (
    ingest_from_output_folder,
    ingest_scraped_jobs,
    purge_dummy_jobs,
    purge_live_jobs,
)
from app.scraper.providers import (
    AutomationJobScraperProvider,
    BaseJobScraperProvider,
    JobSpyProvider,
    ProxyAPIJobScraperProvider,
    RapidJobScraperProvider,
    StealthHarvesterProvider,
)
from app.scraper.service import CompositeJobScraper, JobSpyScraper

__all__ = [
    "AutomationJobScraperProvider",
    "BaseJobScraperProvider",
    "CompositeJobScraper",
    "JobSpyProvider",
    "JobSpyScraper",
    "ProxyAPIJobScraperProvider",
    "RapidJobScraperProvider",
    "StealthHarvesterProvider",
    "ingest_from_output_folder",
    "ingest_scraped_jobs",
    "purge_dummy_jobs",
    "purge_live_jobs",
    "normalize_scraped_job",
    "normalize_job_with_ai",
    "JOB_NORMALIZER_SYSTEM",
    "suggest_job_queries_for_student",
    "EXP_PLUS_SINGLE_RE",
    "EXP_RANGE_RE",
    "FRESHER_TITLE_RE",
    "SENIOR_TITLE_RE",
    "extract_experience_years",
    "is_fresher_job",
    "is_senior_job",
    "parse_salary_lpa",
]
