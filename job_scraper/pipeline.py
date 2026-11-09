from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from job_scraper.core.client import StealthSession
from job_scraper.core.models import RawScrapedJob
from job_scraper.core.normalizer import dedupe_jobs, to_placement_agent_job

logger = logging.getLogger(__name__)

RAW_OUTPUT = "raw_scraped_jobs.json"
NORMALIZED_OUTPUT = "normalized_jobs.json"


@dataclass
class PortalResult:
    portal: str
    jobs: list[RawScrapedJob] = field(default_factory=list)
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _build_registry() -> dict[str, Any]:
    """portal name -> factory(session) -> extractor instance."""
    from job_scraper.extractors.ashby import AshbyExtractor
    from job_scraper.extractors.foundit import FounditExtractor
    from job_scraper.extractors.greenhouse import GreenhouseExtractor
    from job_scraper.extractors.hirist import HiristExtractor
    from job_scraper.extractors.indeed import IndeedExtractor
    from job_scraper.extractors.instahyre import InstahyreExtractor
    from job_scraper.extractors.internshala import InternshalaExtractor
    from job_scraper.extractors.lever import LeverExtractor
    from job_scraper.extractors.linkedin import LinkedInExtractor
    from job_scraper.extractors.naukri import NaukriExtractor
    from job_scraper.extractors.unstop import UnstopExtractor

    return {
        "naukri": lambda s: NaukriExtractor(),
        "linkedin": lambda s: LinkedInExtractor(s),
        "indeed": lambda s: IndeedExtractor(s),
        "foundit": lambda s: FounditExtractor(s),
        "internshala": lambda s: InternshalaExtractor(s),
        "unstop": lambda s: UnstopExtractor(s),
        "instahyre": lambda s: InstahyreExtractor(s),
        "hirist": lambda s: HiristExtractor(s),
        "ashby": lambda s: AshbyExtractor(s),
        "greenhouse": lambda s: GreenhouseExtractor(s),
        "lever": lambda s: LeverExtractor(s),
    }


# Overridable by tests.
EXTRACTOR_REGISTRY: dict[str, Any] = _build_registry()

# Inter-portal politeness gap (seconds).
PORTAL_GAP_SECONDS = 0.5


def _normalize_raw_jobs(raw_jobs: list[RawScrapedJob]) -> tuple[list[dict], list[dict]]:
    """Dedupe raw jobs, then map survivors to placement-ready dicts."""
    raw_dicts = [j.model_dump() for j in raw_jobs]
    deduped = dedupe_jobs(raw_dicts)
    normalized: list[dict[str, Any]] = []
    for d in deduped:
        try:
            normalized.append(to_placement_agent_job(d))
        except Exception as exc:
            logger.warning("normalization failed for %s: %s", d.get("job_url"), exc)
    return raw_dicts, normalized


async def run_pipeline(
    keywords: list[str] | str,
    location: str = "India",
    days_old: int = 7,
    limit: int = 20,
    portals: Optional[list[str]] = None,
    output_dir: Path | str = "output",
    progress: Optional[Any] = None,
) -> dict[str, Any]:
    """Run every portal extractor (error-contained) and write output files.

    Returns a summary dict:
      {
        "raw": [raw dumps], "normalized": [placement dicts],
        "results": [PortalResult, ...],
        "counts": {"raw": n, "normalized": n, "deduped_away": n,
                   "portals_ok": n, "portals_failed": n, "keywords": n},
      }
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords = [k.strip() for k in keywords if k and k.strip()]
    if not keywords:
        raise ValueError("at least one keyword is required")

    registry = dict(EXTRACTOR_REGISTRY)
    if portals:
        registry = {k: v for k, v in registry.items() if k in portals}
        missing = set(portals) - set(registry)
        if missing:
            raise ValueError(f"unknown portals: {sorted(missing)}")

    collected: list[RawScrapedJob] = []
    results: list[PortalResult] = []

    async with StealthSession(min_interval=1.5, max_retries=3) as session:
        for portal, factory in registry.items():
            result = PortalResult(portal=portal)
            t0 = time.monotonic()
            try:
                extractor = factory(session)
                for kw in keywords:
                    try:
                        jobs = await asyncio.wait_for(
                            extractor.search(
                                keyword=kw,
                                location=location,
                                freshness_days=days_old,
                                limit=limit,
                            ),
                            timeout=10.0,
                        )
                        result.jobs.extend(jobs)
                    except Exception as exc:
                        logger.warning(
                            "%s keyword %r failed: %s", portal, kw, exc, exc_info=True
                        )
                        if result.error is None:
                            result.error = f"{type(exc).__name__}: {exc}"
                result.jobs = result.jobs[:limit]
                result.duration_s = time.monotonic() - t0
            except Exception as exc:
                result.duration_s = time.monotonic() - t0
                result.error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "portal %s failed entirely: %s", portal, exc, exc_info=True
                )
            collected.extend(result.jobs)
            results.append(result)
            if progress:
                try:
                    progress(portal, result)
                except Exception:
                    pass
            await asyncio.sleep(PORTAL_GAP_SECONDS)

    raw_dicts, normalized = _normalize_raw_jobs(collected)
    counts = {
        "raw": len(collected),
        "normalized": len(normalized),
        "deduped_away": len(collected) - len(normalized),
        "portals_ok": sum(1 for r in results if r.ok),
        "portals_failed": sum(1 for r in results if not r.ok),
        "keywords": len(keywords),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_OUTPUT
    norm_path = out_dir / NORMALIZED_OUTPUT
    if raw_dicts or not raw_path.exists():
        raw_path.write_text(json.dumps(raw_dicts, indent=2, ensure_ascii=False), encoding="utf-8")
    if normalized or not norm_path.exists():
        norm_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "wrote %d raw / %d normalized jobs to %s",
        len(raw_dicts), len(normalized), out_dir,
    )

    return {
        "raw": raw_dicts,
        "normalized": normalized,
        "results": results,
        "counts": counts,
        "outputs": {"raw": str(raw_path), "normalized": str(norm_path)},
    }
