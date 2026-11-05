import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.models import Job, Student
from app.scraper.pipeline import ingest_scraped_jobs
from app.scraper.service import CompositeJobScraper
from app.security.ratelimit import check_rate_limit
from app.security.rbac import require_role

router = APIRouter()
logger = logging.getLogger(__name__)

#: SSE response headers (ARCHITECTURE §16.2.B): disable caching and proxy
#: buffering so events reach the client as they are produced.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


class ScrapeRequest(BaseModel):
    search_term: str = "Software Engineer"
    search_terms: list[str] | None = None
    location: str = "Bengaluru"
    sites: list[str] = Field(default_factory=lambda: ["naukri", "linkedin", "indeed"])
    results_wanted: int = Field(default=20, ge=1, le=50)
    hours_old: int = 168
    country_indeed: str = "india"
    providers: list[str] | None = None
    live: bool = True
    fresher_only: bool = True


class IngestOutputRequest(BaseModel):
    file_path: str = "output/normalized_jobs.json"


@router.post("/api/jobs/scrape")
def scrape_jobs_endpoint(
    req: ScrapeRequest | None = None,
    session: Session = Depends(get_session),
    ctx: Any = Depends(get_ctx),
    role: str = Depends(require_role("tpo_admin")),
    _rl: None = Depends(check_rate_limit("jobs_scrape")),
) -> dict[str, Any]:
    """Scrape live Indian job postings and ingest them into database and vector index.

    Live external scraping is strictly restricted to ``tpo_admin`` (§17.2.B.3);
    open demo mode (no declared role) remains permissive for existing tests.
    """
    if req is None:
        req = ScrapeRequest()
    # JobSpy-specific board targeting must not be forwarded to providers that
    # advertise different site sets; non-JobSpy providers ignore `sites` entirely.
    scraper = CompositeJobScraper()
    terms: list[str] = []
    if req.search_terms:
        terms = [t.strip() for t in req.search_terms if t.strip()]
    elif req.search_term:
        if "," in req.search_term:
            terms = [t.strip() for t in req.search_term.split(",") if t.strip()]
        else:
            terms = [req.search_term.strip()]
    if not terms:
        terms = ["Software Engineer"]

    original_terms = list(terms)
    query_terms = list(terms)

    fresher_only = req.fresher_only
    if fresher_only:
        fresher_qualifiers = ["Fresher", "Junior", "Intern", "Entry Level"]
        enriched_terms: list[str] = []
        for t in terms:
            if any(q.lower() in t.lower() for q in ("fresher", "junior", "intern", "entry level", "trainee", "associate")):
                enriched_terms.append(t)
            else:
                for q in fresher_qualifiers:
                    enriched_terms.append(f"{t} {q}")
        query_terms = enriched_terms

    providers = req.providers if req.providers is not None else ["jobspy"]
    raw_records = scraper.fetch_jobs(
        search_term=query_terms if len(query_terms) > 1 else query_terms[0],
        location=req.location,
        sites=req.sites,
        results_wanted=req.results_wanted,
        providers=providers,
        hours_old=req.hours_old,
        country_indeed=req.country_indeed,
        live=req.live,
        fresher_only=fresher_only,
        job_type="entry_level",
    )
    result = ingest_scraped_jobs(raw_records, session, ctx)
    return {
        "status": "success",
        "fetched": len(raw_records),
        "ingested": result["ingested_count"],
        "job_ids": result["job_ids"],
        "terms_used": original_terms,
        "providers_used": [p["name"] for p in scraper.get_available_providers()]
        if req.providers is None
        else list(req.providers),
    }


def _sse_chunk(name: str, data: Any) -> str:
    """Format one Server-Sent Events chunk (ARCHITECTURE §16.2.B wire format)."""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/api/jobs/scrape/stream")
def scrape_jobs_stream_endpoint(
    req: ScrapeRequest | None = None,
    session: Session = Depends(get_session),
    ctx: Any = Depends(get_ctx),
    role: str = Depends(require_role("tpo_admin")),
) -> StreamingResponse:
    """Stream live scrape telemetry as Server-Sent Events (§16.2.B).

    Emits ``init`` -> per-portal ``portal_start`` / ``portal_progress`` /
    ``portal_done`` / ``portal_error`` -> ``complete`` events while
    ``CompositeJobScraper.fetch_jobs_streaming`` executes the selected
    providers, then ingests every discovered job into the database and vector
    index before the final ``complete`` event.

    Live external scraping is strictly restricted to ``tpo_admin`` (§17.2.B.3);
    open demo mode (no declared role) remains permissive for existing tests.
    """
    if req is None:
        req = ScrapeRequest()

    scraper = CompositeJobScraper()
    terms: list[str] = []
    if req.search_terms:
        terms = [t.strip() for t in req.search_terms if t.strip()]
    elif req.search_term:
        if "," in req.search_term:
            terms = [t.strip() for t in req.search_term.split(",") if t.strip()]
        else:
            terms = [req.search_term.strip()]
    if not terms:
        terms = ["Software Engineer"]

    fresher_only = req.fresher_only
    if fresher_only:
        fresher_qualifiers = ["Fresher", "Junior", "Intern", "Entry Level"]
        enriched_terms: list[str] = []
        for t in terms:
            if any(q.lower() in t.lower() for q in ("fresher", "junior", "intern", "entry level", "trainee", "associate")):
                enriched_terms.append(t)
            else:
                for q in fresher_qualifiers:
                    enriched_terms.append(f"{t} {q}")
        terms = enriched_terms

    search_term: str | list[str] = terms if len(terms) > 1 else terms[0]

    providers = req.providers if req.providers is not None else ["jobspy"]

    def event_stream():
        # Real-time event queue: lifecycle events are pushed immediately as they
        # occur and streamed concurrently to the client without blocking.
        event_q: queue.Queue[tuple[str, Any] | None] = queue.Queue()

        def on_event(name: str, data: Any) -> None:
            # We filter out the service's placeholder complete event because
            # this route emits the authoritative complete with real DB ingest counts.
            if name == "complete":
                return
            event_q.put((name, data))

        records: list[dict[str, Any]] = []
        ingest_result: dict[str, Any] = {}
        stream_error: list[str] = []
        started_at = time.perf_counter()

        def run_scrape() -> None:
            try:
                records.extend(
                    scraper.fetch_jobs_streaming(
                        search_term=search_term,
                        location=req.location,
                        sites=req.sites,
                        results_wanted=req.results_wanted,
                        providers=providers,
                        on_event=on_event,
                        hours_old=req.hours_old,
                        country_indeed=req.country_indeed,
                        live=req.live,
                        fresher_only=fresher_only,
                        job_type="entry_level",
                    )
                )
                with Session(ctx.engine) as scrape_session:
                    # §20.2.5 — signal the post-harvest phases so the modal stops
                    # showing "SCRAPING IN PROGRESS" during ingest + ranking.
                    event_q.put((
                        "ingest_start",
                        {"status": "ingesting", "total": len(records)},
                    ))
                    ingest_result.update(ingest_scraped_jobs(records, scrape_session, ctx))
                    try:
                        from app.api.routes.students import _rank_and_persist
                        students = scrape_session.exec(select(Student)).all()
                        event_q.put((
                            "ranking_start",
                            {"status": "ranking", "students": len(students)},
                        ))
                        for st in students:
                            _rank_and_persist(scrape_session, ctx, st.id, st, include_scraped=True)
                    except Exception as match_err:
                        logger.warning("Post-scrape student match generation failed: %s", match_err)
                total_jobs = len(records)
                new_ingested = int(ingest_result.get("ingested_count", 0))
                elapsed_total = round(time.perf_counter() - started_at, 3)
                event_q.put((
                    "complete",
                    {
                        "total_jobs": total_jobs,
                        "new_ingested": new_ingested,
                        "elapsed_total_s": elapsed_total,
                    },
                ))
            except Exception as exc:  # noqa: BLE001 - never kill the stream
                stream_error.append(str(exc))
                event_q.put((
                    "portal_error",
                    {
                        "portal": "stream",
                        "status": "warning",
                        "message": str(exc),
                        "duration_s": 0.0,
                    },
                ))
                event_q.put((
                    "complete",
                    {
                        "total_jobs": 0,
                        "new_ingested": 0,
                        "elapsed_total_s": round(time.perf_counter() - started_at, 3),
                    },
                ))
            finally:
                event_q.put(None)

        worker = threading.Thread(target=run_scrape, daemon=True)
        worker.start()

        while True:
            try:
                item = event_q.get(timeout=0.1)
            except queue.Empty:
                if not worker.is_alive() and event_q.empty():
                    break
                continue

            if item is None:
                break
            name, data = item
            yield _sse_chunk(name, data)

        worker.join(timeout=2.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/api/jobs/scrape/ingest-output")
def ingest_output_endpoint(
    req: IngestOutputRequest | None = None,
    session: Session = Depends(get_session),
    ctx: Any = Depends(get_ctx),
) -> dict[str, Any]:
    """Ingest normalized scraped jobs from output folder JSON into database and vector store."""
    file_path = req.file_path if req and req.file_path else "output/normalized_jobs.json"
    p = Path(file_path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Output file not found: {file_path}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read output file: {exc}")
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Expected JSON array in output file")

    result = ingest_scraped_jobs(data, session, ctx)
    return {
        "status": "success",
        "file": file_path,
        "ingested": result["ingested_count"],
        "job_ids": result["job_ids"],
    }


@router.get("/api/jobs/scrape/providers")
def scrape_providers_endpoint() -> dict[str, Any]:
    """Return metadata for all registered scraper providers (name, display_name, sites)."""
    return {"providers": CompositeJobScraper().get_available_providers()}


@router.get("/api/jobs/scrape/status")
def scrape_status_endpoint(
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return scraper readiness status, supported platforms, and external job count."""
    count_ext = session.exec(
        select(func.count(Job.id)).where(Job.id.like("job-ext-%"))
    ).one()
    defaults = {
        "search_term": "Software Engineer",
        "location": "Bengaluru",
        "results_wanted": 20,
        "hours_old": 168,
    }
    return {
        "status": "ready",
        "supported_platforms": ["naukri", "linkedin", "indeed"],
        "supported_sites": ["naukri", "linkedin", "indeed"],
        "total_external_jobs": count_ext,
        "external_jobs_count": count_ext,
        "job_count": count_ext,
        "defaults": defaults,
        "default_params": defaults,
    }


@router.get("/api/jobs")
def list_jobs(session: Session = Depends(get_session)):
    jobs = session.exec(select(Job)).all()
    return [
        {"id": j.id, "title": j.title, "company": j.company, "location": j.location,
         "remote": j.remote, "experience_years": j.experience_years,
         "required_skills": j.required_skills, "preferred_skills": j.preferred_skills,
         "job_url": getattr(j, "job_url", "") or "",
         "source": getattr(j, "source", "local") or "local",
         "scraper_engine": getattr(j, "scraper_engine", "jobspy") or "jobspy",
         "is_fresher": getattr(j, "is_fresher", True)}
        for j in jobs
    ]


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, session: Session = Depends(get_session)):
    j = session.get(Job, job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    return {"id": j.id, "title": j.title, "company": j.company, "description": j.description,
            "location": j.location, "remote": j.remote, "salary_lpa": j.salary_lpa,
            "experience_years": j.experience_years,
            "required_skills": j.required_skills, "preferred_skills": j.preferred_skills,
            "job_url": getattr(j, "job_url", "") or "",
            "source": getattr(j, "source", "local") or "local",
            "scraper_engine": getattr(j, "scraper_engine", "jobspy") or "jobspy",
            "is_fresher": getattr(j, "is_fresher", True)}
