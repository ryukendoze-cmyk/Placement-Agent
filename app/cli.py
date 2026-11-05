"""Command-line entry points for the PS-06 placement agent.

    python -m app.cli seed                          # init_db + seed DATABASE_URL (Postgres + pgvector)
    python -m app.cli demo --file r.pdf --name "Bharat" --top 5

``seed`` opens the Postgres dev DB from ``.env`` (DATABASE_URL), creates the
tables and inserts taxonomy/jobs/courses; on Postgres it also embeds every
job description into the pgvector table (384-dim only, matching EmbeddingRow).

``demo`` never touches the database: it reads jobs straight from
``data/jobs.json`` via ``seed._json`` so the ranked table renders even on a
machine with no database running (and without seeding first). Resume and job
vectors are embedded on the fly with the same embedder the seed uses, keeping
cosine semantics consistent with the seed/API paths.

Test seams (monkeypatched in tests/test_cli.py): :func:`_make_engine`
(sqlite instead of Postgres), :func:`_make_embedder` (hashing instead of the
sentence-transformers model) and :func:`_make_store` (no pgvector on sqlite).
:func:`main` returns an int instead of exiting so tests can call
``main([...])``; argparse's SystemExit is caught and returned as its code
(``bogus`` command -> 2).
"""

import argparse
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from .config import settings
from .db import init_db, make_engine
from .embeddings import HashingEmbedder, LocalEmbedder, PgVectorStore
from .matching.rank import rank_jobs
from .models import Course, Job, Skill
from .resume.parser import parse_resume_bytes
from .scraper.pipeline import ingest_scraped_jobs, purge_live_jobs
from .scraper.service import JobSpyScraper
from .seed import _json, seed, seed_demo_cohort
from .skills.extract import extract_skills
from .taxonomy.taxonomy import load_taxonomy

JOB_REQUIRED_FIELDS = ("id", "title", "company", "description")
JOB_OPTIONAL_FIELDS = {
    "required_skills": [],
    "preferred_skills": [],
    "experience_years": 0,
    "location": "",
    "remote": False,
    "salary_lpa": [0, 0],
    "min_gpa": 0.0,
    "allowed_branches": [],
}


def _make_engine(url: str):
    """Seam: tests replace this with a sqlite engine (no Postgres needed)."""
    return make_engine(url)


def _make_embedder():
    """Seam: tests replace this with ``lambda: HashingEmbedder()`` (no model load)."""
    if settings.embedder == "local":
        return LocalEmbedder()
    return HashingEmbedder()


def _make_store(db: Session, database_url: str):
    """Seam: pgvector store on Postgres, None on sqlite (seed accepts None)."""
    if (database_url or "").startswith("postgres"):
        return PgVectorStore(db, dim=settings.embedding_dim)
    return None


def _load_jobs() -> list[SimpleNamespace]:
    """Jobs from data/jobs.json (not the DB) - demo must work without seeding."""
    return [
        SimpleNamespace(**{k: j[k] for k in JOB_REQUIRED_FIELDS},
                        **{k: j.get(k, v) for k, v in JOB_OPTIONAL_FIELDS.items()})
        for j in _json("jobs.json")
    ]


def cmd_seed(demo_cohort: bool = False) -> int:
    if (settings.database_url or "").startswith("postgres") and settings.embedding_dim != 384:
        print("error: pgvector store requires EMBEDDING_DIM=384 "
              f"(got {settings.embedding_dim})", file=sys.stderr)
        return 1
    try:
        engine = _make_engine(settings.database_url)
        init_db(engine)
        embedder = _make_embedder()
        with Session(engine) as db:
            store = _make_store(db, settings.database_url)
            taxonomy = load_taxonomy()
            seed(db, taxonomy, vector_store=store, embedder=embedder)
            n_skills = len(db.exec(select(Skill)).all())
            n_jobs = len(db.exec(select(Job)).all())
            n_courses = len(db.exec(select(Course)).all())
            n_vectors = 0
            if store is not None:
                n_vectors = sum(1 for j in db.exec(select(Job)).all() if store.get("job", j.id) is not None)
            n_cohort = 0
            if demo_cohort:
                n_cohort = seed_demo_cohort(db, taxonomy=taxonomy, embedder=embedder, vector_store=store)
        print(f"seeded: {n_skills} skills, {n_jobs} jobs, {n_courses} courses")
        if demo_cohort:
            print(f"demo cohort: {n_cohort} students")
        if store is not None:
            print(f"job vectors stored: {n_vectors}")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, SQLAlchemyError) as e:
        print(f"error: seed failed: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_demo(file: str, name: str, top: int) -> int:
    path = Path(file)
    try:
        data = path.read_bytes()
    except OSError as e:
        print(f"error: could not read resume file {file}: {e}", file=sys.stderr)
        return 1
    try:
        text = parse_resume_bytes(data, path.suffix.lower())
    except (ValueError, zipfile.BadZipFile, OSError, RuntimeError) as e:   # RuntimeError: pymupdf corrupt/empty PDF errors
        print(f"error: could not parse resume {file}: {e}", file=sys.stderr)
        return 1
    if not text.strip():
        print(f"error: resume {file} has no readable text", file=sys.stderr)
        return 1
    try:
        taxonomy = load_taxonomy()
        student_skills = extract_skills(text, taxonomy)
        embedder = _make_embedder()
        student_vec = embedder.embed(text)
        jobs = _load_jobs()
        ranked = [js for _, js in rank_jobs(
            jobs, student_vec, student_skills, embedder, taxonomy, top=max(top, 0),
        )]
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as e:
        print(f"error: demo failed: {e}", file=sys.stderr)
        return 1
    jobs_by_id = {j.id: j for j in jobs}
    print(f"matches for {name}" if name else "matches")
    if not ranked:
        print("(no jobs to rank)")
        return 0
    _print_table(ranked, jobs_by_id)
    return 0


def _print_table(ranked, jobs_by_id) -> None:
    cols = ("rank", "score", "band", "job_id", "title", "missing required")
    rows = []
    for i, js in enumerate(ranked, start=1):
        missing = ", ".join(m["name"] for m in js.missing_required) or "-"
        rows.append([
            str(i),
            str(js.score),
            js.band,
            js.job_id,
            jobs_by_id.get(js.job_id).title if js.job_id in jobs_by_id else "-",
            missing,
        ])
    widths = [
        max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
        for i, c in enumerate(cols)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    for r in rows:
        print(fmt.format(*r))


def cmd_scrape_jobs(
    term: str = "Software Engineer",
    location: str = "Bengaluru",
    limit: int = 20,
    sites: str | list[str] = "naukri linkedin indeed",
) -> int:
    """Scrape live Indian tech jobs and ingest them into the database and vector store."""
    try:
        if isinstance(sites, str):
            site_list = sites.split()
        elif isinstance(sites, list):
            site_list = sites
        else:
            site_list = ["naukri", "linkedin", "indeed"]

        engine = _make_engine(settings.database_url)
        init_db(engine)
        embedder = _make_embedder()
        taxonomy = load_taxonomy()
        scraper = JobSpyScraper()
        raw_jobs = scraper.fetch_jobs(
            search_term=term,
            location=location,
            sites=site_list,
            results_wanted=limit,
        )

        with Session(engine) as session:
            store = _make_store(session, settings.database_url)
            ctx = SimpleNamespace(
                engine=engine,
                taxonomy=taxonomy,
                vector_store=store,
                embedder=embedder,
                job_vectors={},
            )
            res = ingest_scraped_jobs(raw_jobs, session, ctx, taxonomy=taxonomy)
            jobs = res["jobs"]

        print(f"Scraped and ingested {len(jobs)} jobs (term: '{term}', location: '{location}')")
        if not jobs:
            print("(no jobs found)")
            return 0

        cols = ("Job ID", "Title", "Company", "Location", "Salary LPA", "Grounded Skills")
        rows = [
            [
                j.id,
                j.title,
                j.company,
                j.location or "-",
                f"{j.salary_lpa[0]}-{j.salary_lpa[1]}" if j.salary_lpa else "-",
                ", ".join(j.required_skills[:3]) if j.required_skills else "-",
            ]
            for j in jobs
        ]
        widths = [
            max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
            for i, c in enumerate(cols)
        ]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*cols))
        for r in rows:
            print(fmt.format(*r))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, SQLAlchemyError) as e:
        print(f"error: scrape-jobs failed: {e}", file=sys.stderr)
        return 1


def cmd_ingest_output(file_path: str = "output/normalized_jobs.json") -> int:
    path = Path(file_path)
    if not path.is_file():
        print(f"error: file not found: {file_path}", file=sys.stderr)
        return 1
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"error: failed to read JSON from {file_path}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(data, list):
        print(f"error: expected JSON array in {file_path}", file=sys.stderr)
        return 1

    try:
        engine = _make_engine(settings.database_url)
        init_db(engine)
        embedder = _make_embedder()
        taxonomy = load_taxonomy()
        with Session(engine) as session:
            store = _make_store(session, settings.database_url)
            ctx = SimpleNamespace(
                engine=engine,
                taxonomy=taxonomy,
                vector_store=store,
                embedder=embedder,
                job_vectors={},
            )
            res = ingest_scraped_jobs(data, session, ctx, taxonomy=taxonomy)
            jobs = res["jobs"]

        print(f"Ingested {len(jobs)} jobs from '{file_path}'")
        if not jobs:
            print("(no jobs ingested)")
            return 0

        cols = ("Job ID", "Title", "Company", "Location", "Engine", "Source")
        rows = [
            [
                j.id,
                j.title,
                j.company,
                j.location or "-",
                getattr(j, "scraper_engine", "-") or "-",
                getattr(j, "source", "-") or "-",
            ]
            for j in jobs
        ]
        widths = [
            max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
            for i, c in enumerate(cols)
        ]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*cols))
        for r in rows:
            print(fmt.format(*r))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, SQLAlchemyError) as e:
        print(f"error: ingest-output failed: {e}", file=sys.stderr)
        return 1


def cmd_purge_live_jobs(keep_output: bool = False) -> int:
    try:
        engine = _make_engine(settings.database_url)
        init_db(engine)
        embedder = _make_embedder()
        with Session(engine) as session:
            store = _make_store(session, settings.database_url)
            ctx = SimpleNamespace(
                engine=engine,
                vector_store=store,
                embedder=embedder,
                job_vectors={},
            )
            count = purge_live_jobs(session, ctx=ctx, clear_output=not keep_output)
        print(f"Purged {count} live job(s) from database and reset output caches")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, SQLAlchemyError) as e:
        print(f"error: purge-live-jobs failed: {e}", file=sys.stderr)
        return 1


def cmd_eval_benchmark() -> int:
    try:
        from .embeddings.store import MemoryVectorStore
        from .eval.benchmark import run_evaluation_benchmark

        engine = _make_engine(settings.database_url)
        init_db(engine)
        embedder = _make_embedder()
        with Session(engine) as session:
            store = _make_store(session, settings.database_url)
            if store is None:
                store = MemoryVectorStore()
            ctx = SimpleNamespace(
                engine=engine,
                taxonomy=load_taxonomy(),
                vector_store=store,
                embedder=embedder,
                job_vectors={},
            )
            bench = run_evaluation_benchmark(session, ctx)
        print("PS-06 Evaluation Benchmark (M23)")
        print(f"  SOP RAG:   recall@1={bench['sop_rag']['recall_at_1']:.2f} "
              f"recall@3={bench['sop_rag']['recall_at_3']:.2f} "
              f"MRR={bench['sop_rag']['mean_reciprocal_rank']:.3f} "
              f"grounding={bench['sop_rag']['citation_grounding_score']}")
        d = bench["matching"]["depth_monotonicity"]
        fs = bench["matching"]["fresher_senior"]
        print(f"  Matching:  senior_eligible={fs['senior_eligible']} "
              f"fresher_eligible={fs['fresher_eligible']} "
              f"depth {d['basic']}<{d['intermediate']}<{d['advanced']}={d['basic'] < d['intermediate'] < d['advanced']}")
        print(f"  Planner:   profiles={bench['planner_dag']['profiles']} "
              f"violations={bench['planner_dag']['violations']} "
              f"cycles={bench['planner_dag']['cycles']}")
        print(f"  Grounding: interception={bench['agent_grounding']['interception_rate']:.0%}")
        print(f"  all_passed={bench['summary']['all_passed']}")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, SQLAlchemyError) as e:
        print(f"error: eval-benchmark failed: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli", description="PS-06 placement agent CLI")
    sub = parser.add_subparsers(dest="command")
    s = sub.add_parser("seed", help="create tables and seed taxonomy/jobs/courses (+ job vectors on Postgres)")
    s.add_argument("--demo-cohort", action="store_true", help="also seed a 12-student demo cohort with matches, plans, progress, interviews, agent traces + 4 pending approvals")
    d = sub.add_parser("demo", help="rank jobs against a resume file (no database needed)")
    d.add_argument("--file", required=True, help="resume file (.pdf, .docx, .txt, .png, .jpg, .jpeg)")
    d.add_argument("--name", default="", help="candidate name for the header")
    d.add_argument("--top", type=int, default=5, help="number of ranked jobs to print (default 5)")
    sc = sub.add_parser("scrape-jobs", help="scrape live Indian tech jobs and ingest into database")
    sc.add_argument("--term", default="Software Engineer", help="search keyword or role (default: Software Engineer)")
    sc.add_argument("--location", default="Bengaluru", help="job location (default: Bengaluru)")
    sc.add_argument("--limit", type=int, default=20, help="maximum results to scrape (default: 20)")
    sc.add_argument("--sites", default="naukri linkedin indeed", help="job boards (space separated)")
    io = sub.add_parser("ingest-output", help="ingest normalized scraped jobs JSON into database and vector store")
    io.add_argument("--file", default="output/normalized_jobs.json", help="path to normalized JSON (default: output/normalized_jobs.json)")
    isj = sub.add_parser("ingest-scraped-jobs", help="alias for ingest-output")
    isj.add_argument("--file", default="output/normalized_jobs.json", help="path to normalized JSON (default: output/normalized_jobs.json)")
    plj = sub.add_parser("purge-live-jobs", help="purge live scraped jobs, related records, embeddings, and output caches")
    plj.add_argument("--keep-output", action="store_true", default=False, help="preserve output JSON caches instead of clearing them")
    sub.add_parser("eval-benchmark", help="run the automated RAG/matching/planner/grounding evaluation benchmark")
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if args.command == "seed":
        return cmd_seed(demo_cohort=args.demo_cohort)
    if args.command == "demo":
        return cmd_demo(args.file, args.name, args.top)
    if args.command == "scrape-jobs":
        return cmd_scrape_jobs(args.term, args.location, args.limit, args.sites)
    if args.command in ("ingest-output", "ingest-scraped-jobs"):
        return cmd_ingest_output(args.file)
    if args.command == "purge-live-jobs":
        return cmd_purge_live_jobs(keep_output=args.keep_output)
    if args.command == "eval-benchmark":
        return cmd_eval_benchmark()
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
