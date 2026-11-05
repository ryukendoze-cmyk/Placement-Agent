"""Main orchestrator entrypoint.

Usage:
    python main.py                                # defaults: python / India / 7d / 20
    python main.py -k "python, fastapi" -l Bengaluru -d 3 -n 20
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_scraper.pipeline import run_pipeline  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="India job scraper orchestrator")
    p.add_argument("-k", "--keywords", default="python",
                   help="Comma-separated search terms")
    p.add_argument("-l", "--location", default="India", help="City or 'India'/'Remote'")
    p.add_argument("-d", "--days-old", type=int, default=7, help="Freshness window (days)")
    p.add_argument("-n", "--limit", type=int, default=20, help="Max jobs per portal")
    p.add_argument("-p", "--portals", default="", help="Comma-separated portal subset")
    p.add_argument("-o", "--output", default="output", help="Output directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    portal_list = [x.strip() for x in args.portals.split(",") if x.strip()] or None

    out = asyncio.run(
        run_pipeline(
            keywords=keywords,
            location=args.location,
            days_old=args.days_old,
            limit=args.limit,
            portals=portal_list,
            output_dir=args.output,
        )
    )
    c = out["counts"]
    print(f"raw={c['raw']} normalized={c['normalized']} deduped_away={c['deduped_away']}"
          f" portals_ok={c['portals_ok']} portals_failed={c['portals_failed']}")
    for r in out["results"]:
        flag = "OK " if r.ok else "ERR"
        err = f" ({r.error})" if r.error else ""
        print(f"  [{flag}] {r.portal:<12} {len(r.jobs):>3} jobs {r.duration_s:6.1f}s{err}")
    print(f"raw output:        {out['outputs']['raw']}")
    print(f"normalized output: {out['outputs']['normalized']}")
    return 0 if c["normalized"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
