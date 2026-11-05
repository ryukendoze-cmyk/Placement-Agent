import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from job_scraper.core.models import RawScrapedJob  # noqa: E402
from job_scraper.pipeline import PortalResult, run_pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def no_portal_gap(monkeypatch):
    import job_scraper.pipeline as pipeline

    monkeypatch.setattr(pipeline, "PORTAL_GAP_SECONDS", 0.0)


def make_raw(site: str, url: str, **kw) -> RawScrapedJob:
    base = dict(
        title="Python Developer",
        company="Acme Corp",
        location="Bengaluru",
        description="Build things with Python, FastAPI, PostgreSQL, Docker, AWS.",
        job_url=url,
        site=site,
    )
    base.update(kw)
    return RawScrapedJob(**base)


class FakeExtractor:
    def __init__(self, session=None, jobs=None, error=None, portal="fake", site="fake"):
        self.session = session
        self.jobs = jobs or []
        self.error = error
        self.portal_name = portal
        self.site = site
        self.calls = []

    async def search(self, keyword, location="India", freshness_days=7, limit=50):
        self.calls.append((keyword, location, freshness_days, limit))
        if self.error:
            raise self.error
        return list(self.jobs[:limit])


def build_registry(monkeypatch, **extractors):
    from job_scraper import pipeline

    reg = {k: (lambda s, j=v, e=None, p=k: FakeExtractor(s, jobs=j, error=e, portal=p))
           for k, v in extractors.items()}
    monkeypatch.setattr(pipeline, "EXTRACTOR_REGISTRY", reg)
    return reg


class TestRunPipeline:
    async def test_collects_jobs_from_all_portals(self, tmp_path, monkeypatch):
        a = make_raw("alpha", "https://a.jobs/j/1")
        b = make_raw("beta", "https://b.jobs/j/2", title="Data Engineer")
        build_registry(monkeypatch,
                       alpha=[a],
                       beta=[b])
        out = await run_pipeline(
            keywords=["python"], location="Bengaluru", days_old=7, limit=10,
            output_dir=tmp_path,
        )
        raws = out["raw"]
        assert {r["site"] for r in raws} == {"alpha", "beta"}
        assert len(out["normalized"]) == 2

    async def test_error_in_one_portal_does_not_halt_others(self, tmp_path, monkeypatch):
        a = make_raw("alpha", "https://a.jobs/j/1")
        build_registry(monkeypatch,
                       alpha=[a],
                       beta=None,
                       gamma=None)
        import job_scraper.pipeline as pipeline

        def bad_factory(session, j=None, e=None, p="beta"):
            return FakeExtractor(session, jobs=[], error=RuntimeError(f"{p} exploded"))

        monkeypatch.setattr(pipeline, "EXTRACTOR_REGISTRY", {
            "alpha": lambda s: FakeExtractor(s, jobs=[a], portal="alpha", site="alpha"),
            "beta": bad_factory,
            "gamma": lambda s: FakeExtractor(s, jobs=[make_raw("gamma", "https://c.jobs/j/9", title="QA")], portal="gamma", site="gamma"),
        })
        out = await run_pipeline(keywords=["python"], limit=10, output_dir=tmp_path)
        assert {r["site"] for r in out["raw"]} == {"alpha", "gamma"}
        errors = {r.portal: r.error for r in out["results"] if r.error}
        assert "beta" in errors
        assert "exploded" in errors["beta"]

    async def test_cross_portal_dedupe_by_canonical_url(self, tmp_path, monkeypatch):
        a = make_raw("alpha", "https://a.jobs/j/1?utm_source=x")
        a2 = make_raw("beta", "https://a.jobs/j/1?ref=y",
                      description="A much longer description for the duplicate record with extra detail about the role and the team and the stack involved")
        build_registry(monkeypatch, alpha=[a], beta=[a2])
        out = await run_pipeline(keywords=["python"], limit=10, output_dir=tmp_path)
        assert len(out["normalized"]) == 1
        # Longer description wins.
        assert "much longer" in out["normalized"][0]["description"]

    async def test_writes_output_files(self, tmp_path, monkeypatch):
        a = make_raw("alpha", "https://a.jobs/j/1")
        build_registry(monkeypatch, alpha=[a])
        await run_pipeline(keywords=["python"], limit=10, output_dir=tmp_path)
        raw_file = tmp_path / "raw_scraped_jobs.json"
        norm_file = tmp_path / "normalized_jobs.json"
        assert raw_file.exists() and norm_file.exists()
        raw = json.loads(raw_file.read_text(encoding="utf-8"))
        norm = json.loads(norm_file.read_text(encoding="utf-8"))
        assert raw[0]["job_url"] == "https://a.jobs/j/1"
        assert norm[0]["id"].startswith("job-ext-")
        assert norm[0]["salary_lpa"] == [0, 0]
        assert norm[0]["salary_disclosed"] is False
        assert norm[0]["source"] == "alpha"
        assert "CSE" in norm[0]["allowed_branches"]

    async def test_keyword_list_runs_each_keyword(self, tmp_path, monkeypatch):
        a = make_raw("alpha", "https://a.jobs/j/1")
        build_registry(monkeypatch, alpha=[a])
        out = await run_pipeline(keywords=["python", "fastapi"], limit=10, output_dir=tmp_path)
        # Same URL for both keywords -> deduped to 1, but both keywords were queried.
        assert len(out["normalized"]) == 1
        assert out["counts"]["keywords"] == 2

    async def test_normalized_salary_is_integer_pair(self, tmp_path, monkeypatch):
        a = make_raw("alpha", "https://a.jobs/j/1", min_amount=10.0, max_amount=18.0, interval="lpa")
        build_registry(monkeypatch, alpha=[a])
        out = await run_pipeline(keywords=["python"], limit=10, output_dir=tmp_path)
        assert out["normalized"][0]["salary_lpa"] == [10, 18]
        assert all(isinstance(x, int) for x in out["normalized"][0]["salary_lpa"])

    async def test_limit_respected_per_portal(self, tmp_path, monkeypatch):
        jobs = [make_raw("alpha", f"https://a.jobs/j/{i}", title=f"Role {i}")
                for i in range(10)]
        build_registry(monkeypatch, alpha=jobs)
        out = await run_pipeline(keywords=["python"], limit=4, output_dir=tmp_path)
        assert len(out["normalized"]) <= 4

    async def test_limit_caps_total_across_keywords(self, tmp_path, monkeypatch):
        """Per-portal limit must bound the TOTAL, not per-keyword, so two
        keywords can't double the requested cap."""
        import job_scraper.pipeline as pipeline

        class MultiKwFake:
            portal_name = "alpha"
            site = "alpha"

            async def search(self, keyword, location="India", freshness_days=7, limit=50):
                # Each keyword yields its own distinct jobs (no cross-kw dedupe).
                return [
                    make_raw("alpha", f"https://a.jobs/{keyword}/{i}", title=f"{keyword} {i}")
                    for i in range(limit)
                ]

        monkeypatch.setattr(pipeline, "EXTRACTOR_REGISTRY",
                            {"alpha": lambda s: MultiKwFake()})
        out = await run_pipeline(keywords=["python", "fastapi"], limit=4, output_dir=tmp_path)
        alpha_jobs = [r for r in out["raw"] if r["site"] == "alpha"]
        assert len(alpha_jobs) <= 4

    def test_portal_result_shape(self):
        r = PortalResult(portal="x", jobs=[], duration_s=1.0, error=None)
        assert r.error is None
        r2 = PortalResult(portal="y", jobs=[], duration_s=0.5, error="boom")
        assert r2.error == "boom"
