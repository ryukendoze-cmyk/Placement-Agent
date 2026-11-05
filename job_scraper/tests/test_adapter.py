import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestAdapterProvider:
    def test_provider_inherits_base_and_has_name(self):
        from job_scraper.adapter.placement_agent import (
            AdvancedStealthScraperProvider,
            BaseJobScraperProvider,
        )

        provider = AdvancedStealthScraperProvider()
        assert isinstance(provider, BaseJobScraperProvider)
        assert provider.name == "advanced_stealth_harvester"

    async def test_scrape_jobs_uses_pipeline(self, monkeypatch, tmp_path):
        from job_scraper import pipeline
        from job_scraper.adapter.placement_agent import AdvancedStealthScraperProvider

        captured = {}

        async def fake_run_pipeline(**kw):
            captured.update(kw)
            return {
                "normalized": [{"id": "job-ext-abc", "title": "X"}],
                "raw": [],
                "results": [],
                "counts": {},
                "outputs": {},
            }

        monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
        provider = AdvancedStealthScraperProvider()
        jobs = await provider.scrape_jobs(
            ["Python"], location="Pune", days_old=3, limit=5, output_dir=str(tmp_path)
        )
        assert jobs == [{"id": "job-ext-abc", "title": "X"}]
        assert captured["keywords"] == ["Python"]
        assert captured["location"] == "Pune"
        assert captured["days_old"] == 3
        assert captured["limit"] == 5

    def test_sync_wrapper_runs_event_loop(self, monkeypatch, tmp_path):
        from job_scraper import pipeline
        from job_scraper.adapter.placement_agent import AdvancedStealthScraperProvider

        async def fake_run_pipeline(**kw):
            return {"normalized": [{"id": "job-ext-xyz"}]}

        monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
        provider = AdvancedStealthScraperProvider()
        jobs = provider.scrape_jobs_sync(["python"], output_dir=str(tmp_path))
        assert jobs == [{"id": "job-ext-xyz"}]

    async def test_sync_wrapper_safe_from_running_loop(self, monkeypatch, tmp_path):
        """scrape_jobs_sync must not crash when called from a running loop
        (FastAPI/Celery host contexts)."""
        from job_scraper import pipeline
        from job_scraper.adapter.placement_agent import AdvancedStealthScraperProvider

        async def fake_run_pipeline(**kw):
            return {"normalized": [{"id": "job-ext-loop"}]}

        monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
        provider = AdvancedStealthScraperProvider()
        # We are inside a running event loop here (async test) — must not raise.
        jobs = provider.scrape_jobs_sync(["python"], output_dir=str(tmp_path))
        assert jobs == [{"id": "job-ext-loop"}]
