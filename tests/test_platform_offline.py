"""Platform P0 offline-readiness & P1 responsive checks."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from unittest.mock import patch

TEMPLATES_DIR = Path("app/templates")
VENDOR_DIR = Path("app/static/vendor")

_EXTERNAL_TAG_RE = re.compile(
    r"""(?:<script\b[^>]*\bsrc\s*=\s*["']https?://|<link\b[^>]*\bhref\s*=\s*["']https?://|@import\s+(?:url\()?["']?https?://|url\(\s*["']?https?://)""",
    re.IGNORECASE,
)


def test_templates_have_zero_external_asset_urls():
    """No .html template under app/templates may reference http(s):// for scripts, links, or fonts."""
    offenders: dict[str, list[str]] = {}
    for html_path in sorted(TEMPLATES_DIR.rglob("*.html")):
        text = html_path.read_text(encoding="utf-8")
        hits = [m.group(0) for m in _EXTERNAL_TAG_RE.finditer(text)]
        if hits:
            offenders[str(html_path)] = hits
    assert offenders == {}, f"External asset URLs found in templates: {offenders}"


def test_vendored_assets_and_fonts_exist_without_external_urls():
    for fname in ("tailwindcss.js", "htmx.min.js", "chart.umd.min.js", "fonts.css"):
        p = VENDOR_DIR / fname
        assert p.is_file() and p.stat().st_size > 0, f"Missing vendored asset: {p}"

    woff2_files = list((VENDOR_DIR / "fonts").glob("*.woff2"))
    assert len(woff2_files) >= 3, "Expected vendored .woff2 font files in app/static/vendor/fonts/"

    fonts_css = (VENDOR_DIR / "fonts.css").read_text(encoding="utf-8")
    assert "http://" not in fonts_css and "https://" not in fonts_css, (
        "fonts.css must not contain external http(s):// URLs"
    )
    assert "/static/vendor/fonts/" in fonts_css


def test_vendored_and_pwa_routes_served(client):
    for path in (
        "/static/vendor/tailwindcss.js",
        "/static/vendor/htmx.min.js",
        "/static/vendor/chart.umd.min.js",
        "/static/vendor/fonts.css",
        "/static/responsive.css",
        "/manifest.webmanifest",
        "/sw.js",
    ):
        r = client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"


def test_base_html_includes_responsive_css_and_pwa():
    base = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
    assert "/static/responsive.css" in base
    assert 'rel="manifest" href="/manifest.webmanifest"' in base
    assert "serviceWorker" in base
    assert "/sw.js" in base


def test_responsive_css_mobile_drawer_and_stacking():
    css = Path("app/static/responsive.css").read_text(encoding="utf-8")
    assert "@media (max-width: 768px)" in css
    assert "#sidebar" in css
    assert "#master-list-column" in css
    assert "#cards-container" in css
    assert "44px" in css


def test_local_embedder_offline_fallback_and_warning(monkeypatch, caplog):
    """When SentenceTransformer fails offline, LocalEmbedder falls back to HashingEmbedder(384)."""
    from app.embeddings.embedder import HashingEmbedder, LocalEmbedder

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with patch("sentence_transformers.SentenceTransformer", side_effect=OSError("offline cache miss")):
        with caplog.at_level(logging.WARNING):
            emb = LocalEmbedder()
    assert emb.dim == 384
    assert isinstance(emb._fallback, HashingEmbedder)
    vec = emb.embed("Python FastAPI Docker")
    assert len(vec) == 384
    assert any("HashingEmbedder" in rec.message for rec in caplog.records)


def test_cli_warm_cache_returns_zero_offline(monkeypatch):
    from app import cli

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("no network")):
        code = cli.main(["warm-cache"])
    assert code == 0
