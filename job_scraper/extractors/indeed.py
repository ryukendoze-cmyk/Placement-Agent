from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional
from urllib.parse import quote_plus

from job_scraper.core.exceptions import BotDetectionError  # noqa: E402
from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

BASE = "https://in.indeed.com"
SERP = f"{BASE}/jobs"

_MOSAIC_ASSIGN = 'window.mosaic.providerData["mosaic-provider-jobcards"]'
# Markers indicating a Cloudflare managed challenge / Turnstile page.
_CF_MARKERS = (
    "security check",
    "indeed_cloudflare_static_page",
    "_cf_chl_opt",
    "challenge-running",
    "challenge-platform",
    "cf-chl",
)

# In-page wait condition for the mosaic payload (Tier B).
_MOSAIC_READY_JS = (
    '() => Boolean(window.mosaic && window.mosaic.providerData &&'
    ' window.mosaic.providerData["mosaic-provider-jobcards"])'
)

# Flexible salary extraction regex supporting Western and Indian comma grouping (ARCHITECTURE §20.2.1).
SALARY_RANGE_RE = re.compile(
    r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)\s*(?:to|-|–)\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
SALARY_SINGLE_RE = re.compile(
    r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _balanced_json(text: str, start: int) -> str:
    """Extract a balanced {...} JSON object starting at text[start] (a '{')."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def extract_mosaic_json(html: str) -> Optional[dict[str, Any]]:
    """Pull the job-cards mosaic payload out of the Indeed SERP HTML."""
    if not html:
        return None
    idx = html.find(_MOSAIC_ASSIGN)
    if idx == -1:
        return None
    eq = html.find("=", idx)
    brace = html.find("{", eq)
    if brace == -1:
        return None
    blob = _balanced_json(html, brace)
    if not blob:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError as exc:
        logger.warning("indeed mosaic JSON decode error: %s", exc)
        return None


def parse_indeed_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    model = ((payload.get("metaData") or {}).get("mosaicProviderJobCardsModel")) or {}
    return model.get("results") or []


def extract_indeed_detail(html: str) -> Optional[str]:
    """Extract full job description text from Indeed viewjob HTML."""
    if not html:
        return None
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        desc = soup.find(id="jobDescriptionText") or soup.find(class_="jobsearch-jobDescriptionText")
        if desc:
            return BaseExtractor.html_to_text(str(desc))
    except Exception as exc:
        logger.debug("extract_indeed_detail error: %s", exc)
    return None


def parse_indeed_result(r: dict[str, Any]) -> Optional[RawScrapedJob]:
    jk = r.get("jobkey") or ""
    if not jk:
        return None
    url = f"{BASE}/viewjob?jk={jk}"

    snippet = r.get("snippet") or ""
    salary = r.get("salarySnippet") or ""
    if isinstance(salary, dict):
        salary = " ".join(str(v) for v in salary.values() if isinstance(v, str))
    elif not isinstance(salary, str):
        salary = ""
    min_sal = max_sal = None
    interval = None
    m = SALARY_RANGE_RE.search(salary)
    if m:
        min_sal = float(m.group(1).replace(",", ""))
        max_sal = float(m.group(2).replace(",", ""))
        interval = "annual" if (max_sal or 0) >= 100000 else "monthly"
    elif salary:
        m2 = SALARY_SINGLE_RE.search(salary)
        if m2:
            min_sal = max_sal = float(m2.group(1).replace(",", ""))
            interval = "annual" if min_sal >= 100000 else "monthly"

    loc = (r.get("formattedLocation") or "").strip()
    remote = bool(r.get("remoteLocation")) or "remote" in loc.lower()

    # Synthesize rich description from all card details and requirements
    desc_parts = []
    snippet_clean = BaseExtractor.html_to_text(snippet)
    if snippet_clean:
        desc_parts.append(snippet_clean)

    req_model = r.get("jobCardRequirementsModel") or {}
    job_reqs = req_model.get("jobOnlyRequirements") or []
    if job_reqs:
        req_lines = [str(x.get("text") or x) for x in job_reqs if x]
        if req_lines:
            desc_parts.append("Requirements:\n" + "\n".join(f"- {line}" for line in req_lines))

    match_summary = r.get("jobSeekerMatchSummaryModel") or {}
    entities = (
        (match_summary.get("sortedMisMatchingEntityDisplayText") or [])
        + (match_summary.get("sortedMatchingEntityDisplayText") or [])
        + (match_summary.get("sortedEntityDisplayText") or [])
    )
    seen_ent = set()
    unique_entities = []
    for e in entities:
        if isinstance(e, str) and e.strip() and e.lower() not in seen_ent:
            seen_ent.add(e.lower())
            unique_entities.append(e.strip())

    if unique_entities:
        desc_parts.append("Key Competencies & Qualifications:\n" + ", ".join(unique_entities))

    taxo_attrs = r.get("taxonomyAttributes") or []
    attr_labels = []
    for group in taxo_attrs:
        for attr in group.get("attributes") or []:
            lbl = attr.get("label")
            if lbl:
                attr_labels.append(lbl)
    if attr_labels:
        desc_parts.append("Job Attributes: " + ", ".join(attr_labels))

    description = "\n\n".join(desc_parts) if desc_parts else snippet_clean

    return BaseExtractor.make_job(
        title=(r.get("displayTitle") or r.get("normTitle") or "").strip(),
        company=(r.get("company") or "").strip(),
        location=loc or "India",
        description=description,
        job_url=url,
        date_posted=r.get("createDate") or r.get("formattedRelativeTime"),
        min_amount=min_sal,
        max_amount=max_sal,
        interval=interval,
        is_remote=remote,
        tags=unique_entities[:15],
        site="indeed",
    )


def parse_indeed_html(html: str) -> list[RawScrapedJob]:
    payload = extract_mosaic_json(html)
    if not payload:
        return []
    jobs = []
    for r in parse_indeed_results(payload):
        try:
            job = parse_indeed_result(r)
            if job:
                jobs.append(job)
        except Exception as exc:
            logger.debug("indeed parse error: %s", exc)
    return jobs


def parse_mosaic_payload(payload: dict[str, Any]) -> list[RawScrapedJob]:
    jobs = []
    for r in parse_indeed_results(payload):
        try:
            job = parse_indeed_result(r)
            if job:
                jobs.append(job)
        except Exception as exc:
            logger.debug("indeed parse error: %s", exc)
    return jobs


def parse_indeed_html(html: str) -> list[RawScrapedJob]:
    payload = extract_mosaic_json(html)
    if not payload:
        return []
    return parse_mosaic_payload(payload)


def is_cloudflare_challenge(status_code: int, body: str) -> bool:
    if status_code not in (403, 503):
        return False
    lowered = (body or "").lower()
    if any(m in lowered for m in _CF_MARKERS):
        return True
    # A bare 403 on the SERP is also treated as a wall (escalate to browser).
    return status_code == 403


class IndeedExtractor(BaseExtractor):
    """Indeed India SERP with two-tier resilience.

    Tier A: curl_cffi TLS-impersonated fetch (~300ms when the egress IP is clean).
    Tier B: Playwright stealth browser — passes the Cloudflare managed
    challenge (invisible Turnstile auto-solves), then reads the mosaic payload
    directly from the page context. Once Tier B is needed, the rest of the run
    stays in the browser (cf_clearance is TLS-bound and not portable to
    curl_cffi).
    """

    portal_name = "indeed"
    site = "indeed"

    def build_search_url(
        self, keyword: str, location: str, freshness_days: int, start: int
    ) -> str:
        q = quote_plus(keyword)
        url = f"{SERP}?q={q}"
        if location and location.lower() not in {"india", ""}:
            url += f"&l={quote_plus(location)}"
        if freshness_days and freshness_days > 0:
            url += f"&fromage={int(freshness_days)}"
        if start:
            url += f"&start={start}"
        return url

    async def _fetch_page(
        self, url: str, use_browser: bool
    ) -> tuple[list[RawScrapedJob], bool]:
        """Fetch one SERP page. Returns (jobs, cf_detected)."""
        if not use_browser:
            try:
                resp = await self.session.get(
                    url,
                    headers={"Referer": f"{BASE}/", "Accept-Language": "en-IN,en;q=0.9"},
                    timeout=40,
                )
            except BotDetectionError:
                logger.warning("indeed SERP bot challenge detected; escalating to stealth browser")
                return [], True
            if is_cloudflare_challenge(resp.status_code, resp.text):
                logger.warning(
                    "indeed SERP hit Cloudflare challenge (HTTP %d); escalating to stealth browser",
                    resp.status_code,
                )
                return [], True
            if resp.status_code != 200:
                logger.warning("indeed SERP HTTP %d", resp.status_code)
                return [], False
            return parse_indeed_html(resp.text), False

        # Tier B — Playwright stealth (one shared browser across pages;
        # cf_clearance + cf_cookies persist in the context).
        from job_scraper.core.client import BrowserStealthContext

        if getattr(self, "_browser_page", None) is None:
            if getattr(self, "_browser", None) is None:
                self._browser = BrowserStealthContext()
            self._browser_page = await self._browser.__aenter__()
        page = self._browser_page
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        try:
            await page.wait_for_function(_MOSAIC_READY_JS, timeout=25000)
        except Exception:
            # Interactive Turnstile may be waiting on a checkbox click.
            try:
                frame = page.frame_locator('iframe[src*="challenges.cloudflare.com"]')
                await frame.locator(
                    "#challenge-stage, .ctp-checkbox-label, input[type='checkbox']"
                ).click(timeout=3000)
                await page.wait_for_function(_MOSAIC_READY_JS, timeout=15000)
            except Exception:
                pass
        payload = await page.evaluate(
            '() => window.mosaic && window.mosaic.providerData &&'
            ' window.mosaic.providerData["mosaic-provider-jobcards"] || null'
        )
        if payload:
            return parse_mosaic_payload(payload), False
        html = await page.content()
        jobs = parse_indeed_html(html)
        if not jobs:
            title = (await page.title() or "").lower()
            if "moment" in title or "security" in title or "challenge" in html:
                raise BotDetectionError(
                    f"Indeed Cloudflare wall unresolved in browser: "
                    f"{title or 'challenge page'}",
                    url=url,
                )
        return jobs, False

    async def aclose(self) -> None:
        """Release the shared stealth browser (idempotent)."""
        if getattr(self, "_browser", None) is not None:
            try:
                await self._browser.__aexit__(None, None, None)
            finally:
                self._browser = None
                self._browser_page = None

    async def _enrich_descriptions_browser(
        self, jobs: list[RawScrapedJob], page: Any
    ) -> None:
        """Enrich jobs with full HTML descriptions via Indeed's /rpc/jobdescs endpoint inside Playwright."""
        jk_map: dict[str, RawScrapedJob] = {}
        for j in jobs:
            m = re.search(r"jk=([a-zA-Z0-9]+)", j.job_url)
            if m:
                jk_map[m.group(1)] = j
        if not jk_map:
            return

        try:
            desc_dict = await page.evaluate(
                '''async (jks) => {
                    const out = {};
                    for (const jk of jks) {
                        try {
                            const r = await fetch('/rpc/jobdescs?jks=' + jk, {
                                headers: {'Accept': 'application/json'}
                            });
                            if (r.ok) {
                                const d = await r.json();
                                Object.assign(out, d);
                            }
                        } catch (e) {}
                    }
                    return out;
                }''',
                list(jk_map.keys()),
            )
            if isinstance(desc_dict, dict):
                for jk, html in desc_dict.items():
                    if jk in jk_map and html:
                        txt = BaseExtractor.html_to_text(html)
                        if txt and len(txt.strip()) > len(jk_map[jk].description):
                            jk_map[jk].description = txt.strip()
        except Exception as exc:
            logger.debug("browser detail enrichment failed: %s", exc)

    async def _enrich_descriptions(self, jobs: list[RawScrapedJob]) -> None:
        """Enrich jobs with full HTML descriptions via internal /rpc/jobdescs or stealth browser."""
        if not jobs:
            return
        if getattr(self, "_browser_page", None) is not None:
            await self._enrich_descriptions_browser(jobs, self._browser_page)
            return

        jk_map: dict[str, RawScrapedJob] = {}
        for j in jobs:
            m = re.search(r"jk=([a-zA-Z0-9]+)", j.job_url)
            if m:
                jk_map[m.group(1)] = j

        needs_browser = False
        for jk, job in jk_map.items():
            try:
                resp = await self.session.get(
                    f"{BASE}/rpc/jobdescs?jks={jk}",
                    headers={"Accept": "application/json"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    html = data.get(jk, "")
                    if html:
                        txt = BaseExtractor.html_to_text(html)
                        if txt and len(txt.strip()) > len(job.description):
                            job.description = txt.strip()
                elif resp.status_code == 403:
                    needs_browser = True
                    break
            except Exception:
                needs_browser = True
                break

        if needs_browser:
            from job_scraper.core.client import BrowserStealthContext
            try:
                async with BrowserStealthContext() as page:
                    await page.goto(
                        f"{BASE}/jobs?q=jobs",
                        wait_until="domcontentloaded",
                        timeout=45000,
                    )
                    await self._enrich_descriptions_browser(jobs, page)
            except Exception as exc:
                logger.debug("stealth browser detail fallback failed: %s", exc)

    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        if self.session is None:
            raise RuntimeError("IndeedExtractor requires a StealthSession")

        results: list[RawScrapedJob] = []
        start = 0
        page_size = 15
        use_browser = False
        try:
            while len(results) < limit:
                url = self.build_search_url(keyword, location, freshness_days, start)
                try:
                    page_jobs, cf = await self._fetch_page(url, use_browser)
                except BotDetectionError:
                    raise
                except Exception as exc:
                    logger.warning("indeed page start=%d failed: %s", start, exc)
                    break
                if cf:
                    use_browser = True
                    try:
                        page_jobs, _ = await self._fetch_page(url, True)
                    except BotDetectionError:
                        raise
                    except Exception as exc:
                        logger.warning("indeed browser fallback failed: %s", exc)
                        break
                if not page_jobs:
                    break
                results.extend(page_jobs)
                if len(page_jobs) < page_size:
                    break
                start += page_size
                if use_browser:
                    await asyncio.sleep(2.0)

            # De-dup by jobkey
            seen: set[str] = set()
            unique: list[RawScrapedJob] = []
            for j in results:
                if j.job_url not in seen:
                    seen.add(j.job_url)
                    unique.append(j)

            filtered = [
                j for j in unique
                if self.location_match(j.location, location)
                and self.is_fresh(j.date_posted, freshness_days)
            ]
            if location.lower() == "remote":
                remote = [j for j in filtered if j.is_remote]
                if remote:
                    filtered = remote
            filtered = filtered[:limit]

            # Enrich with full descriptions while browser/session is still active
            await self._enrich_descriptions(filtered)
            return filtered
        finally:
            await self.aclose()
