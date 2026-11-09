from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAM_PREFIXES = ("utm_",)
# Tracking/marketing params only. Never strip identity params (id, jobId, jk,
# tok, jobcode, position, etc.) — those distinguish distinct job postings.
TRACKING_PARAMS = {
    "ref", "ref_", "refid", "trackingid", "trk", "trkinfo", "gclid", "fbclid",
    "msclkid", "mc_cid", "mc_eid", "igshid", "vero_id", "oly_enc_id",
    "oly_anon_id", "spm", "src", "srcid", "submissionid", "pos",
    "itm_source", "itm_medium", "itm_campaign",
}

# Params that must never be stripped (identity/functional).
IDENTITY_PARAMS = {"id", "jobid", "jk", "tok", "jobcode", "position", "start",
                   "limit", "q", "k", "l", "fromage", "days", "page"}

DEFAULT_SALARY_LPA = [0, 0]

# Canonical skill taxonomy: token -> canonical skill name.
SKILL_TAXONOMY: dict[str, str] = {
    "python": "python",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "c++": "c++",
    "c#": "c#",
    "csharp": "c#",
    "golang": "golang",
    "go": "golang",
    "rust": "rust",
    "ruby": "ruby",
    "php": "php",
    "kotlin": "kotlin",
    "swift": "swift",
    "scala": "scala",
    "r": "r",
    "dart": "dart",
    "fastapi": "fastapi",
    "django": "django",
    "flask": "flask",
    "spring": "spring",
    "spring boot": "spring boot",
    "node.js": "node.js",
    "nodejs": "node.js",
    "node": "node.js",
    "express": "express",
    "express.js": "express",
    "next.js": "next.js",
    "nextjs": "next.js",
    "react": "react",
    "react.js": "react",
    "reactjs": "react",
    "angular": "angular",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "svelte": "svelte",
    "graphql": "graphql",
    "rest": "rest",
    "rest api": "rest api",
    "restful": "rest api",
    "grpc": "grpc",
    "sql": "sql",
    "mysql": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "mongo": "mongodb",
    "redis": "redis",
    "cassandra": "cassandra",
    "dynamodb": "dynamodb",
    "elasticsearch": "elasticsearch",
    "kafka": "kafka",
    "rabbitmq": "rabbitmq",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "google cloud": "gcp",
    "git": "git",
    "github": "github",
    "gitlab": "gitlab",
    "ci/cd": "ci/cd",
    "cicd": "ci/cd",
    "jenkins": "jenkins",
    "terraform": "terraform",
    "ansible": "ansible",
    "linux": "linux",
    "bash": "bash",
    "shell scripting": "shell scripting",
    "microservices": "microservices",
    "system design": "system design",
    "data structures": "data structures",
    "algorithms": "algorithms",
    "oop": "oop",
    "machine learning": "machine learning",
    "ml": "machine learning",
    "deep learning": "deep learning",
    "nlp": "nlp",
    "pandas": "pandas",
    "numpy": "numpy",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "llm": "llm",
    "genai": "genai",
    "generative ai": "genai",
    "ai": "ai",
    "etl": "etl",
    "airflow": "airflow",
    "spark": "spark",
    "hadoop": "hadoop",
    "hive": "hive",
    "tableau": "tableau",
    "power bi": "power bi",
    "excel": "excel",
    "selenium": "selenium",
    "playwright": "playwright",
    "cypress": "cypress",
    "jest": "jest",
    "pytest": "pytest",
    "junit": "junit",
    "git actions": "github actions",
    "github actions": "github actions",
    "nginx": "nginx",
    "postgres sql": "postgresql",
    "py": "python",
    "software development": "software development",
    "agile": "agile",
    "scrum": "scrum",
    "jira": "jira",
    "html": "html",
    "css": "css",
    "sass": "sass",
    "tailwind": "tailwind",
    "bootstrap": "bootstrap",
    "websocket": "websocket",
    "websockets": "websocket",
    "jwt": "jwt",
    "oauth": "oauth",
    "oauth2": "oauth2",
    "security": "security",
    "ci": "ci",
    "cd": "cd",
    "devops": "devops",
    "sre": "sre",
    "terraform cloud": "terraform",
}

# Multi-word skills must be matched first (longest-first matching).
_MULTI_WORD_SKILLS = sorted(
    (k for k in SKILL_TAXONOMY if " " in k or "/" in k or "+" in k or "#" in k or "." in k),
    key=len,
    reverse=True,
)
_SINGLE_WORD_SKILLS = [k for k in SKILL_TAXONOMY if k not in _MULTI_WORD_SKILLS]

_DEFAULT_ALLOWED_BRANCHES = ["CSE", "IT", "ECE"]

# Keep phrases before taxonomy matching so n-grams survive tokenization.
_PRESERVE_PHRASES = ["ci/cd", "power bi", "spring boot"]


def clean_url(url: str) -> str:
    """Canonicalize a job URL: strip tracking params, fragments, trailing slash."""
    if not url:
        raise ValueError("empty url")
    parts = urlsplit(url.strip())
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
        and not any(k.lower().startswith(p) for p in TRACKING_PARAM_PREFIXES)
    ]
    query = urlencode(query_pairs, doseq=True)
    path = parts.path
    if path.endswith("/") and path != "/":
        path = path.rstrip("/")
    # Normalize scheme/host casing, drop fragment.
    netloc = parts.netloc.lower()
    scheme = parts.scheme.lower()
    return urlunsplit((scheme, netloc, path, query, ""))


def generate_job_id(canonical_url: str) -> str:
    """Deterministic MD5-based external job ID: job-ext-<10 hex chars>."""
    digest = hashlib.md5(canonical_url.encode("utf-8")).hexdigest()[:10]
    return f"job-ext-{digest}"


def parse_salary_lpa(
    min_amt: Optional[float] = None,
    max_amt: Optional[float] = None,
    interval: Optional[str] = None,
) -> list[int]:
    """Normalize salary figures to an integer [min_lpa, max_lpa] pair.

    Rules:
      - interval "lpa": values already in Lakhs Per Annum (or <=100 heuristic).
      - interval "monthly": INR/month -> amt * 12 / 100000.
      - interval "annual": INR/year -> amt / 100000.
      - Missing, empty, or zero -> returns [0, 0] (never injects synthetic [6, 12] defaults).
    """

    def to_float(v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            val = float(v)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

    def to_lpa(value: float, iv: str) -> int:
        if iv == "monthly":
            lpa = value * 12 / 100_000
        elif iv == "annual":
            lpa = value / 100_000
        else:  # lpa or unknown numeric
            # Heuristic: numbers >= 10,000 are raw INR annual, while smaller numbers are already in LPA.
            if value >= 10_000 and iv in (None, "lpa"):
                lpa = value / 100_000
            else:
                lpa = value
        return max(0, round(lpa))

    num_min = to_float(min_amt)
    num_max = to_float(max_amt)
    if num_min is None and num_max is None:
        return list(DEFAULT_SALARY_LPA)
    iv = interval or "lpa"
    lo = to_lpa(num_min, iv) if num_min is not None else None
    hi = to_lpa(num_max, iv) if num_max is not None else None
    if lo is None and hi is None:
        return list(DEFAULT_SALARY_LPA)
    if lo is None:
        lo = hi
    if hi is None:
        hi = lo
    if lo == 0 and hi == 0:
        return list(DEFAULT_SALARY_LPA)
    if lo > hi:
        lo, hi = hi, lo
    return [int(lo), int(hi)]


def extract_skills(text: str, limit: int = 12) -> list[str]:
    """Match a technical-skill taxonomy against free text (word-boundary aware)."""
    if not text:
        return []
    haystack = " " + re.sub(r"\s+", " ", text.lower()) + " "
    found: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        canonical = SKILL_TAXONOMY[token]
        if canonical not in seen:
            seen.add(canonical)
            found.append(canonical)

    # Multi-word / symbolic skills first.
    for token in _MULTI_WORD_SKILLS:
        if len(found) >= limit:
            break
        pattern = r"(?<![\w+#./-])" + re.escape(token) + r"(?![\w+#./-])"
        if re.search(pattern, haystack):
            add(token)

    # Single-word skills with word boundaries.
    for token in _SINGLE_WORD_SKILLS:
        if len(found) >= limit:
            break
        pattern = r"(?<![\w])" + re.escape(token) + r"(?![\w])"
        if re.search(pattern, haystack):
            add(token)

    return found


def normalize_tags(tags: Iterable[str], limit: int = 12) -> list[str]:
    """Map raw portal tags through the skill taxonomy; fall back to text match.

    Returns canonical skill names (deduplicated, order-preserving).
    """
    text = " ".join(str(t) for t in tags if t)
    return extract_skills(text, limit=limit)


SENIOR_TITLE_RE = re.compile(
    r"\b("
    r"senior|sr\.?|lead|staff|principal|architect|manager|head|director|vp|specialist|consultant|expert|"
    r"(?:sde|engineer|developer|scientist|analyst|specialist|consultant|programmer)\s*[- ]*(?:iii|iv|v|vi|[3-9])|"
    r"(?:level|lvl|l|ic|e)[- ]*[3-9]|"
    r"(?:iii|iv|v|vi)"
    r")\b",
    re.IGNORECASE,
)

_EXPERIENCE_PATTERNS = [
    re.compile(r"(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?|yoe)", re.I),
    re.compile(r"(\d+)\s*\+\s*(?:years?|yrs?|yoe)", re.I),
    re.compile(r"(\d+)\s*(?:years?|yrs?|yoe)\s*(?:of)?\s*(?:experience)?", re.I),
    re.compile(r"experience\s*(?:of|:)?\s*(\d+)\s*(?:years?|yrs?|yoe)?", re.I),
]


def parse_experience_years(text: str) -> Optional[float]:
    """Extract minimum years of experience from free text; 0 for fresher."""
    if not text:
        return None
    if re.search(r"\bfresher\b|\bentry[\s-]level\b|\b0[\s-]+(?:1|2)\s*years?\b", text, re.I):
        return 0.0
    m = _EXPERIENCE_PATTERNS[0].search(text)
    if m:
        return float(m.group(1))
    for pat in _EXPERIENCE_PATTERNS[1:]:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None


def parse_date_posted(value: Any) -> Optional[str]:
    """Normalize heterogeneous date inputs to UTC ISO-8601 (e.g. 2026-11-01T08:30:00Z).

    Accepts:
      - epoch seconds/millis (int/float or numeric string)
      - ISO-8601 strings ("2026-10-25T14:22:00.000Z", "2026-10-25 14:22:00")
      - relative text ("3 days ago", "Posted 5 hours ago", "Today", "Few Hours Ago")
    Returns None when unparseable.
    """
    if value is None:
        return None

    # Numeric epoch.
    if isinstance(value, (int, float)) or (
        isinstance(value, str) and re.fullmatch(r"\d{9,16}(\.\d+)?", value.strip())
    ):
        try:
            num = float(value)
        except (TypeError, ValueError):
            num = None
        if num is not None:
            if num > 1e12:  # milliseconds
                num /= 1000.0
            if num > 1e9:  # plausible epoch
                try:
                    return datetime.fromtimestamp(num, tz=timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                except (OverflowError, OSError, ValueError):
                    return None

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    # ISO-ish strings.
    iso = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass

    # Relative English text.
    now = datetime.now(timezone.utc)
    lowered = text.lower()
    if re.search(r"\btoday\b|\bfew hours ago\b|\bjust now\b|\bhour[s]? ago\b|\bminutes? ago\b", lowered):
        return now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if re.search(r"\byesterday\b", lowered):
        return (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    m = re.search(r"(\d+)\s*(day|days|week|weeks|month|months)\s*ago", lowered)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("day"):
            delta = timedelta(days=n)
        elif unit.startswith("week"):
            delta = timedelta(weeks=n)
        else:
            delta = timedelta(days=30 * n)
        return (now - delta).strftime("%Y-%m-%dT%H:%M:%SZ")

    m = re.search(r"(\d+)\s*/\s*(\d+)\s*/\s*(\d{4})", text)
    if m:
        d, mo, y = (int(x) for x in m.groups())
        try:
            return datetime(y, mo, d, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None

    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1) + "T00:00:00Z"

    return None


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("&amp;", "&").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def _footer(source: str, engine: str, job_url: str) -> str:
    return f"Apply URL: {job_url}\nSource: {source} (Scraped via {engine})"


def to_placement_agent_job(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a RawScrapedJob-shaped dict into a PlacementJob-shaped dict."""
    from job_scraper.core.models import PlacementJob

    title = _normalize_text(raw.get("title"))
    company = _normalize_text(raw.get("company"))
    location = _normalize_text(raw.get("location")) or "India"
    engine = raw.get("scraper_engine") or "stealth_harvester"
    source = raw.get("site") or raw.get("source") or "unknown"

    canonical = clean_url(raw["job_url"])
    description_body = _normalize_text(raw.get("description"))
    footer = _footer(source, engine, canonical)
    description = f"{description_body}\n\n{footer}".strip()

    # Map raw portal tags through the taxonomy so non-technical phrases
    # ("Business process improvement") never leak into required_skills.
    raw_tags = [str(t) for t in (raw.get("tags") or []) if t]
    skill_pool = normalize_tags(raw_tags)
    # Supplement with taxonomy matches from title/description when sparse.
    if len(skill_pool) < 6:
        for s in extract_skills(f"{title} {description_body}"):
            if s not in skill_pool:
                skill_pool.append(s)
            if len(skill_pool) >= 12:
                break
    required = skill_pool[:6]
    preferred = skill_pool[6:12]

    salary = parse_salary_lpa(raw.get("min_amount"), raw.get("max_amount"), raw.get("interval"))
    salary_disclosed = bool(salary and any(x > 0 for x in salary))
    exp = raw.get("experience_years")
    if exp is None:
        exp = parse_experience_years(f"{title} {description_body}")
    if (exp is None or float(exp) <= 0) and SENIOR_TITLE_RE.search(title):
        title_low = title.lower()
        if re.search(r"\b(?:iii|iv|v|[3-9])\b", title_low):
            exp = 5.0
        else:
            exp = 3.0
    elif (exp is None or float(exp) <= 0) and re.search(r"\b(?:mid[\s-]level|mid[\s-]senior|intermediate|(?:sde|engineer|developer|scientist|analyst|programmer)\s*[- ]*(?:ii|2)\b|(?:level|lvl|l|ic|e)[- ]*2\b)\b", title, re.I):
        exp = 2.0
    elif (exp is None or float(exp) <= 0) and len(description_body) < 50 and not re.search(r"\b(?:interns?|internship|trainees?|freshers?|graduate\s+engineer|graduate\s+trainee|entry[\s-]level|associate|junior|jr\.?)\b", title, re.I):
        if source not in ("internshala", "unstop"):
            exp = 2.0

    job = PlacementJob(
        id=generate_job_id(canonical),
        title=title,
        company=company,
        description=description,
        required_skills=required,
        preferred_skills=preferred,
        experience_years=float(exp) if exp is not None else None,
        location=location,
        remote=bool(raw.get("is_remote")),
        salary_lpa=salary,
        salary_disclosed=salary_disclosed,
        min_gpa=7.0,
        allowed_branches=list(_DEFAULT_ALLOWED_BRANCHES),
        job_url=canonical,
        source=source,
        scraper_engine=engine,
    )
    return job.model_dump()


def _norm_key_part(value: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_text(value).lower())


def _soft_key(job: dict[str, Any]) -> tuple[str, str, str]:
    """Soft identity: (title, company, location).

    Location is required so multiple distinct openings for the same role at the
    same company (e.g. 30 Infosys Python Developer roles across cities) are not
    collapsed into one record.
    """
    return (
        _norm_key_part(job.get("title")),
        _norm_key_part(job.get("company")),
        _norm_key_part(job.get("location")),
    )


def dedupe_jobs(jobs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross-portal dedup: primary on canonical URL, secondary on
    (title, company, location).

    When two records collide, the one with the longer description wins.
    """
    by_url: dict[str, dict[str, Any]] = {}
    soft_index: dict[tuple[str, str, str], str] = {}  # soft key -> canonical url key

    def description_len(j: dict[str, Any]) -> int:
        return len(j.get("description") or "")

    kept: list[dict[str, Any]] = []
    for job in jobs:
        if not job.get("job_url"):
            continue
        try:
            canonical = clean_url(job["job_url"])
        except ValueError:
            continue

        soft = _soft_key(job)
        existing_canonical = soft_index.get(soft)
        if existing_canonical is not None:
            existing = by_url[existing_canonical]
            if description_len(job) > description_len(existing):
                by_url[existing_canonical] = job
                # keep old canonical as identity key to preserve ID stability
            continue

        if canonical in by_url:
            existing = by_url[canonical]
            if description_len(job) > description_len(existing):
                by_url[canonical] = job
            continue

        by_url[canonical] = job
        soft_index[soft] = canonical
        kept.append(job)

    # Re-materialize, replacing kept entries with possibly upgraded versions.
    result = []
    seen_ids = set()
    for job in kept:
        canonical = clean_url(job["job_url"])
        final = by_url.get(canonical, job)
        ident = id(final)
        if ident in seen_ids:
            continue
        seen_ids.add(ident)
        result.append(final)
    return result
