"""Dynamic, AI-powered resume skill & proficiency extraction (Milestone 1).

Replaces the legacy static keyword interception (``ADVANCED_HINT`` /
``INTERMEDIATE_HINT``) and naive line-mention counting with a contextual
pipeline driven by the configured ``LLMClient`` (``TASK: extract_resume``):

* ``extract_resume_with_ai`` - the primary entry point. Sends the scrubbed
  resume plus the 124-skill taxonomy to the LLM, strictly grounds every
  returned skill against the taxonomy (hallucination guard), verifies that
  each accepted skill is actually present in the resume text, merges any
  explicitly-mentioned skill the LLM omitted, and falls back to deterministic
  contextual analysis whenever the LLM is unavailable, errors, or returns
  invalid JSON - so a resume upload never fails with HTTP 500.
* ``extract_skills_with_llm`` - convenience wrapper returning only the
  ``{skill_id: skill_record}`` mapping.
* ``extract_skills`` - legacy contract preserved for existing callers
  (``{skill_id: [evidence lines]}``); delegates to the AI engine when an
  ``llm`` is supplied and performs the word-boundary alias scan otherwise.
* ``level_for`` - proficiency heuristic with an optional AI-assisted path,
  fully backward compatible with the original ``level_for(5) == "advanced"``
  unit-test contract.

``app/resume/extractor.py`` re-exports the public API for module-path
compatibility.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import re
import weakref
from collections import OrderedDict

from ..security.guardrails import sanitize_untrusted_text
from ..taxonomy.branches import normalize_branch
from ..taxonomy.taxonomy import Skill, Taxonomy

logger = logging.getLogger(__name__)

BOUND = r"(?<![a-z0-9]){alias}(?![a-z0-9])"

_NEGATIVE_OR_FUTURE_RE = re.compile(
    r"\b(?:"
    r"did\s+not\s+(?:use|know|work\s+with|have)|didn'?t\s+(?:use|know|work)|"
    r"have\s+not\s+used|haven'?t\s+(?:used|touched|worked\s+with)|never\s+(?:used|worked\s+with)|"
    r"not\s+used|didn'?t\s+touch|"
    r"no\s+(?:experience|prior\s+experience|hands[- ]on|exposure|knowledge)\s+(?:with|in|of)?|"
    r"not\s+(?:familiar|experienced|proficient|fluent)\s+(?:with|in)|"
    r"without\s+(?:using|any)|lack(?:s|ing)?\s+(?:experience\s+(?:with|in)\s+)?|"
    r"(?:coursework|topics?|skills?)\s+excluded|excluded\s+(?:from|coursework)|"
    r"familiarity\s+hoped|hoped\s+for|"
    r"(?:interested|keen)\s+in(?:\s+(?:learning|exploring|working\s+with|about))?|"
    r"(?:interested|keen|eager|hoping|planning|willing|looking|aiming|would\s+like)\s+(?:in\s+|to\s+)?(?:learn|explore|study|pick\s+up|work\s+with)(?:ing)?|"
    r"want\s+to\s+(?:explore|learn|study)|plan\s+to\s+learn|hope\s+to\s+learn|"
    r"yet\s+to\s+(?:learn|explore|use)|"
    r"aspiring\s+to\s+learn|currently\s+learning\s+(?:about\s+|the\s+basics\s+of\s+)?"
    r")\b",
    re.IGNORECASE,
)
_POSITIVE_RESET_RE = re.compile(
    r"\b(?:used|using|built|developed|implemented|deployed|shipped|architected|"
    r"worked\s+with|working\s+with|experienced\s+in|proficient\s+in|skilled\s+in|"
    r"skills?\s*:|tools?\s*:|tech\s+stack\s*:)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_GO_PHRASES = (
    "go-to-market",
    "go to market",
    "ready to go",
    "on the go",
    "on-the-go",
    "way to go",
    "go beyond",
    "go through",
    "go into",
    "go back",
    "go ahead",
    "let's go",
    "lets go",
)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    low = alias.lower().strip()
    if low == "c":
        return re.compile(
            r"(?<!objective-)(?<!objective )(?<![a-z0-9+#&.])c"
            r"(?!\+\+|#|\s*[-/]\s*(?:suite|level|section))(?![a-z0-9+#.])",
            re.IGNORECASE,
        )
    if low == "c++":
        return re.compile(r"(?<![a-z0-9+#.])c\+\+(?:\d{2})?(?![a-z_+#])", re.IGNORECASE)
    if low == "c#":
        return re.compile(r"(?<![a-z0-9+#.])c#(?![a-z0-9+#])", re.IGNORECASE)
    if low in (".net", "dotnet"):
        return re.compile(r"(?:(?<![a-z0-9_])|(?<=asp))\.net(?![a-z0-9])|(?<![a-z0-9])dotnet(?![a-z0-9])", re.IGNORECASE)
    if low in ("node.js", "nodejs", "node js"):
        return re.compile(r"(?<![a-z0-9])node(?:\.|\s+|)js(?![a-z0-9])", re.IGNORECASE)
    if low in ("next.js", "nextjs", "next js"):
        return re.compile(r"(?<![a-z0-9])next(?:\.|\s+|)js(?![a-z0-9])", re.IGNORECASE)
    if low in ("vue.js", "vuejs"):
        return re.compile(r"(?<![a-z0-9])vue(?:\.|\s+|)js(?![a-z0-9])", re.IGNORECASE)
    if low in ("ci/cd", "ci-cd", "cicd", "ci cd"):
        return re.compile(r"(?<![a-z0-9])ci(?:\s*/\s*|\s*-\s*|\s*)cd(?![a-z0-9])", re.IGNORECASE)
    if low == "r":
        return re.compile(r"(?<![a-z0-9&/.-])r(?![a-z0-9&/.-])", re.IGNORECASE)
    if low == "go":
        return re.compile(r"(?<![a-z0-9-])go(?![a-z0-9-])", re.IGNORECASE)
    if low == "rust":
        return re.compile(r"(?<![a-z0-9-])rust(?!-(?:free|proof))(?![a-z0-9])", re.IGNORECASE)
    if any(sep in low for sep in (" ", "-", "/", ".")):
        parts = [re.escape(p) for p in re.split(r"[\s/.-]+", low) if p]
        if len(parts) >= 2:
            sep_pat = r"[\s./-]+"
            return re.compile(rf"(?<![a-z0-9]){sep_pat.join(parts)}(?![a-z0-9])", re.IGNORECASE)
    return re.compile(BOUND.format(alias=re.escape(low)), re.IGNORECASE)


def _match_is_positive(line: str, alias: str, m: re.Match[str]) -> bool:
    low_alias = alias.lower()
    low_line = line.lower()
    matched_txt = line[m.start() : m.end()]

    if low_alias == "r":
        # Reject R&D / R/W contexts or casual lowercase 'r' outside explicit skills lists
        window = low_line[max(0, m.start() - 6) : min(len(low_line), m.end() + 6)]
        if "&" in window or "r&d" in window or "r & d" in window:
            return False
        if matched_txt != "R" and not re.search(
            r"\b(?:skills?|tools?|languages?|r\s+(?:programming|language|studio)|ggplot|dplyr|tidyverse)\b",
            low_line,
        ):
            return False

    if low_alias == "go":
        win = low_line[max(0, m.start() - 16) : min(len(low_line), m.end() + 16)]
        if any(phrase in win for phrase in _AMBIGUOUS_GO_PHRASES):
            return False
        if matched_txt != "Go" and not re.search(
            r"\b(?:skills?|languages?|golang|goroutine|gin|fiber|backend|microservices?)\b",
            low_line,
        ):
            return False

    if low_alias == "c":
        before = low_line[max(0, m.start() - 12) : m.start()]
        if re.search(r"\b(?:option|section|plan|tier|part|vitamin|grade)\s+$", before):
            return False

    # Negation & future-learning clause check (look back to last sentence boundary)
    clause_start = max(
        low_line.rfind(".", 0, m.start()),
        low_line.rfind(";", 0, m.start()),
        low_line.rfind("!", 0, m.start()),
    ) + 1
    prefix = low_line[clause_start : m.start()]
    neg_matches = list(_NEGATIVE_OR_FUTURE_RE.finditer(prefix))
    if neg_matches:
        last_neg_end = neg_matches[-1].end()
        tail_after_neg = prefix[last_neg_end:]
        if not _POSITIVE_RESET_RE.search(tail_after_neg):
            return False
    return True


def _line_matches_alias(line: str, alias: str) -> bool:
    pat = _alias_pattern(alias)
    for m in pat.finditer(line):
        if _match_is_positive(line, alias, m):
            return True
    return False

LEVELS = ("basic", "intermediate", "advanced")

EXTRACT_RESUME_SYSTEM = (
    "TASK: extract_resume\n"
    "You are an expert technical recruiter and AI resume parser. Analyze the candidate's "
    "resume dynamically and extract:\n"
    "1. Candidate profile metadata (name, branch, year, gpa, summary) if present in the text.\n"
    "For candidate profile 'branch', return standard canonical branch codes (e.g. 'CSE', 'AI&DS', 'ECE', 'IT', 'EE', 'ME', 'CE', 'CHE', or 'BT') whenever applicable.\n"
    "2. Every technical and soft skill from the provided taxonomy that the candidate demonstrates.\n"
    "Apply this strict calibration rubric consistently for each skill's proficiency level ('basic', 'intermediate', or 'advanced'):\n"
    "- 'advanced': production internships or full-time work experience, complex end-to-end architecture ownership, quantified impact metrics (latency/throughput/users/%), or deep evidence across >=3 distinct projects/roles.\n"
    "- 'intermediate': applied hands-on project implementation, practical framework/tool usage with concrete deliverables, certifications with project work, or >=2 applied mentions.\n"
    "- 'basic': listed only in a Skills/Tools list, introductory coursework without project execution, or a single flat mention without applied depth.\n"
    "Do NOT extract negated, excluded, or aspirational ('willing/interested to learn') skills. "
    "You MUST only return skills that exist in the provided taxonomy and are grounded in the resume text.\n"
    "Respond ONLY with valid JSON matching this schema:\n"
    '{"profile": {"name": str, "branch": str, "year": int, "gpa": float, "summary": str}, '
    '"skills": [{"skill_id": str, "level": "basic"|"intermediate"|"advanced", "mentions": int, "evidence": [str]}]}'
)

# --- contextual (deterministic) analysis primitives -------------------------

_SECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("summary", re.compile(r"^\s*(?:summary|executive\s+summary|objective|profile|about(?:\s+me)?)\s*(?:[:\-]|$)", re.I)),
    ("skills", re.compile(r"^\s*(?:technical\s+skills?|core\s+skills?|skills?|technologies|tech\s+stack|tools|competencies)\s*(?:[:\-]|$)", re.I)),
    ("experience", re.compile(r"^\s*(?:work\s+experience|internship\s+experience|professional\s+experience|experience|employment|internships?)\s*(?:[:\-]|$)", re.I)),
    ("projects", re.compile(r"^\s*(?:key\s+projects?|academic\s+projects?|personal\s+projects?|projects?|capstone)\s*(?:[:\-]|$)", re.I)),
    ("coursework", re.compile(r"^\s*(?:relevant\s+coursework|coursework|courses|subjects)\s*(?:[:\-]|$)", re.I)),
    ("certifications", re.compile(r"^\s*(?:certifications?|certificates?|licenses?|achievements?|awards?)\s*(?:[:\-]|$)", re.I)),
    ("education", re.compile(r"^\s*(?:education|academics?|academic\s+background)\s*(?:[:\-]|$)", re.I)),
)
_DURATION_RE = re.compile(
    r"\b(?:\d+\+?\s*(?:years?|yrs?|months?|mos?)|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+20\d{2}\s*[-–to]+\s*(?:present|current|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+20\d{2})|"
    r"20\d{2}\s*[-–]\s*(?:20\d{2}|present|current))\b",
    re.I,
)
_COURSEWORK_ONLY_HINT = re.compile(r"\b(coursework|curriculum|syllabus|introductory|beginner)\b", re.I)

_EXTRACTION_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_INSTANCE_EXTRACTION_CACHE: weakref.WeakKeyDictionary[object, OrderedDict[tuple[str, str], dict]] = (
    weakref.WeakKeyDictionary()
)
_EXTRACTION_CACHE_MAX = 256

_METRIC_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|percent|ms|s\b|k\b|m\b|x\b|requests?|users?|customers?|rows?|"
    r"gb|mb|tb|qps|rps|lakh|crore|million|billion|times?|faster)",
    re.I,
)
_STRONG_VERBS = {
    "architected", "engineered", "optimized", "scaled", "led", "launched", "shipped",
    "orchestrated", "spearheaded", "automated", "migrated", "deployed", "refactored",
    "streamlined", "accelerated", "reduced", "improved",
}
_APPLIED_VERBS = {
    "built", "developed", "implemented", "created", "designed", "integrated", "tested",
    "configured", "maintained", "analyzed", "modeled", "delivered",
}
_EXPERIENCE_HINT = re.compile(
    r"\b(intern(ship)?|experience|employment|production|deployed|employer|company|"
    r"work(ed)? with|spearheaded|led)\b",
    re.I,
)
_PROJECT_HINT = re.compile(r"\b(project|built|developed|implemented|capstone|hackathon)\b", re.I)
_SKILLS_SECTION_HINT = re.compile(r"^\s*(skills?|technologies|tech stack|tools)\s*:", re.I)
_QUANTIFIED_SECTION_HINT = re.compile(
    r"\b(achievement|award|experience|internship|project|portfolio)\b", re.I
)


def _strip_code_fences(raw: str) -> str:
    """Drop optional markdown code fences around a JSON payload."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = [
            ln for ln in text.splitlines()
            if not ln.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()
    return text


def _safe_json(raw: str) -> dict | None:
    """Parse ``raw`` as a JSON object, tolerating markdown fences; None on failure."""
    if not isinstance(raw, str):
        return None
    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned or raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_skill(raw_id, taxonomy: Taxonomy) -> Skill | None:
    """Resolve an LLM-returned skill token to a canonical ``Skill`` (or None).

    Accepts the skill id, an alias, or a case-insensitive skill-name match so
    slightly-off LLM outputs still ground against the taxonomy.
    """
    if not isinstance(raw_id, str) or not raw_id.strip():
        return None
    token = raw_id.strip()
    skill = taxonomy.get_opt(token)
    if skill is not None:
        return skill
    resolved_id = taxonomy.resolve(token)
    if resolved_id is not None:
        return taxonomy.get_opt(resolved_id)
    low = token.lower()
    for candidate in taxonomy.all():
        if candidate.name.lower() == low:
            return candidate
    return None


def _alias_variants(skill: Skill, taxonomy: Taxonomy) -> list[str]:
    """Lower-cased name/id/alias variants used for grounding checks."""
    variants = {skill.id.lower(), skill.name.lower()}
    variants.update(a.lower() for a in (skill.aliases or ()))
    # alias_map is keyed by lower-cased alias/id and maps back to skill ids
    for alias, sid in taxonomy.alias_map.items():
        if sid == skill.id:
            variants.add(alias)
    return sorted(v for v in variants if v)


def _grounded(skill: Skill, text: str, evidence: list[str], taxonomy: Taxonomy) -> bool:
    """True when the skill (or one of its aliases) appears positively in ``text`` OR at
    least one non-empty positive evidence line grounds it."""
    lines = text.splitlines()
    for ln in lines:
        for variant in _alias_variants(skill, taxonomy):
            if _line_matches_alias(ln, variant):
                return True
    low_text = text.lower()
    for snippet in evidence:
        snippet_clean = str(snippet).strip()
        if not snippet_clean or snippet_clean.lower() not in low_text:
            continue
        if any(_line_matches_alias(snippet_clean, v) for v in _alias_variants(skill, taxonomy)):
            return True
    return False


def _skill_lines(skill: Skill, text: str, taxonomy: Taxonomy) -> list[str]:
    """Resume lines in which this skill's name/id/aliases appear positively (word-bounded)."""
    out: list[str] = []
    for line in text.splitlines():
        for variant in _alias_variants(skill, taxonomy):
            if _line_matches_alias(line, variant):
                stripped = line.strip()
                if stripped and stripped not in out:
                    out.append(stripped)
                break
    return out


def _line_has_metric(line: str) -> bool:
    return bool(_METRIC_RE.search(line))


def _leading_verb(line: str) -> str:
    for token in re.findall(r"[A-Za-z]+", line):
        if token.isalpha():
            return token.lower()
    return ""


def _parse_sections(text: str) -> list[tuple[str, str]]:
    """Tag each non-empty resume line with its active section name."""
    tagged: list[tuple[str, str]] = []
    current = "general"
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        matched_section = None
        for sec_name, pat in _SECTION_RULES:
            if pat.match(line):
                matched_section = sec_name
                break
        if matched_section:
            current = matched_section
            tagged.append((current, line))
        else:
            tagged.append((current, line))
    return tagged


def _line_sections_map(text: str) -> dict[str, str]:
    """Map stripped resume line -> detected section name."""
    out: dict[str, str] = {}
    for sec, line in _parse_sections(text):
        out.setdefault(line, sec)
    return out


def _contextual_level(lines: list[str], *, sections: dict[str, str] | None = None) -> str:
    """Deterministic proficiency assessment from a skill's evidence lines.

    Combines section weight, concrete action verbs, duration/years signals,
    quantified impact metrics, and multi-project evidence.
    """
    if not lines:
        return "basic"
    joined = " ".join(lines).lower()
    sec_map = sections or {}
    line_secs = [sec_map.get(ln.strip(), "general") for ln in lines]
    metric_hits = sum(1 for ln in lines if _line_has_metric(ln))
    duration_hits = sum(1 for ln in lines if _DURATION_RE.search(ln))
    strong_verb_hits = sum(
        1 for ln in lines
        if _leading_verb(ln) in _STRONG_VERBS
        or any(v in ln.lower() for v in _STRONG_VERBS)
    )
    applied_hits = sum(
        1 for ln in lines
        if _leading_verb(ln) in _APPLIED_VERBS
        or any(v in ln.lower() for v in _APPLIED_VERBS)
    )
    experience_ctx = bool(_EXPERIENCE_HINT.search(joined)) or ("experience" in line_secs)
    project_ctx = bool(_PROJECT_HINT.search(joined)) or ("projects" in line_secs)
    applied_ctx = experience_ctx or project_ctx or ("certifications" in line_secs)
    flat_list_only = all(
        _SKILLS_SECTION_HINT.search(ln) or sec_map.get(ln.strip()) in ("skills", "coursework", "education")
        for ln in lines
    ) and not (applied_hits or strong_verb_hits or metric_hits)

    # Strong evidence: production/experience + metrics or strong verbs or duration,
    # or >=3 distinct non-flat project/experience lines.
    if metric_hits >= 1 and (applied_ctx or strong_verb_hits >= 1):
        return "advanced"
    if strong_verb_hits >= 1 and (experience_ctx or metric_hits >= 1 or duration_hits >= 1):
        return "advanced"
    if experience_ctx and duration_hits >= 1 and (applied_hits >= 1 or len(lines) >= 2):
        return "advanced"
    if len(lines) >= 3 and not flat_list_only:
        return "advanced"
    if metric_hits >= 1 or strong_verb_hits >= 1:
        return "intermediate"
    if applied_ctx or applied_hits >= 1 or duration_hits >= 1:
        return "intermediate"
    if len(lines) >= 2 and not all(_COURSEWORK_ONLY_HINT.search(ln) for ln in lines):
        return "intermediate"
    return "basic"


_LEVEL_RANK = {"basic": 0, "intermediate": 1, "advanced": 2}


def _reconcile_level(
    llm_level: str,
    ctx_level: str,
    lines: list[str],
    mentions: int = 1,
) -> str:
    """Reconcile an LLM-assigned level with evidence-anchored contextual signals."""
    ctx_norm = _normalize_level(ctx_level, default="basic")
    llm_norm = _normalize_level(llm_level, default="")
    if not llm_norm:
        return ctx_norm
    if not lines:
        return llm_norm

    flat_list_only = all(_SKILLS_SECTION_HINT.search(ln) for ln in lines) and not any(
        _line_has_metric(ln) or any(v in ln.lower() for v in (_STRONG_VERBS | _APPLIED_VERBS))
        for ln in lines
    )
    if flat_list_only and len(lines) <= 1 and mentions <= 1:
        return "basic"

    diff = abs(_LEVEL_RANK[llm_norm] - _LEVEL_RANK[ctx_norm])
    if diff == 0:
        return ctx_norm
    if diff >= 2:
        # Never allow 2-tier drift (basic <-> advanced) away from actual resume evidence
        return ctx_norm
    if ctx_norm == "basic" and flat_list_only:
        return "basic"
    return llm_norm


def _normalize_level(value, default: str = "basic") -> str:
    level = str(value or "").strip().lower()
    return level if level in LEVELS else default


def _normalize_evidence(value, text: str) -> list[str]:
    """Deduplicated, non-empty evidence snippets grounded in ``text``."""
    raw_items = value if isinstance(value, (list, tuple)) else [value]
    low_text = text.lower()
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        snippet = str(item or "").strip()
        if not snippet:
            continue
        key = snippet.lower()
        if key in seen:
            continue
        # keep only snippets traceable to the resume text (anti-hallucination)
        if snippet in text or key in low_text:
            seen.add(key)
            out.append(snippet)
    return out


def _normalize_mentions(value, evidence: list[str], lines: list[str]) -> int:
    try:
        mentions = int(value)
    except (TypeError, ValueError):
        mentions = 0
    if mentions <= 0:
        mentions = max(1, len(evidence) or len(lines) or 1)
    return max(1, mentions)


def _taxonomy_payload(taxonomy: Taxonomy) -> list[dict]:
    return [
        {"id": s.id, "name": s.name, "aliases": sorted(set(s.aliases))}
        for s in sorted(taxonomy.all(), key=lambda item: item.id)
    ]


def _empty_profile() -> dict:
    return {"name": "", "branch": "", "year": None, "gpa": None, "summary": ""}


def _normalize_profile(value) -> dict:
    profile = _empty_profile()
    if not isinstance(value, dict):
        return profile
    profile["name"] = str(value.get("name") or "").strip()
    profile["branch"] = normalize_branch(str(value.get("branch") or ""))
    summary = str(value.get("summary") or "").strip()
    profile["summary"] = summary
    year = value.get("year")
    try:
        profile["year"] = int(year) if year is not None and str(year).strip() != "" else None
    except (TypeError, ValueError):
        profile["year"] = None
    gpa = value.get("gpa")
    try:
        profile["gpa"] = float(gpa) if gpa is not None and str(gpa).strip() != "" else None
    except (TypeError, ValueError):
        profile["gpa"] = None
    return profile


def _extract_contextual_branch(text: str) -> str:
    """Detect and canonicalize a branch mention from resume text."""
    # 1. Explicit Branch / Department label or degree-prefixed branch
    explicit = re.search(
        r"(?:\b(?:branch|department|dept|discipline|specialization|major)\s*[:\-]\s*|"
        r"\b(?:b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|b\.?sc|m\.?sc)\s+(?:in\s+)?)"
        r"([A-Za-z][A-Za-z0-9\s&/.\-]+?(?:\([^)]+\))?)(?=\s*(?:[,|\n;]|year|cgpa|gpa|batch|class|20\d{2}|$))",
        text,
        re.I,
    )
    if explicit:
        cand = explicit.group(1).strip()
        norm = normalize_branch(cand)
        if norm:
            return norm

    # 2. Parenthetical specializations & full engineering branch names (case-insensitive)
    phrase_match = re.search(
        r"\b(CSE\s*\([^)]+\)|ECE\s*\([^)]+\)|AI\s*&\s*DS|AI-DS|AIDS|AIML|"
        r"computer\s+science(?:\s*(?:&|and)\s*engineering)?|computer\s+engineering|"
        r"artificial\s+intelligence(?:\s*(?:&|and)\s*(?:data\s+science|machine\s+learning))?|"
        r"information\s+technology|"
        r"electronics(?:\s*(?:&|and)\s*communication(?:\s+engineering)?)?|"
        r"electrical(?:\s*(?:&|and)\s*(?:electronics|engineering))?|"
        r"mechanical(?:\s+engineering)?|"
        r"civil(?:\s+engineering)?|"
        r"chemical(?:\s+engineering)?|"
        r"biotech(?:nology)?)\b",
        text,
        re.I,
    )
    if phrase_match:
        norm = normalize_branch(phrase_match.group(1).strip())
        if norm:
            return norm

    # 3. Standalone uppercase branch abbreviations (case-sensitive to avoid English 'it'/'me')
    acro_match = re.search(r"\b(CSE|CS|AI&DS|AIDS|AIML|ECE|EEE|EE|IT|ME|CE|CHE|CH|BT)\b", text)
    if acro_match:
        norm = normalize_branch(acro_match.group(1).strip())
        if norm:
            return norm
    return ""


def _extract_contextual_summary(text: str) -> str:
    """Extract concise summary text from a Summary / Objective section if present."""
    parts: list[str] = []
    for sec, line in _parse_sections(text):
        if sec != "summary":
            continue
        cleaned = re.sub(
            r"^(?:summary|executive\s+summary|objective|profile|about(?:\s+me)?)\s*[:\-]?\s*",
            "",
            line,
            flags=re.I,
        ).strip()
        if cleaned:
            parts.append(cleaned)
        if len(parts) >= 3:
            break
    return " ".join(parts)[:400].strip()


def _contextual_profile(text: str) -> dict:
    """Lightweight deterministic profile guess used by the fallback path.

    Extracts a GPA, a canonical branch (CSE/AI&DS/ECE/IT/...), a graduation
    year, a summary section, and a leading candidate name.
    """
    profile = _empty_profile()
    if not text.strip():
        return profile

    gpa_match = re.search(
        r"\b(?:gpa|cgpa)\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,3})?)\b", text, re.I
    )
    if gpa_match:
        try:
            profile["gpa"] = float(gpa_match.group(1))
        except ValueError:
            profile["gpa"] = None

    profile["branch"] = _extract_contextual_branch(text)
    profile["summary"] = _extract_contextual_summary(text)

    year_match = re.search(
        r"\b(?:graduat|batch|class of|class of \d{4}|b\.?tech.*?)(?:e?d)?\s*(?:year)?\s*[:\-]?\s*(20\d{2}|\d)\b",
        text, re.I,
    )
    if year_match:
        token = year_match.group(1)
        profile["year"] = int(token) if len(token) == 4 else int(token)

    # Leading candidate-name heuristic: first capitalized two-or-three word line
    # that is not a section header, contact line, degree/year/gpa marker, or date.
    contact_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|https?://|www\.|linkedin|github", re.I)
    header_re = re.compile(
        r"^(resume|curriculum vitae|cv|skills?|education|experience|projects?|"
        r"achievements?|summary|objective|contact|profile|certifications?)\b",
        re.I,
    )
    non_name_re = re.compile(
        r"\b(b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?e\b|m\.?e\b|bca|mca|"
        r"year|cgpa|gpa|semester|batch|student|engineer|developer|aspirant)\b",
        re.I,
    )
    verb_start_re = re.compile(
        r"^(built|developed|designed|implemented|created|architected|engineered|"
        r"led|ran|made|worked|spearheaded|launched|optimized|deployed|maintained|"
        r"used|wrote|configured|delivered|automated|integrated|tested)\b",
        re.I,
    )
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if not stripped or len(stripped) > 60:
            continue
        if contact_re.search(stripped) or header_re.match(stripped):
            continue
        words = re.findall(r"[A-Z][a-z]+", stripped)
        if not (2 <= len(words) <= 4):
            continue
        if non_name_re.search(stripped) or verb_start_re.match(stripped):
            continue
        profile["name"] = " ".join(words)
        break

    # Comma-separated header form: "Rohan Gupta, B.Tech CSE, Year 3, CGPA 8.4"
    # The leading segment before the first comma is often the clean candidate
    # name; validate it has no degree/year/gpa/verb markers before accepting.
    if not profile["name"]:
        for line in text.splitlines()[:8]:
            stripped = line.strip()
            if not stripped or len(stripped) > 60:
                continue
            if contact_re.search(stripped) or header_re.match(stripped):
                continue
            head, sep, _tail = stripped.partition(",")
            seg = head if sep else stripped
            if non_name_re.search(seg) or verb_start_re.match(seg):
                continue
            seg_words = re.findall(r"[A-Za-z]+", re.sub(r"[^A-Za-z\s]", " ", seg))
            if not (2 <= len(seg_words) <= 3):
                continue
            if any(not w[0].isupper() for w in seg_words):
                continue
            profile["name"] = " ".join(seg_words)
            break
    return profile


def _cache_parts(clean_text: str, taxonomy: Taxonomy, llm: object) -> tuple[bool, tuple]:
    text_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    tax_sig = hashlib.sha256(
        ",".join(sorted(s.id for s in taxonomy.all())).encode("utf-8")
    ).hexdigest()[:16]
    mod = getattr(type(llm), "__module__", "")
    is_real_provider = mod.startswith("app.llm.") and type(llm).__name__ != "MockLLM"
    if is_real_provider:
        llm_sig = (
            mod,
            type(llm).__name__,
            getattr(llm, "model", ""),
            getattr(llm, "model_id", ""),
        )
        return True, (text_hash, tax_sig, llm_sig)
    return False, (text_hash, tax_sig)


def _cache_get(clean_text: str, taxonomy: Taxonomy, llm: object) -> dict | None:
    is_real, key = _cache_parts(clean_text, taxonomy, llm)
    if is_real:
        cached = _EXTRACTION_CACHE.get(key)
        if cached is not None:
            _EXTRACTION_CACHE.move_to_end(key)
            return copy.deepcopy(cached)
        return None
    try:
        bucket = _INSTANCE_EXTRACTION_CACHE.get(llm)
    except TypeError:
        return None
    if bucket is None:
        return None
    cached = bucket.get(key)
    if cached is not None:
        bucket.move_to_end(key)
        return copy.deepcopy(cached)
    return None


def _cache_put(clean_text: str, taxonomy: Taxonomy, llm: object, result: dict) -> None:
    is_real, key = _cache_parts(clean_text, taxonomy, llm)
    if is_real:
        _EXTRACTION_CACHE[key] = copy.deepcopy(result)
        if len(_EXTRACTION_CACHE) > _EXTRACTION_CACHE_MAX:
            _EXTRACTION_CACHE.popitem(last=False)
        return
    try:
        bucket = _INSTANCE_EXTRACTION_CACHE.get(llm)
        if bucket is None:
            bucket = OrderedDict()
            _INSTANCE_EXTRACTION_CACHE[llm] = bucket
    except TypeError:
        return
    bucket[key] = copy.deepcopy(result)
    if len(bucket) > _EXTRACTION_CACHE_MAX:
        bucket.popitem(last=False)


def _call_llm_deterministic(llm: object, system: str, user: str) -> str:
    """Invoke ``llm.complete`` with ``temperature=0.0`` when the signature allows it."""
    complete = getattr(llm, "complete")
    kwargs: dict = {"system": system, "user": user, "json_mode": True}
    try:
        sig = inspect.signature(complete)
        params = sig.parameters
        if "temperature" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            kwargs["temperature"] = 0.0
    except (TypeError, ValueError):
        pass
    try:
        return complete(**kwargs)
    except TypeError:
        kwargs.pop("temperature", None)
        return complete(**kwargs)


def _contextual_extract_resume(text: str, taxonomy: Taxonomy) -> dict:
    """Deterministic fallback: word-boundary alias scan + section-aware contextual levels."""
    skills: dict[str, dict] = {}
    if not text.strip():
        return {"profile": _empty_profile(), "skills": {}, "ai_parsed": False}

    sec_map = _line_sections_map(text)
    for skill in sorted(taxonomy.all(), key=lambda s: s.id):
        lines = _skill_lines(skill, text, taxonomy)
        if not lines:
            continue
        skills[skill.id] = {
            "skill_id": skill.id,
            "level": _contextual_level(lines, sections=sec_map),
            "mentions": max(1, len(lines)),
            "evidence": list(lines),
        }
    return {
        "profile": _contextual_profile(text),
        "skills": skills,
        "ai_parsed": False,
    }


def extract_resume_with_ai(text: str, taxonomy: Taxonomy, llm: object | None = None) -> dict:
    """Parse a resume into ``{profile, skills, ai_parsed}``.

    When ``llm`` is provided the model is called once with
    ``EXTRACT_RESUME_SYSTEM`` and a ``{"resume_text", "taxonomy"}`` payload.
    Every returned skill is resolved against the taxonomy and must be grounded
    in the resume text (alias/name match or evidence substring), otherwise it is
    dropped as a hallucination. Explicitly mentioned skills the model omitted
    are merged back with a contextual level.

    If ``llm`` is ``None``, ``llm.complete`` raises, or the response is not
    valid JSON, the function logs a warning and falls back to deterministic
    contextual extraction - resume uploads never fail because of the LLM.
    """
    if not isinstance(text, str) or not text.strip():
        return {"profile": _empty_profile(), "skills": {}, "ai_parsed": False}

    clean_text, _ = sanitize_untrusted_text(text, source="resume")
    if llm is None:
        logger.warning("extract_resume_with_ai: no LLM configured; using contextual fallback")
        return _contextual_extract_resume(clean_text, taxonomy)

    cached = _cache_get(clean_text, taxonomy, llm)
    if cached is not None:
        return cached

    payload = json.dumps({"resume_text": clean_text, "taxonomy": _taxonomy_payload(taxonomy)})
    try:
        raw = _call_llm_deterministic(llm, system=EXTRACT_RESUME_SYSTEM, user=payload)
    except Exception as exc:  # noqa: BLE001 - resilience contract: never fail the upload
        logger.warning("extract_resume_with_ai: LLM call failed (%s); falling back", exc)
        return _contextual_extract_resume(text, taxonomy)

    parsed = _safe_json(raw)
    if parsed is None:
        logger.warning("extract_resume_with_ai: LLM returned invalid JSON; falling back")
        return _contextual_extract_resume(text, taxonomy)

    sec_map = _line_sections_map(text)
    profile = _normalize_profile(parsed.get("profile"))
    skills: dict[str, dict] = {}
    raw_skills = parsed.get("skills")
    if not isinstance(raw_skills, list):
        raw_skills = []

    for entry in raw_skills:
        if not isinstance(entry, dict):
            continue
        skill = _resolve_skill(entry.get("skill_id"), taxonomy)
        if skill is None:
            continue  # not in taxonomy -> reject (FK-safe)
        evidence = _normalize_evidence(entry.get("evidence"), text)
        if not _grounded(skill, text, evidence, taxonomy):
            continue  # ungrounded -> reject (hallucination guard)
        lines = _skill_lines(skill, text, taxonomy)
        evidence = evidence or lines
        if not evidence:
            continue
        mentions = _normalize_mentions(entry.get("mentions"), evidence, lines)
        anchor_lines = evidence or lines
        ctx_level = _contextual_level(anchor_lines, sections=sec_map)
        level = _reconcile_level(entry.get("level"), ctx_level, anchor_lines, mentions)
        skills[skill.id] = {
            "skill_id": skill.id,
            "level": level,
            "mentions": mentions,
            "evidence": evidence,
        }

    # Merge explicit taxonomy skills the LLM omitted so they are never lost.
    for skill in sorted(taxonomy.all(), key=lambda s: s.id):
        if skill.id in skills:
            continue
        lines = _skill_lines(skill, text, taxonomy)
        if not lines:
            continue
        skills[skill.id] = {
            "skill_id": skill.id,
            "level": _contextual_level(lines, sections=sec_map),
            "mentions": max(1, len(lines)),
            "evidence": list(lines),
        }

    # Backfill any remaining blank fields from the contextual pass. The inner
    # per-field check makes this a no-op for keys the LLM already set, so no
    # outer gate is needed - any non-trivial condition would miss some
    # filled/empty field combination (regression-pinned by Test-Workhorse).
    ctx_profile = _contextual_profile(text)
    for key in ("name", "branch", "year", "gpa", "summary"):
        if not profile[key] and ctx_profile[key]:
            profile[key] = ctx_profile[key]

    ordered_skills = {sid: skills[sid] for sid in sorted(skills)}
    result = {"profile": profile, "skills": ordered_skills, "ai_parsed": True}
    _cache_put(clean_text, taxonomy, llm, result)
    return result


def extract_skills_with_llm(
    text: str, taxonomy: Taxonomy, llm: object | None = None
) -> dict[str, dict]:
    """Return ``{skill_id: skill_record}`` from the AI resume extraction."""
    return extract_resume_with_ai(text, taxonomy, llm)["skills"]


def extract_skills(
    text: str, taxonomy: Taxonomy, llm: object | None = None
) -> dict[str, list[str]]:
    """Return ``{skill_id: [evidence lines]}`` (legacy contract).

    With ``llm`` this delegates to the AI engine and maps each skill record to
    its evidence list; without it the original word-boundary alias scan runs
    unchanged for 100% backward compatibility.
    """
    if llm is not None:
        return {
            skill_id: list(record.get("evidence") or [])
            for skill_id, record in extract_resume_with_ai(text, taxonomy, llm)["skills"].items()
        }
    found: dict[str, list[str]] = {}
    for line in text.splitlines():
        for alias, skill_id in taxonomy.alias_map.items():
            if _line_matches_alias(line, alias):
                bucket = found.setdefault(skill_id, [])
                if line.strip() not in bucket:
                    bucket.append(line.strip())
    return found


ADVANCED_HINT = ("advanced", "expert", "proficient")
INTERMEDIATE_HINT = ("intermediate", "experienced", "worked with")


def level_for(
    mentions: int,
    evidence_lines: list[str] | None = None,
    *,
    llm: object | None = None,
    skill_id: str | None = None,
    resume_text: str | None = None,
    taxonomy: Taxonomy | None = None,
) -> str:
    """Proficiency level for a skill.

    Backward compatible with ``level_for(mentions, lines)`` and the
    ``level_for(5) == "advanced"`` / ``level_for(2) == "intermediate"`` /
    ``level_for(1) == "basic"`` unit-test contract.

    When ``llm`` and a resume context (``resume_text`` or ``evidence_lines``)
    are supplied, the AI contextual assessment
    (``extract_resume_with_ai``) decides the level; otherwise deterministic
    contextual signals in ``evidence_lines`` are evaluated.
    """
    if llm is not None and taxonomy is not None and (resume_text or evidence_lines):
        source = resume_text if (resume_text or "").strip() else "\n".join(evidence_lines or [])
        parsed = extract_resume_with_ai(source, taxonomy, llm)
        record = parsed["skills"].get(skill_id) if skill_id else None
        if record is not None:
            return _normalize_level(record.get("level"))

    lines = [str(ln) for ln in (evidence_lines or []) if str(ln).strip()]
    joined = " ".join(lines).lower()

    if mentions >= 4 or any(h in joined for h in ADVANCED_HINT):
        return "advanced"
    if mentions >= 2 or any(h in joined for h in INTERMEDIATE_HINT):
        return "intermediate"

    # Contextual signals (no static keyword trigger): a lone mention that shows
    # quantified impact / engineering depth in its evidence still counts.
    if lines and _contextual_level(lines) in ("advanced", "intermediate"):
        return _contextual_level(lines)
    return "basic"
