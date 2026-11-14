"""Pure-Python deterministic JD parser (fallback and hybrid-merge baseline)."""
from __future__ import annotations

import re
from typing import Any

from app.skills.extract import _alias_variants
from app.taxonomy.branches import normalize_branch
from app.taxonomy.taxonomy import Skill, Taxonomy

from .policy import apply_policy_and_defaults

CANONICAL_BRANCHES = {"CSE", "IT", "ECE", "EE", "ME", "CE", "AI&DS", "CHE", "BT"}

_UNDISCLOSED_RE = re.compile(
    r"\b(?:not\s+disclosed|undisclosed|as\s+per\s+(?:industry|company)\s+standards?|"
    r"competitive\s*(?:salary|compensation|package|ctc)?|"
    r"commensurate\s+with\s+experience|best\s+in\s+(?:the\s+)?industry|negotiable)\b",
    re.IGNORECASE,
)

_STIPEND_RE = re.compile(
    r"(?i)(?:stipend\s*[:\-]?\s*(?:₹|rs\.?|inr)?\s*|"
    r"(?:₹|rs\.?|inr)\s*(?=\d[\d,]*(?:\.\d+)?\s*(?:k)?\s*(?:/\s*(?:mo|month)|per\s+month|monthly)))"
    r"(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:(?:/\s*(?:mo|month)|per\s+month|monthly|p\.?m\.?))?",
)
_STIPEND_INLINE_RE = re.compile(
    r"(?i)(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:/\s*(?:mo|month)|per\s+month|monthly|p\.?m\.?)"
)

_CTC_RANGE_LPA_RE = re.compile(
    r"(?i)(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*"
    r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?\s*(?:per\s*annum|p\.?a\.?)|l\b)"
)
_CTC_SINGLE_LPA_RE = re.compile(
    r"(?i)(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?\s*(?:per\s*annum|p\.?a\.?))"
)
_CTC_SHORT_L_RE = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*L(?![A-Za-z0-9])"
)
_CTC_RANGE_INR_RE = re.compile(
    r"(?i)(?:₹|rs\.?|inr)\s*(\d[\d,]{5,})\s*(?:-|–|to)\s*(?:₹|rs\.?|inr)?\s*(\d[\d,]{5,})"
)
_CTC_SINGLE_INR_RE = re.compile(
    r"(?i)(?:₹|rs\.?|inr)\s*(\d{1,3}(?:,\d{2,3})+|\d{6,8})(?!\s*(?:/\s*(?:mo|month)|per\s+month|monthly))"
)

_CGPA_RE_LIST = (
    re.compile(r"(?i)\b(?:min(?:imum)?\.?\s*)?(?:cgpa|gpa)\s*(?:>=|≥|:|of|is|at\s*least)?\s*(\d+(?:\.\d{1,2})?)\s*(?:\+|and\s+above|or\s+above)?"),
    re.compile(r"(?i)\b(?:min(?:imum)?\.?|at\s*least|>=|≥)\s*(\d+(?:\.\d{1,2})?)\s*(?:cgpa|gpa)\b"),
    re.compile(r"(?i)\b(\d+(?:\.\d{1,2})?)\s*\+?\s*(?:cgpa|gpa)\s*(?:and\s+above|or\s+above|or\s+higher|min(?:imum)?)?\b"),
)
_PERCENT_RE = re.compile(
    r"(?i)(?:\b\d{2}(?:\.\d+)?\s*%\s*(?:throughout|aggregate|in\s+10th|in\s+12th|in\s+b\.?tech|or\s+above|min(?:imum)?)|\bmin(?:imum)?\s+\d{2}(?:\.\d+)?\s*%)"
)

_ZERO_BACKLOG_RE = re.compile(
    r"(?i)\b(?:no\s+(?:active\s+|live\s+|standing\s+)?backlogs?|zero\s+(?:active\s+)?backlogs?|"
    r"0\s+(?:active\s+)?backlogs?|no\s+(?:standing\s+)?arrears)\b"
)
_MAX_BACKLOG_RE = re.compile(
    r"(?i)\b(?:max(?:imum)?|up\s*to|at\s*most|<=|≤)\s*(\d+)\s*(?:active\s+)?backlogs?\b|"
    r"\b(\d+)\s*(?:active\s+)?backlogs?\s*(?:allowed|max(?:imum)?)\b"
)

_ALL_BRANCHES_RE = re.compile(
    r"(?i)\b(?:all\s+branches|open\s+to\s+all(?:\s+branches|\s+streams)?|"
    r"any\s+(?:engineering\s+)?branch(?:es)?|all\s+streams)\b"
)
_BRANCH_LINE_RE = re.compile(
    r"(?i)\b(?:branch(?:es)?|eligible\s+(?:degrees?|streams?|branches?)|b\.?tech|b\.?e\.?|degree|stream)\b"
)

_GRAD_YEAR_RE = re.compile(
    r"(?i)\b(?:(20[23]\d)\s*(?:batch|passout|pass-out|graduates?|graduating)|"
    r"(?:batch\s*(?:of)?|class\s*(?:of)?|graduating\s*(?:in|year)?|passout\s*(?:of|in)?)\s*[:\-]?\s*(20[23]\d))\b"
)

_WORK_MODE_RE = re.compile(
    r"(?i)\b(?:work\s*mode\s*[:\-]\s*(on-?site|remote|hybrid)|"
    r"(on-?site|onsite|in[- ]office|work\s+from\s+office|hybrid|remote|work\s+from\s+home|wfh))\b"
)

_INTERN_RE = re.compile(r"(?i)\b(?:intern(?:ship)?|summer\s+intern|winter\s+intern|trainee\s+intern)\b")
_FRESHER_RE = re.compile(r"(?i)\b(?:fresher|full[- ]?time|fte|graduate\s+engineer|entry[- ]?level)\b")

_COMPANY_LABEL_RE = re.compile(
    r"(?i)^(?:company(?:\s+name)?|organi[sz]ation|employer|recruiter)\s*[:\-]\s*(.+)$"
)
_ROLE_LABEL_RE = re.compile(
    r"(?i)^(?:role(?:\s+title)?|position|job\s+title|designation|profile|hiring\s+for)\s*[:\-]\s*(.+)$"
)
_LOCATION_LABEL_RE = re.compile(
    r"(?i)^(?:(?:job|work|base)?\s*location|place)\s*[:\-]\s*(.+)$"
)

_KNOWN_CITIES = (
    "Bengaluru", "Bangalore", "Hyderabad", "Pune", "Gurugram", "Gurgaon",
    "Noida", "Delhi NCR", "New Delhi", "Delhi", "Mumbai", "Chennai",
    "Jaipur", "Kolkata", "Ahmedabad", "Chandigarh", "Indore", "Kochi",
)

_ROLE_KEYWORDS_RE = re.compile(
    r"(?i)\b(?:software\s+(?:engineer|developer)|sde(?:\s*[-–]?\s*(?:1|i|intern))?|"
    r"backend\s+(?:engineer|developer)|frontend\s+(?:engineer|developer)|"
    r"full\s*stack\s+(?:engineer|developer)|data\s+(?:scientist|analyst|engineer)|"
    r"ml\s+engineer|machine\s+learning\s+engineer|ai\s+engineer|"
    r"devops\s+engineer|cloud\s+engineer|qa\s+engineer|product\s+analyst|"
    r"associate\s+engineer|graduate\s+engineer\s+trainee|systems?\s+engineer|"
    r"summer\s+intern|engineering\s+intern|tech(?:nical)?\s+intern)\b"
)

_PREFERRED_HEADER_RE = re.compile(
    r"(?i)^(?:#+\s*)?(?:good\s+to\s+have|nice\s+to\s+have|preferred(?:\s+skills|\s+qualifications)?|"
    r"bonus(?:\s+skills|\s+points)?|desirable(?:\s+skills)?|optional(?:\s+skills)?|"
    r"plus|added\s+advantage)\b"
)
_REQUIRED_HEADER_RE = re.compile(
    r"(?i)^(?:#+\s*)?(?:must\s+have|required(?:\s+skills|\s+qualifications)?|"
    r"mandatory(?:\s+skills)?|essential(?:\s+skills)?|core\s+skills|"
    r"technical\s+skills|skills\s+required|key\s+skills|tech\s+stack|"
    r"requirements|qualifications|skills)\b"
)
_SKILLS_CONTEXT_RE = re.compile(
    r"(?i)\b(?:skills?|tech\s*stack|technologies|tools|must\s+have|good\s+to\s+have|"
    r"nice\s+to\s+have|preferred|required|proficiency\s+in|experience\s+(?:in|with)|"
    r"knowledge\s+of|familiarity\s+with|bonus)\b"
)

_NON_SKILL_WORDS = {
    "skills", "required", "preferred", "good to have", "nice to have", "must have",
    "bonus", "optional", "mandatory", "essential", "experience", "knowledge",
    "proficiency", "familiarity", "strong", "solid", "understanding", "ability",
    "b.tech", "btech", "b.e", "be", "m.tech", "mca", "cgpa", "gpa", "ctc", "lpa",
    "stipend", "fresher", "internship", "remote", "hybrid", "on-site", "onsite",
    "location", "company", "role", "position", "eligibility", "branches", "batch",
    "requirements", "responsibilities", "qualifications", "benefits", "perks",
    "and", "or", "with", "in", "of", "for", "the", "a", "an", "to", "etc",
    "hands-on", "hands on", "problem-solving", "communication", "teamwork",
}


def _parse_inr_number(raw: str, has_k: bool = False) -> int:
    val = float(raw.replace(",", "").strip())
    if has_k:
        val *= 1000.0
    return round(val)


def _skill_boundary_re(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.lower())
    ends_special = alias.lower().endswith(("+", "#"))
    prefix = r"(?<![a-z0-9\-])"
    suffix = r"(?![a-z0-9\-])" if ends_special else r"(?![a-z0-9+\-#])"
    return re.compile(f"{prefix}{escaped}{suffix}")


def _extract_compound_unmapped_tokens(
    text: str,
    taxonomy: Taxonomy,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Identify tech-looking compound/hyphenated/plus tokens not in taxonomy.

    Returns (unmapped_mentions, masked_spans) so compound tokens like
    ``Blockchain-Quantum-Rust++`` or ``Kotlin Multiplatform`` are never
    partially decomposed into taxonomy skills (e.g., ``rust`` or ``kotlin``).
    """
    unmapped: list[str] = []
    seen_low: set[str] = set()
    spans: list[tuple[int, int]] = []

    def _add(tok: str, span: tuple[int, int]) -> None:
        clean = tok.strip(".,;:()[]{}\"' ")
        low = clean.lower()
        if not clean or len(clean) < 2 or low in _NON_SKILL_WORDS:
            return
        if taxonomy.resolve(low) is not None or taxonomy.get_opt(low) is not None:
            return
        if low not in seen_low:
            seen_low.add(low)
            unmapped.append(clean)
        spans.append(span)

    # 1. Hyphenated/symbol-suffixed compound tech tokens like "Blockchain-Quantum-Rust++"
    compound_re = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+){1,}(?:\+\+|#)?|\b[A-Za-z][A-Za-z0-9]{1,}\+\+")
    for m in compound_re.finditer(text):
        tok = m.group(0)
        low = tok.lower()
        if low in ("on-site", "full-time", "part-time", "e-commerce", "real-time", "end-to-end",
                   "pass-out", "well-versed", "hands-on", "self-motivated", "cross-functional",
                   "problem-solving", "object-oriented", "test-driven", "event-driven",
                   "high-level", "low-level", "entry-level", "mid-level", "state-of-the-art",
                   "multi-threaded", "ci-cd", "ai-ds", "b-tech", "non-cse"):
            continue
        if taxonomy.resolve(low) is not None:
            continue
        # Only treat as tech-looking if capitalized parts or ++/# or inside skills context
        has_caps = sum(1 for part in tok.split("-") if part and part[0].isupper()) >= 2
        if has_caps or "++" in tok or "#" in tok:
            _add(tok, m.span())

    # 2. Multi-word capitalized tech phrases inside skills lines (e.g., "Kotlin Multiplatform")
    for line_m in re.finditer(r"[^\n]+", text):
        line = line_m.group(0)
        if not (_SKILLS_CONTEXT_RE.search(line) or line.lstrip().startswith("- ")):
            continue
        # Check comma/semicolon/slash separated segments
        offset = line_m.start()
        for seg_m in re.finditer(r"[^,;|/()\n]+", line):
            seg = seg_m.group(0).strip()
            # Strip leading bullet or header label
            seg_clean = re.sub(r"^(?:-\s+|[^:]{1,30}:\s*)", "", seg).strip()
            if not seg_clean or len(seg_clean) > 48:
                continue
            words = seg_clean.split()
            if 2 <= len(words) <= 3 and all(w[0].isupper() for w in words if w[0].isalpha()):
                low = seg_clean.lower()
                if low in _NON_SKILL_WORDS or taxonomy.resolve(low) is not None:
                    continue
                # Skip plain section/company/city phrases
                if any(w.lower() in _NON_SKILL_WORDS for w in words):
                    continue
                if any(city.lower() == low for city in _KNOWN_CITIES):
                    continue
                # If at least one word is tech-like (in taxonomy or ends with JS/DB/UI/ML/AI/etc.)
                # and the full multi-word phrase is NOT in taxonomy, capture as unmapped phrase
                if any(taxonomy.resolve(w.lower()) is not None for w in words) or any(
                    w.lower().endswith(("js", "db", "ml", "ai", "ql", "++", "platform", "kit", "sdk"))
                    for w in words
                ):
                    abs_start = offset + seg_m.start()
                    abs_end = offset + seg_m.end()
                    _add(seg_clean, (abs_start, abs_end))

    return unmapped, spans


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = " "
    return "".join(chars)


def _line_matches_skill(line: str, skill: Skill, taxonomy: Taxonomy, *, is_skills_line: bool) -> bool:
    low = line.lower()
    for variant in _alias_variants(skill, taxonomy):
        if variant == "go":
            # Guard against English verb "go" unless capitalized Go, "golang", or skills list
            if "golang" in low:
                return True
            if re.search(r"(?<![A-Za-z0-9\-])Go(?![A-Za-z0-9+\-#])", line):
                return True
            if is_skills_line and re.search(
                r"(?<![a-z0-9\-])go(?![a-z0-9+\-#])(?!\s+(?:to|through|beyond|live|ahead|back|on|into|for|with|above))",
                low,
            ):
                return True
            continue
        if variant == "c":
            # Require uppercase C (C++ and C# have their own skill entries)
            if re.search(r"(?<![A-Za-z0-9\-])C(?![A-Za-z0-9+\-#])", line):
                return True
            continue
        if _skill_boundary_re(variant).search(low):
            return True
    return False


def _extract_skills_split(
    text: str, taxonomy: Taxonomy
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """Extract (required_skills, preferred_skills, unmapped_mentions, evidence_map)."""
    unmapped, spans = _extract_compound_unmapped_tokens(text, taxonomy)
    masked = _mask_spans(text, spans)

    required: list[str] = []
    preferred: list[str] = []
    req_set: set[str] = set()
    pref_set: set[str] = set()
    ev: dict[str, str] = {}

    section = "neutral"
    for raw_line in masked.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Detect section headers
        if _PREFERRED_HEADER_RE.match(line):
            section = "preferred"
        elif _REQUIRED_HEADER_RE.match(line):
            section = "required"

        # Handle inline preferred marker inside a line
        segments: list[tuple[str, str]] = []
        inline_pref = re.search(r"(?i)\b(good\s+to\s+have|nice\s+to\s+have|preferred|bonus)\s*:", line)
        if inline_pref:
            before = line[: inline_pref.start()].strip()
            after = line[inline_pref.start() :].strip()
            if before:
                segments.append(("required" if section != "preferred" else "preferred", before))
            if after:
                segments.append(("preferred", after))
        else:
            segments.append((section, line))

        for seg_kind, seg in segments:
            is_skills_line = bool(
                _SKILLS_CONTEXT_RE.search(seg)
                or seg.startswith("- ")
                or seg_kind in ("required", "preferred")
            )
            for skill in taxonomy.all():
                if _line_matches_skill(seg, skill, taxonomy, is_skills_line=is_skills_line):
                    if seg_kind == "preferred":
                        if skill.id not in req_set and skill.id not in pref_set:
                            pref_set.add(skill.id)
                            preferred.append(skill.id)
                            ev.setdefault("preferred_skills", line)
                    else:
                        if skill.id not in req_set:
                            req_set.add(skill.id)
                            required.append(skill.id)
                            ev.setdefault("required_skills", line)
                        if skill.id in pref_set:
                            pref_set.discard(skill.id)
                            preferred = [x for x in preferred if x != skill.id]

        # Also catch isolated capitalized/camelCase/backticked tech tokens in explicit skills lines
        if is_skills_line and (":" in line or line.startswith("- ")):
            body = line.split(":", 1)[1] if ":" in line else line[2:]
            for part in re.split(r"[,;/|]", body):
                cand = part.strip(" .()[]{}\"'")
                if not cand or len(cand) < 2 or len(cand) > 40:
                    continue
                low = cand.lower()
                if low in _NON_SKILL_WORDS or low in {u.lower() for u in unmapped}:
                    continue
                if taxonomy.resolve(low) is not None or taxonomy.get_opt(low) is not None:
                    continue
                # Skip if it contains a matched taxonomy skill inside prose (like "Python scripting")
                if any(_skill_boundary_re(a).search(low) for a in taxonomy.alias_map):
                    continue
                if re.search(r"(?:\+\+|#|\.js\b|\.py\b|\.ts\b|[A-Z][a-z]+[A-Z])", cand):
                    unmapped.append(cand)

    # Deduplicate unmapped preserving order
    seen_u: set[str] = set()
    clean_unmapped: list[str] = []
    for item in unmapped:
        k = item.lower()
        if k not in seen_u:
            seen_u.add(k)
            clean_unmapped.append(item)

    return required, preferred, clean_unmapped, ev


def _extract_branches(text: str) -> tuple[list[str], str | None]:
    m_all = _ALL_BRANCHES_RE.search(text)
    if m_all:
        for ln in text.splitlines():
            if m_all.group(0).lower() in ln.lower():
                return ([], ln.strip())
        return ([], m_all.group(0))

    found: list[str] = []
    evidence: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not _BRANCH_LINE_RE.search(stripped) and not re.search(
            r"\b(?:CSE|ECE|AI&DS|AI/DS|AIML|EEE|IT|ME|CE)\b", stripped
        ):
            continue
        # Protect AI&DS / AI & DS / AI/DS before splitting on delimiters
        norm_line = re.sub(r"(?i)\bAI\s*(?:&|and|/)\s*DS\b", "AI_DS_TOKEN", stripped)
        norm_line = re.sub(r"(?i)\b(?:B\.?Tech|B\.?E\.?|M\.?Tech|M\.?E\.?)\b", " ", norm_line)
        norm_line = re.sub(
            r"(?i)\b(?:Computer\s+Science(?:\s*(?:&|and)\s*Engineering)?|"
            r"Information\s+Technology|"
            r"Electronics(?:\s*(?:&|and)\s*Communication(?:\s+Engineering)?)?|"
            r"Electrical(?:\s*(?:&|and)\s*Electronics(?:\s+Engineering)?)?|"
            r"Mechanical(?:\s+Engineering)?|"
            r"Civil(?:\s+Engineering)?|"
            r"Artificial\s+Intelligence(?:\s*(?:&|and)\s*Data\s+Science)?|"
            r"Data\s+Science|"
            r"Chemical(?:\s+Engineering)?|"
            r"Biotechnology)\b",
            lambda m: f" {normalize_branch(m.group(0))} ",
            norm_line,
        )
        tokens = re.split(r"[/,|;()\[\]]+|\band\b|\bor\b|\s+", norm_line)
        line_branches: list[str] = []
        for tok in tokens:
            t = tok.strip(" .:-")
            if not t:
                continue
            if t == "AI_DS_TOKEN":
                code = "AI&DS"
            else:
                # Only map short uppercase-ish codes or exact branch aliases (avoid "IT" in normal prose!)
                if t.lower() == "it" and t != "IT":
                    continue
                if t.lower() in ("me", "be", "ce", "ai", "ds", "cs", "ch") and t != t.upper():
                    continue
                code = normalize_branch(t)
            if code in CANONICAL_BRANCHES and code not in line_branches:
                line_branches.append(code)
        if line_branches:
            evidence = evidence or stripped
            for b in line_branches:
                if b not in found:
                    found.append(b)
    return (found, evidence)


def parse_jd_static(text: str, taxonomy: Taxonomy) -> dict[str, Any]:
    """Deterministic pure-Python campus JD parser."""
    raw_text = (text or "").strip()
    evidence: dict[str, str] = {}
    field_confidence: dict[str, float] = {}
    missing_flags: list[str] = []
    explicit: set[str] = set()

    # 1. Company, Role Title, Location
    company = ""
    role_title = ""
    location = ""
    for line in raw_text.splitlines():
        s = line.strip().lstrip("- ").strip()
        if not s:
            continue
        if not company:
            mc = _COMPANY_LABEL_RE.match(s)
            if mc:
                company = mc.group(1).strip()
                evidence["company"] = s
                field_confidence["company"] = 0.9
                explicit.add("company")
                continue
        if not role_title:
            mr = _ROLE_LABEL_RE.match(s)
            if mr:
                role_title = mr.group(1).strip()
                evidence["role_title"] = s
                field_confidence["role_title"] = 0.9
                explicit.add("role_title")
                continue
        if not location:
            ml = _LOCATION_LABEL_RE.match(s)
            if ml:
                location = ml.group(1).strip()
                evidence["location"] = s
                field_confidence["location"] = 0.9
                explicit.add("location")
                continue

    # Heuristic fallbacks for company, role_title, location
    if not company:
        m_hiring = re.search(r"\b([A-Z][A-Za-z0-9&.\- ]{1,40}?)\s+is\s+(?:hiring|looking\s+for|inviting)\b", raw_text)
        if m_hiring:
            company = m_hiring.group(1).strip()
            evidence["company"] = m_hiring.group(0)
            field_confidence["company"] = 0.6
            explicit.add("company")
        else:
            m_about = re.search(r"(?im)^(?:about|welcome\s+to)\s+([A-Z][A-Za-z0-9&.\- ]{1,40})$", raw_text)
            if m_about:
                company = m_about.group(1).strip()
                evidence["company"] = m_about.group(0).strip()
                field_confidence["company"] = 0.6
                explicit.add("company")

    if not role_title:
        for line in raw_text.splitlines()[:10]:
            s = line.strip().lstrip("- ").strip()
            mr = _ROLE_KEYWORDS_RE.search(s)
            if mr:
                role_title = mr.group(0).strip()
                evidence["role_title"] = s
                field_confidence["role_title"] = 0.6
                explicit.add("role_title")
                break

    if not company:
        # First non-label short header line (e.g. "Razorpay Campus Drive - SDE")
        for line in raw_text.splitlines()[:5]:
            s = line.strip().lstrip("- ").strip()
            if not s or len(s) > 65 or ":" in s:
                continue
            if role_title and role_title.lower() in s.lower():
                parts = re.split(r"\s*[-–|@]\s*", s)
                if len(parts) >= 2 and parts[0].strip():
                    company = parts[0].strip()
                    evidence["company"] = s
                    field_confidence["company"] = 0.6
                    explicit.add("company")
                    break

    if not location:
        for city in _KNOWN_CITIES:
            if re.search(rf"\b{re.escape(city)}\b", raw_text, re.IGNORECASE):
                location = city
                for ln in raw_text.splitlines():
                    if city.lower() in ln.lower():
                        evidence["location"] = ln.strip()
                        break
                field_confidence["location"] = 0.6
                explicit.add("location")
                break

    # 2. Work mode
    work_mode: str | None = None
    m_wm = _WORK_MODE_RE.search(raw_text)
    if m_wm:
        token = (m_wm.group(1) or m_wm.group(2) or "").lower()
        if "remote" in token or "home" in token or "wfh" in token:
            work_mode = "Remote"
        elif "hybrid" in token:
            work_mode = "Hybrid"
        else:
            work_mode = "On-site"
        evidence["work_mode"] = m_wm.group(0).strip()
        field_confidence["work_mode"] = 0.9
        explicit.add("work_mode")

    # 3. Stipend & CTC & salary_disclosed
    stipend_inr_per_month: int | None = None
    ctc_lpa_min: float | None = None
    ctc_lpa_max: float | None = None
    salary_disclosed = False

    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()

        if stipend_inr_per_month is None and ("stipend" in low or "/mo" in low or "per month" in low or "monthly" in low):
            ms = _STIPEND_RE.search(s) or _STIPEND_INLINE_RE.search(s)
            if ms:
                val = _parse_inr_number(ms.group(1), bool(ms.group(2)))
                if 1000 <= val <= 500000:
                    stipend_inr_per_month = val
                    evidence["stipend_inr_per_month"] = s
                    field_confidence["stipend_inr_per_month"] = 0.9
                    explicit.add("stipend_inr_per_month")
                    continue

        if ctc_lpa_max is None:
            m_rng = _CTC_RANGE_LPA_RE.search(s)
            if m_rng:
                lo, hi = float(m_rng.group(1)), float(m_rng.group(2))
                if 0.5 <= lo <= 200 and 0.5 <= hi <= 200:
                    ctc_lpa_min = min(lo, hi)
                    ctc_lpa_max = max(lo, hi)
                    salary_disclosed = True
                    evidence["ctc_lpa_max"] = s
                    field_confidence["ctc_lpa_max"] = 0.9
                    explicit.add("ctc_lpa_max")
                    continue

            m_sgl = _CTC_SINGLE_LPA_RE.search(s)
            if m_sgl:
                val = float(m_sgl.group(1))
                if 0.5 <= val <= 200:
                    ctc_lpa_min = val
                    ctc_lpa_max = val
                    salary_disclosed = True
                    evidence["ctc_lpa_max"] = s
                    field_confidence["ctc_lpa_max"] = 0.9
                    explicit.add("ctc_lpa_max")
                    continue

            m_inr_rng = _CTC_RANGE_INR_RE.search(s)
            if m_inr_rng and "stipend" not in low:
                lo = _parse_inr_number(m_inr_rng.group(1)) / 100000.0
                hi = _parse_inr_number(m_inr_rng.group(2)) / 100000.0
                if 0.5 <= lo <= 200 and 0.5 <= hi <= 200:
                    ctc_lpa_min = round(min(lo, hi), 2)
                    ctc_lpa_max = round(max(lo, hi), 2)
                    salary_disclosed = True
                    evidence["ctc_lpa_max"] = s
                    field_confidence["ctc_lpa_max"] = 0.9
                    explicit.add("ctc_lpa_max")
                    continue

            m_inr = _CTC_SINGLE_INR_RE.search(s)
            if m_inr and "stipend" not in low:
                val = round(_parse_inr_number(m_inr.group(1)) / 100000.0, 2)
                if 0.5 <= val <= 200:
                    ctc_lpa_min = val
                    ctc_lpa_max = val
                    salary_disclosed = True
                    evidence["ctc_lpa_max"] = s
                    field_confidence["ctc_lpa_max"] = 0.9
                    explicit.add("ctc_lpa_max")
                    continue

            m_short = _CTC_SHORT_L_RE.search(s)
            if m_short and "stipend" not in low:
                val = float(m_short.group(1))
                if 1.0 <= val <= 200:
                    ctc_lpa_min = val
                    ctc_lpa_max = val
                    salary_disclosed = True
                    evidence["ctc_lpa_max"] = s
                    field_confidence["ctc_lpa_max"] = 0.9
                    explicit.add("ctc_lpa_max")
                    continue

    if ctc_lpa_max is None:
        m_undisc = _UNDISCLOSED_RE.search(raw_text)
        if m_undisc:
            salary_disclosed = False
            for ln in raw_text.splitlines():
                if m_undisc.group(0).lower() in ln.lower():
                    evidence["salary_disclosed"] = ln.strip()
                    break
            field_confidence["ctc_lpa_max"] = 0.9
        else:
            field_confidence["ctc_lpa_max"] = 0.3

    # 4. Experience level
    experience_level: str | None = None
    if _INTERN_RE.search(role_title) or (
        _INTERN_RE.search(raw_text) and stipend_inr_per_month is not None and ctc_lpa_max is None
    ) or re.search(r"(?i)\b(?:role\s*type|engagement|employment\s*type)\s*[:\-]\s*internship\b", raw_text):
        experience_level = "Internship"
        evidence["experience_level"] = role_title or evidence.get("stipend_inr_per_month", "Internship")
        field_confidence["experience_level"] = 0.9
        explicit.add("experience_level")
    elif _FRESHER_RE.search(raw_text) or ctc_lpa_max is not None:
        experience_level = "Fresher"
        m_fr = _FRESHER_RE.search(raw_text)
        if m_fr:
            evidence["experience_level"] = m_fr.group(0)
            field_confidence["experience_level"] = 0.9
        else:
            field_confidence["experience_level"] = 0.6
        explicit.add("experience_level")

    # 5. Eligibility: CGPA, percentage warning, backlogs, branches, graduation_year
    min_cgpa: float | None = None
    for pat in _CGPA_RE_LIST:
        m_gpa = pat.search(raw_text)
        if m_gpa:
            val = float(m_gpa.group(1))
            if 1.0 <= val <= 10.0:
                min_cgpa = round(val, 2)
                for ln in raw_text.splitlines():
                    if m_gpa.group(0).lower() in ln.lower():
                        evidence["min_cgpa"] = ln.strip()
                        break
                field_confidence["min_cgpa"] = 0.9
                explicit.add("min_cgpa")
                break

    m_pct = _PERCENT_RE.search(raw_text)
    if m_pct:
        for ln in raw_text.splitlines():
            if m_pct.group(0).lower() in ln.lower():
                evidence["percentage_criteria"] = ln.strip()
                break
        missing_flags.append("percentage_cutoff_unconverted")

    max_active_backlogs: int | None = None
    if _ZERO_BACKLOG_RE.search(raw_text):
        max_active_backlogs = 0
        m_zb = _ZERO_BACKLOG_RE.search(raw_text)
        for ln in raw_text.splitlines():
            if m_zb and m_zb.group(0).lower() in ln.lower():
                evidence["max_active_backlogs"] = ln.strip()
                break
        field_confidence["max_active_backlogs"] = 0.9
        explicit.add("max_active_backlogs")
    else:
        m_mb = _MAX_BACKLOG_RE.search(raw_text)
        if m_mb:
            num = m_mb.group(1) or m_mb.group(2)
            if num is not None:
                max_active_backlogs = int(num)
                for ln in raw_text.splitlines():
                    if m_mb.group(0).lower() in ln.lower():
                        evidence["max_active_backlogs"] = ln.strip()
                        break
                field_confidence["max_active_backlogs"] = 0.9
                explicit.add("max_active_backlogs")

    allowed_branches, branch_ev = _extract_branches(raw_text)
    if branch_ev:
        evidence["allowed_branches"] = branch_ev
        field_confidence["allowed_branches"] = 0.9
        explicit.add("allowed_branches")

    graduation_year: int | None = None
    m_yr = _GRAD_YEAR_RE.search(raw_text)
    if m_yr:
        yr_str = m_yr.group(1) or m_yr.group(2)
        if yr_str:
            graduation_year = int(yr_str)
            for ln in raw_text.splitlines():
                if yr_str in ln:
                    evidence["graduation_year"] = ln.strip()
                    break
            field_confidence["graduation_year"] = 0.9
            explicit.add("graduation_year")

    # 6. Skills
    req_skills, pref_skills, unmapped_mentions, skill_ev = _extract_skills_split(raw_text, taxonomy)
    evidence.update(skill_ev)
    field_confidence["skills"] = 0.9 if (req_skills or pref_skills) else 0.3

    drive = {
        "company": company,
        "role_title": role_title,
        "location": location,
        "work_mode": work_mode,
        "ctc_lpa_min": ctc_lpa_min,
        "ctc_lpa_max": ctc_lpa_max,
        "salary_disclosed": salary_disclosed,
        "stipend_inr_per_month": stipend_inr_per_month,
        "policy_tier": None,
        "policy_tier_source": None,
        "experience_level": experience_level,
        "source_portal": "On-Campus (Local)",
        "source_badge": "On-Campus · JKLU",
        "eligibility": {
            "min_cgpa": min_cgpa,
            "allowed_branches": allowed_branches,
            "max_active_backlogs": max_active_backlogs,
            "graduation_year": graduation_year,
        },
        "required_skills": req_skills,
        "preferred_skills": pref_skills,
        "unmapped_skill_mentions": unmapped_mentions,
        "raw_jd_text": raw_text,
        "parser_metadata": {
            "engine_used": "static_fallback",
            "fallback_reason": None,
            "attempts": [],
            "confidence_score": 0.0,
            "field_confidence": field_confidence,
            "evidence": evidence,
            "missing_fields_flagged": missing_flags,
            "defaults_applied": [],
        },
        "_explicit_fields": sorted(explicit),
    }
    return apply_policy_and_defaults(drive, explicitly_found=explicit)
