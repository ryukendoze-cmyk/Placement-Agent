"""Resume X-Ray: deterministic (no-LLM) ATS readiness scoring plus a grounded
bullet improver.

`xray_resume` is pure and cheap so it can back the profile KPI cards and a read-only
endpoint. `improve_weak_bullets` calls the LLM but is strictly constrained to only
reference skills that are actually present in the resume (zero-hallucination).
"""
import json
import re

from ..skills.extract import extract_skills

SECTION_PATTERNS = {
    "education": re.compile(r"educat|academic|qualifications?|degree|university|college", re.I),
    "projects": re.compile(r"project", re.I),
    "experience": re.compile(r"experience|internship|work history|professional", re.I),
    "achievements": re.compile(r"achievement|award|honou?rs?|certif|accomplishment", re.I),
}
URL_RE = re.compile(r"(https?://\S+|www\.\S+|github\.com/\S+|linkedin\.in/\S+|\.github\.io/\S+)", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\-\s]{7,}\d)(?!\d)")
BULLET_MARK = re.compile(r"^[-*•▪◦\u2022]|\s*\d+[.)]\s+")
SKILL_LIST_RE = re.compile(r"^[A-Za-z0-9#+.\s]+:\s*([A-Za-z0-9#.\s]+,\s*)+[A-Za-z0-9#.\s]+$")
METRIC_RE = re.compile(r"\d+(\.\d+)?\s*(%|percent|ms|k\b|lakh|crore|requests?|users?|gb|mb|tb|qps|faster|times?|x\b)", re.I)

ACTION_VERBS = {
    "built", "developed", "designed", "implemented", "engineered", "architected",
    "created", "led", "launched", "shipped", "optimized", "reduced", "improved",
    "automated", "migrated", "scaled", "delivered", "streamlined", "maintained",
    "refactored", "deployed", "orchestrated", "spearheaded", "accelerated",
    "mentored", "presented", "analyzed", "modeled", "integrated", "solved",
}

IMPROVE_BULLETS_SYSTEM = (
    "TASK: improve_bullets\n"
    "You are an expert resume coach. Rewrite the given weak resume bullets into strong, "
    "STAR-method, impact-focused bullets. You MUST only reference skills and projects that "
    "appear in 'available_skills' or the original bullet text - never invent new skills, "
    "tools, employers, or metrics that are not already present. Respond ONLY with valid JSON: "
    '{"improvements": [{"improved": str, "reason": str}, ...]} in the same order as the '
    "input bullets (up to 3)."
)


def _safe_json(raw: str):
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")).strip()
    try:
        return json.loads(cleaned or raw)
    except (ValueError, TypeError):
        return None


def _is_header(ln: str) -> bool:
    return len(ln) <= 40 and any(p.search(ln) for p in SECTION_PATTERNS.values())


def _is_contact(ln: str) -> bool:
    return len(ln) < 100 and bool(EMAIL_RE.search(ln) or PHONE_RE.search(ln) or URL_RE.search(ln))


_LEADING_CLEAN = re.compile(r"^[-*•▪◦\u2022]+\s*|\s*\d+[.)]\s*")


def _leading_word(ln: str) -> str:
    """First alphabetic word of a line, ignoring a bullet marker/number."""
    cleaned = _LEADING_CLEAN.sub("", ln.strip()).strip()
    for tok in cleaned.split():
        if tok.isalpha():
            return tok.lower()
    return ""


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _bullets(text: str) -> list[str]:
    out = []
    for ln in _lines(text):
        if BULLET_MARK.match(ln) or (len(ln) >= 20 and not _is_header(ln) and not _is_contact(ln)):
            out.append(ln)
    return out


def _sections(text: str) -> dict[str, bool]:
    low = text.lower()
    return {name: bool(p.search(low)) for name, p in SECTION_PATTERNS.items()}


def xray_resume(text: str, taxonomy=None) -> dict:
    """Deterministic ATS readiness analysis of resume text."""
    lines = _lines(text)
    bullets = _bullets(text)
    sections = _sections(text)
    links = sorted({m for ln in lines for m in URL_RE.findall(ln)})

    quantified = [b for b in bullets if METRIC_RE.search(b)]
    verb_bullets = [b for b in bullets if _leading_word(b) in ACTION_VERBS]

    impact = round(100 * len(quantified) / len(bullets)) if bullets else 0
    verb = round(100 * len(verb_bullets) / len(bullets)) if bullets else 0

    distinct_skills = 0
    total_evidence = 0
    if taxonomy is not None:
        found = extract_skills(text, taxonomy)
        distinct_skills = len(found)
        total_evidence = sum(len(v) for v in found.values())
    density = min(100, round(100 * distinct_skills / 12) if distinct_skills else 0)
    completeness = round(100 * sum(sections.values()) / len(SECTION_PATTERNS))

    ats = round(0.30 * impact + 0.25 * verb + 0.25 * density + 0.20 * completeness)
    ats = max(0, min(100, ats))

    return {
        "ats_score": ats,
        "sub_scores": {
            "impact_quantification": impact,
            "action_verb_strength": verb,
            "skill_density": density,
            "section_completeness": completeness,
        },
        "sections": sections,
        "links": links,
        "metrics": {
            "bullet_count": len(bullets),
            "quantified_bullets": len(quantified),
            "action_verb_bullets": len(verb_bullets),
            "distinct_skills": distinct_skills,
            "evidence_lines": total_evidence,
        },
    }


def pick_weak_bullets(text: str, taxonomy=None, k: int = 3) -> list[str]:
    """Up to `k` weakest bullets: skill-bearing preferred, non-quantified, no strong
    leading verb, not a header/contact/skill-list, 25-300 chars."""
    candidates = []
    for b in _bullets(text):
        if not (25 <= len(b) <= 300):
            continue
        if _is_header(b) or _is_contact(b) or SKILL_LIST_RE.match(b):
            continue
        if METRIC_RE.search(b) or _leading_word(b) in ACTION_VERBS:
            continue
        candidates.append(b)
    def _key(b):
        skill_bearing = 0 if (taxonomy is not None and extract_skills(b, taxonomy)) else 1
        return (skill_bearing, len(b))
    candidates.sort(key=_key)
    return candidates[:k]


def improve_weak_bullets(text: str, present_skills: list[str], llm, *, taxonomy=None) -> list[dict]:
    """Grounded STAR rewrites of the top-3 weakest bullets using ONLY present skills."""
    weak = pick_weak_bullets(text, taxonomy)
    if not weak:
        return []
    payload = json.dumps({"bullets": weak, "available_skills": list(present_skills or [])})
    parsed = _safe_json(llm.complete(system=IMPROVE_BULLETS_SYSTEM, user=payload, json_mode=True))
    improvements: list[dict] = []
    if isinstance(parsed, dict):
        for i, item in enumerate(parsed.get("improvements", [])):
            if i >= len(weak) or not isinstance(item, dict):
                continue
            improvements.append({
                "original": weak[i],
                "improved": str(item.get("improved", "")),
                "reason": str(item.get("reason", "")),
            })
    return improvements[:3]
