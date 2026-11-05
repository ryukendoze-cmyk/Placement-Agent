import re
from dataclasses import dataclass

from ..taxonomy.branches import is_branch_eligible
from ..taxonomy.taxonomy import Taxonomy


@dataclass
class JobScore:
    job_id: str
    score: int                 # 0..100
    band: str                  # strong|moderate|developing|early
    matched_required: list[dict]
    missing_required: list[dict]
    matched_preferred: list[dict]
    missing_preferred: list[dict]
    semantic_sim: float
    proficiency_bonus: int = 0  # additive 0..10 applied when per-skill levels are supplied
    #: every required skill where student depth < required depth
    #: ({skill, current_level, target_level}) -> feeds the Upskilling Planner.
    proficiency_gaps: list[dict] = None  # type: ignore[assignment]
    #: -25 when a campus fresher is scored against a senior role (else 0).
    seniority_penalty: int = 0

    def __post_init__(self) -> None:
        if self.proficiency_gaps is None:
            self.proficiency_gaps = []


WEIGHTS = {"coverage": 0.6, "preferred": 0.2, "semantic": 0.2}

#: Freshers evaluated against a senior role lose these composite points.
SENIORITY_PENALTY = -25

#: senior -> Fresher -> Senior (ARCHITECTURE §19.2).
SENIOR_TITLE_RE = re.compile(
    r"\b("
    r"senior|sr\.?|lead|staff|principal|architect|manager|head|director|vp|specialist|consultant|expert|"
    r"(?:sde|engineer|developer|scientist|analyst|specialist|consultant|programmer)\s*[- ]*(?:iii|iv|v|vi|[3-9])|"
    r"(?:level|lvl|l|ic|e)[- ]*[3-9]|"
    r"(?:iii|iv|v|vi)"
    r")\b",
    re.IGNORECASE,
)
FRESHER_TITLE_RE = re.compile(
    r"\b(intern|internship|trainee|fresher|graduate\s+engineer|graduate\s+trainee|"
    r"get|entry[\s-]level|associate|junior|jr\.?|sde[\s-]?[1i]\b|analyst\s+1|engineer\s+1)\b",
    re.IGNORECASE,
)

# Proficiency adds a small, explainable bonus to the base score (capped so the
# total never exceeds 100). basic adds nothing; intermediate ~5; advanced ~10,
# scaled by the share of required skills the student actually matched.
PROFICIENCY_POINTS = {"basic": 0, "intermediate": 5, "advanced": 10}
LEVEL_ORDER = {"basic": 0, "intermediate": 1, "advanced": 2}


def band_for(score: int) -> str:
    return "strong" if score >= 80 else "moderate" if score >= 60 else "developing" if score >= 40 else "early"


def level_rank(level: str | None) -> int:
    return LEVEL_ORDER.get((level or "basic").lower(), 0)


def is_senior_job(job) -> bool:
    """True for Senior/Lead/Staff/Specialist titles OR >= 3 years experience (§19.2)."""
    if not job:
        return False
    title = str((job.get("title") if isinstance(job, dict) else getattr(job, "title", "")) or "")
    exp = (job.get("experience_years") if isinstance(job, dict) else getattr(job, "experience_years", 0)) or 0
    try:
        exp_val = float(exp)
    except (ValueError, TypeError):
        exp_val = 0.0
    if FRESHER_TITLE_RE.search(title) and exp_val < 3.0:
        return False
    return bool(SENIOR_TITLE_RE.search(title)) or exp_val >= 3.0


def is_fresher_job(job) -> bool:
    """True for entry-level roles suitable for freshers (ARCHITECTURE §21.2.2).
    - If is_senior_job -> False.
    - External scraped roles (job-ext-): ANY exp > 0 -> False.
    - Campus / local roles: exp > 1.0 -> False; 0 < exp <= 1 -> True.
    - If exp_val <= 0.0 -> True.
    """
    if not job:
        return False
    if (isinstance(job, dict) and "year" in job) or hasattr(job, "year"):
        return is_campus_fresher(job)
    if is_senior_job(job):
        return False
    exp = (job.get("experience_years") if isinstance(job, dict) else getattr(job, "experience_years", 0)) or 0
    try:
        exp_val = float(exp)
    except (ValueError, TypeError):
        exp_val = 0.0

    # Determine if this is an external scraped job
    job_id = str((job.get("id") if isinstance(job, dict) else getattr(job, "id", "")) or "")
    source = str((job.get("source") if isinstance(job, dict) else getattr(job, "source", "")) or "").strip().lower()
    is_external = bool(job_id.startswith("job-ext-") or (source and source != "local"))

    # STRICT RULE: For external scraped jobs, ANY experience > 0 disqualifies.
    # This is the critical fix — external jobs listing even 1 year experience
    # are NOT fresher roles.
    if is_external and exp_val > 0:
        return False

    # For campus/local jobs, allow up to 1 year as fresher
    if exp_val > 1.0:
        return False
    if exp_val <= 0.0:
        return True

    # 0.0 < exp_val <= 1.0 for campus/local roles -> fresher
    return True


def is_campus_fresher(student) -> bool:
    """Campus fresher: undergrad (year <= 4) or < 2 years experience (§19.3.2)."""
    year = getattr(student, "year", None)
    if year is not None and int(year) <= 4:
        return True
    exp = getattr(student, "years_experience", 0) or getattr(student, "experience_years", 0) or 0
    return float(exp) < 2


def target_depth_for(job) -> str:
    """Senior roles demand 'advanced', standard roles 'intermediate' (§19.3.1)."""
    return "advanced" if is_senior_job(job) else "intermediate"


def depth_weight(student_level: str | None, target: str) -> float:
    """Per-required-skill credit: 1.0 exact-or-higher, 0.4 one below, 0.1 two below."""
    have = level_rank(student_level)
    want = level_rank(target)
    gap = want - have
    if gap <= 0:
        return 1.0
    if gap == 1:
        return 0.4
    return 0.1


def higher_level(a: str | None, b: str | None) -> str:
    """Return whichever of the two levels is higher (used so a simulator never
    downgrades a skill the student already has)."""
    return a if level_rank(a) >= level_rank(b) else b


def job_tier(job) -> str:
    """Derive the JKLU placement tier from the top of the salary band:
    standard < 8 LPA, dream 8-15 LPA, super_dream >= 15 LPA."""
    salary = list(getattr(job, "salary_lpa", None) or [0, 0])
    upper = max(salary) if salary else 0
    if upper >= 15:
        return "super_dream"
    if upper >= 8:
        return "dream"
    return "standard"


def job_eligibility(student, job) -> tuple[bool, list[str]]:
    """Explainable eligibility gate (GPA cutoff + branch restriction + experience).
    Empty allowed_branches means open to all branches."""
    reasons: list[str] = []
    # §19.3.3 — campus freshers are not eligible for senior/experienced roles.
    if is_campus_fresher(student) and is_senior_job(job):
        reasons.append(
            "Role requires experienced professional (Senior title or 3+ years experience required)"
        )
    min_gpa = float(getattr(job, "min_gpa", 0.0) or 0.0)
    gpa = float(getattr(student, "gpa", 0.0) or 0.0)
    if min_gpa and gpa < min_gpa:
        reasons.append(f"GPA {gpa:.2f} is below the {min_gpa:.2f} cutoff for this role")
    allowed = [str(b).strip() for b in (getattr(job, "allowed_branches", None) or []) if str(b).strip()]
    branch = str(getattr(student, "branch", "") or "").strip()
    if allowed and not is_branch_eligible(branch, allowed):
        allowed_disp = sorted({str(b).strip().upper() for b in allowed if str(b).strip()})
        reasons.append(f"Branch '{branch.upper() or 'n/a'}' is not in the allowed branches ({', '.join(allowed_disp)})")
    return (len(reasons) == 0, reasons)


def _as_dict(skill) -> dict:
    return {"skill": skill.id, "name": skill.name, "category": skill.category}


def _with_evidence(skill, student_skills: dict[str, list[str]], levels: dict[str, str] | None) -> dict:
    d = _as_dict(skill)
    ev = student_skills.get(skill.id) or []
    d["evidence"] = ev[0] if ev else ""
    d["level"] = (levels or {}).get(skill.id, "basic")
    return d


def _proficiency_bonus(matched_req, req_count: int, levels: dict[str, str] | None) -> int:
    if not levels or req_count <= 0:
        return 0
    pts = sum(PROFICIENCY_POINTS.get(str(levels.get(s.id) or "basic").lower(), 0) for s in matched_req)
    return min(10, round(pts / req_count))


def score_job(student_skills: dict[str, list[str]], job, sim: float, taxonomy: Taxonomy,
              *, levels: dict[str, str] | None = None, student=None) -> JobScore:
    """score = 100 * (0.6*required_coverage + 0.2*preferred_ratio + 0.2*semantic_sim),
    plus a small proficiency bonus (0..10) when `levels` is supplied.

    Coverage is depth-weighted (§19.3.1): each required skill credits 1.0 at or
    above the target depth (advanced for senior roles, intermediate otherwise),
    0.4 one level below, 0.1 two below. When `levels` is None every present
    skill counts as meeting the target, so the score stays byte-for-byte equal
    to the original flat formula.

    When `student` is supplied and is a campus fresher scored against a senior
    role, -25 points are deducted (§19.3.2). Required skills the student has at
    a lower depth than the target are reported in `proficiency_gaps` (§19.3.4).

    Job skills that are not in the taxonomy are skipped (not a crash).
    """
    req = [s for s in (taxonomy.get_opt(x) for x in job.required_skills) if s is not None]
    pref = [s for s in (taxonomy.get_opt(x) for x in job.preferred_skills) if s is not None]
    matched_req = [s for s in req if s.id in student_skills]
    missing_req = [s for s in req if s.id not in student_skills]
    matched_pref = [s for s in pref if s.id in student_skills]
    missing_pref = [s for s in pref if s.id not in student_skills]

    # §19.3.1 — depth-weighted required coverage.
    target = target_depth_for(job)
    weights: list[float] = []
    proficiency_gaps: list[dict] = []
    # `levels is None` = the original flat formula: a present skill is assumed to
    # meet the target (byte-for-byte parity, §19.3.1 docstring). With levels
    # supplied, depth is graded against the target and any shortfall becomes a gap.
    for s in req:
        if s.id not in student_skills:
            # Absent entirely: no credit under the old formula either, and it is a
            # reportable gap (the planner turns gaps into modules).
            weights.append(0.0)
            proficiency_gaps.append({
                "skill": s.id, "current_level": "none",
                "target_level": target,
            })
            continue
        if levels is None:
            weights.append(1.0)
            continue
        have = str(levels.get(s.id) or "basic").lower()
        weights.append(depth_weight(have, target))
        if level_rank(have) < level_rank(target):
            proficiency_gaps.append({
                "skill": s.id, "current_level": have,
                "target_level": target,
            })
    coverage = (sum(weights) / len(req)) if req else 1.0
    pref_score = len(matched_pref) / len(pref) if pref else 0.0
    sim = max(0.0, min(1.0, sim))
    base = round(100 * (WEIGHTS["coverage"] * coverage + WEIGHTS["preferred"] * pref_score + WEIGHTS["semantic"] * sim))
    bonus = _proficiency_bonus(matched_req, len(req), levels)

    # §19.3.2 — experience mismatch penalty for campus freshers on senior roles.
    penalty = 0
    if student is not None and is_campus_fresher(student) and is_senior_job(job):
        penalty = SENIORITY_PENALTY

    score = max(0, min(100, base + bonus + penalty))
    return JobScore(
        job_id=job.id, score=score, band=band_for(score),
        matched_required=[_with_evidence(s, student_skills, levels) for s in matched_req],
        missing_required=[_as_dict(s) for s in missing_req],
        matched_preferred=[_as_dict(s) for s in matched_pref],
        missing_preferred=[_as_dict(s) for s in missing_pref],
        semantic_sim=round(sim, 3),
        proficiency_bonus=bonus,
        proficiency_gaps=proficiency_gaps,
        seniority_penalty=penalty,
    )
