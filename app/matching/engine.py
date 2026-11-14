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
    #: Matched required/preferred skills flagged unverified or needing reverification.
    unverified_skills: list[str] = None  # type: ignore[assignment]
    needs_reverification_skills: list[str] = None  # type: ignore[assignment]
    verification_statuses: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.proficiency_gaps is None:
            self.proficiency_gaps = []
        if self.unverified_skills is None:
            self.unverified_skills = []
        if self.needs_reverification_skills is None:
            self.needs_reverification_skills = []
        if self.verification_statuses is None:
            self.verification_statuses = {}


WEIGHTS = {"coverage": 0.6, "preferred": 0.2, "semantic": 0.2}

VERIFICATION_WEIGHTS = {
    "verified": 1.0,
    "claimed": 0.85,
    "needs_reverification": 0.5,
    "unverified": 0.2,
}

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
MID_LEVEL_TITLE_RE = re.compile(
    r"\b("
    r"mid[\s-]level|mid[\s-]senior|intermediate|"
    r"(?:sde|engineer|developer|scientist|analyst|programmer)\s*[- ]*(?:ii|2)\b|"
    r"(?:level|lvl|l|ic|e)[- ]*2\b"
    r")\b",
    re.IGNORECASE,
)
FRESHER_TITLE_RE = re.compile(
    r"\b(interns?|internship|trainees?|freshers?|graduate\s+engineer|graduate\s+trainee|"
    r"get|entry[\s-]level|associate|junior|jr\.?|sde[\s-]?[1i]\b|analyst\s+1|engineer\s+1)\b",
    re.IGNORECASE,
)

# Proficiency adds a small, explainable bonus to the base score (capped so the
# total never exceeds 100). basic adds nothing; intermediate ~5; advanced ~10,
# scaled by the share of required skills the student actually matched.
PROFICIENCY_POINTS = {"basic": 0, "intermediate": 5, "advanced": 10}
LEVEL_ORDER = {"basic": 0, "intermediate": 1, "advanced": 2}


def band_for(score: int) -> str:
    """Map readiness score 0..100 to band using SOP-05 min_coverage_pct as the developing floor."""
    from ..rag.policy_rag import load_sop_params

    sop05 = load_sop_params("SOP-05")
    developing_min = int(sop05.get("min_coverage_pct", 40))
    strong_min = max(80, developing_min)
    moderate_min = min(strong_min, max(60, developing_min))
    if score >= strong_min:
        return "strong"
    if score >= moderate_min:
        return "moderate"
    if score >= developing_min:
        return "developing"
    return "early"


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
    - If exp_val <= 0.0 -> True ONLY IF title/description/URL do not require experience.
    """
    if not job:
        return False
    if (isinstance(job, dict) and "year" in job) or hasattr(job, "year"):
        return is_campus_fresher(job)
    if is_senior_job(job):
        return False
    title = str((job.get("title") if isinstance(job, dict) else getattr(job, "title", "")) or "")
    if MID_LEVEL_TITLE_RE.search(title):
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

    # STRICT RULE: For external scraped jobs, ANY recorded experience > 0 disqualifies.
    if is_external and exp_val > 0:
        return False

    # For campus/local jobs, allow up to 1 year as fresher
    if not is_external and exp_val > 1.0:
        return False

    # CRITICAL FAILSAFE: If exp_val <= 0 (or campus <= 1.0), inspect title, description, and job_url.
    # A job whose text, URL, or title specifies required experience (e.g. "2-to-4-years", "4+ Years",
    # "2-4 Years of experience") MUST NEVER pass as a fresher role, even if raw scraper or DB left exp at 0.
    from app.scraper.normalizer import extract_experience_years
    desc = str((job.get("description") if isinstance(job, dict) else getattr(job, "description", "")) or "")
    url = str((job.get("job_url") if isinstance(job, dict) else getattr(job, "job_url", "")) or "")
    combined = f"{title} {desc} {url}".strip()

    if combined:
        extracted = extract_experience_years(combined)
        if is_external and extracted > 0:
            return False
        if not is_external and extracted > 1.0:
            return False

        if is_external and not FRESHER_TITLE_RE.search(title):
            m_range = re.search(r"(?i)\b(\d+)[- ]*to[- ]*(\d+)[- ]*(?:years?|yrs?|yoe)\b", combined)
            if m_range:
                try:
                    high = int(m_range.group(2))
                    if high >= 2:
                        return False
                except (ValueError, TypeError):
                    pass

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
    """Derive the JKLU placement tier dynamically from SOP-01 CTC thresholds."""
    from ..rag.policy_rag import load_sop_params

    sop01 = load_sop_params("SOP-01")
    super_dream_min = float(sop01.get("super_dream_min_lpa", 15.0))
    dream_min = float(sop01.get("dream_min_lpa", 8.0))
    salary = list(getattr(job, "salary_lpa", None) or [0, 0])
    upper = max(salary) if salary else 0
    if upper >= super_dream_min:
        return "super_dream"
    if upper >= dream_min:
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
    max_backlogs = getattr(job, "max_backlogs", None)
    if max_backlogs is not None:
        student_backlogs = int(getattr(student, "backlogs", 0) or 0)
        if student_backlogs > int(max_backlogs):
            reasons.append(
                f"Active backlogs ({student_backlogs}) exceed maximum allowed ({int(max_backlogs)}) per SOP-03"
            )
    return (len(reasons) == 0, reasons)


def _as_dict(skill) -> dict:
    return {"skill": skill.id, "name": skill.name, "category": skill.category}


def _with_evidence(
    skill,
    student_skills: dict[str, list[str]],
    levels: dict[str, str] | None,
    verification_statuses: dict[str, str] | None = None,
) -> dict:
    d = _as_dict(skill)
    ev = list(student_skills.get(skill.id) or [])
    resume_ev = [e for e in ev if not str(e).lower().startswith(("[interview ", "interview "))]
    interview_notes = [e for e in ev if str(e).lower().startswith(("[interview ", "interview "))]
    d["evidence"] = resume_ev[0] if resume_ev else (ev[0] if ev else "")
    if interview_notes:
        d["verification_note"] = interview_notes[-1]
    d["level"] = (levels or {}).get(skill.id, "basic")
    if verification_statuses is not None:
        status = str(verification_statuses.get(skill.id) or "claimed").lower()
        if status not in VERIFICATION_WEIGHTS:
            status = "claimed"
        d["verification_status"] = status
        if status == "unverified":
            d["effective_level"] = "basic"
    return d


def _preferred_with_status(skill, verification_statuses: dict[str, str] | None = None) -> dict:
    d = _as_dict(skill)
    if verification_statuses is not None:
        status = str(verification_statuses.get(skill.id) or "claimed").lower()
        if status not in VERIFICATION_WEIGHTS:
            status = "claimed"
        d["verification_status"] = status
    return d


def _proficiency_bonus(
    matched_req,
    req_count: int,
    levels: dict[str, str] | None,
    verification_statuses: dict[str, str] | None = None,
) -> int:
    if not levels or req_count <= 0:
        return 0
    pts = 0.0
    for s in matched_req:
        lvl = str(levels.get(s.id) or "basic").lower()
        if verification_statuses is not None:
            vs = str(verification_statuses.get(s.id) or "claimed").lower()
            if vs not in VERIFICATION_WEIGHTS:
                vs = "claimed"
            if vs == "unverified":
                lvl = "basic"
            pts += PROFICIENCY_POINTS.get(lvl, 0) * VERIFICATION_WEIGHTS[vs]
        else:
            pts += PROFICIENCY_POINTS.get(lvl, 0)
    return min(10, round(pts / req_count))


def score_job(
    student_skills: dict[str, list[str]],
    job,
    sim: float,
    taxonomy: Taxonomy,
    *,
    levels: dict[str, str] | None = None,
    student=None,
    verification_statuses: dict[str, str] | None = None,
) -> JobScore:
    """score = 100 * (0.6*required_coverage + 0.2*preferred_ratio + 0.2*semantic_sim),
    plus a small proficiency bonus (0..10) when `levels` is supplied.

    Coverage is depth-weighted (§19.3.1): each required skill credits 1.0 at or
    above the target depth (advanced for senior roles, intermediate otherwise),
    0.4 one level below, 0.1 two below. When `levels` is None and
    `verification_statuses` is None every present skill counts as meeting the
    target, so the score stays byte-for-byte equal to the original flat formula.

    When `verification_statuses` is provided, coverage and proficiency bonus
    apply verification multipliers (`verified=1.0`, `claimed=0.85`,
    `needs_reverification=0.5`, `unverified=0.2` with effective depth capped at
    `basic`) and flag unverified / needs-reverification job skills explicitly.

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

    # §19.3.1 — depth-weighted + verification-weighted required coverage.
    target = target_depth_for(job)
    weights: list[float] = []
    proficiency_gaps: list[dict] = []
    for s in req:
        if s.id not in student_skills:
            weights.append(0.0)
            proficiency_gaps.append({
                "skill": s.id, "current_level": "none",
                "target_level": target,
            })
            continue
        if levels is None and verification_statuses is None:
            weights.append(1.0)
            continue
        v_status = "verified"
        v_weight = 1.0
        if verification_statuses is not None:
            v_status = str(verification_statuses.get(s.id) or "claimed").lower()
            if v_status not in VERIFICATION_WEIGHTS:
                v_status = "claimed"
            v_weight = VERIFICATION_WEIGHTS[v_status]

        have_raw = str((levels or {}).get(s.id) or "basic").lower()
        eff_have = "basic" if v_status == "unverified" else have_raw
        dw = depth_weight(eff_have, target) if levels is not None else 1.0
        weights.append(round(dw * v_weight, 4))

        below_target = (levels is not None) and (level_rank(eff_have) < level_rank(target))
        if below_target or v_status in ("unverified", "needs_reverification"):
            gap_entry: dict[str, str] = {
                "skill": s.id,
                "current_level": eff_have,
                "target_level": target,
            }
            if v_status in ("unverified", "needs_reverification"):
                gap_entry["verification_status"] = v_status
            proficiency_gaps.append(gap_entry)

    coverage = (sum(weights) / len(req)) if req else 1.0
    if verification_statuses is not None and pref:
        pref_weights = []
        for s in matched_pref:
            vs = str(verification_statuses.get(s.id) or "claimed").lower()
            pref_weights.append(VERIFICATION_WEIGHTS.get(vs, 0.85))
        pref_score = sum(pref_weights) / len(pref)
    else:
        pref_score = len(matched_pref) / len(pref) if pref else 0.0
    sim = max(0.0, min(1.0, sim))
    base = round(100 * (WEIGHTS["coverage"] * coverage + WEIGHTS["preferred"] * pref_score + WEIGHTS["semantic"] * sim))
    bonus = _proficiency_bonus(matched_req, len(req), levels, verification_statuses)

    # §19.3.2 — experience mismatch penalty for campus freshers on senior roles.
    penalty = 0
    if student is not None and is_campus_fresher(student) and is_senior_job(job):
        penalty = SENIORITY_PENALTY

    unverified_skills: list[str] = []
    needs_rev_skills: list[str] = []
    matched_statuses: dict[str, str] = {}
    if verification_statuses is not None:
        for s in matched_req + matched_pref:
            if s.id in matched_statuses:
                continue
            vs = str(verification_statuses.get(s.id) or "claimed").lower()
            if vs not in VERIFICATION_WEIGHTS:
                vs = "claimed"
            matched_statuses[s.id] = vs
            if vs == "unverified":
                unverified_skills.append(s.id)
            elif vs == "needs_reverification":
                needs_rev_skills.append(s.id)

    score = max(0, min(100, base + bonus + penalty))
    return JobScore(
        job_id=job.id, score=score, band=band_for(score),
        matched_required=[_with_evidence(s, student_skills, levels, verification_statuses) for s in matched_req],
        missing_required=[_as_dict(s) for s in missing_req],
        matched_preferred=[_preferred_with_status(s, verification_statuses) for s in matched_pref],
        missing_preferred=[_as_dict(s) for s in missing_pref],
        semantic_sim=round(sim, 3),
        proficiency_bonus=bonus,
        proficiency_gaps=proficiency_gaps,
        seniority_penalty=penalty,
        unverified_skills=unverified_skills,
        needs_reverification_skills=needs_rev_skills,
        verification_statuses=matched_statuses,
    )
