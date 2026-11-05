from types import SimpleNamespace

import pytest

from app.matching.engine import band_for, score_job
from app.taxonomy.taxonomy import Taxonomy

T = Taxonomy([
    {"id": "python", "name": "Python", "category": "languages", "aliases": ["python"]},
    {"id": "fastapi", "name": "FastAPI", "category": "web_backend", "aliases": ["fastapi"]},
    {"id": "postgresql", "name": "PostgreSQL", "category": "databases", "aliases": ["postgresql"]},
    {"id": "docker", "name": "Docker", "category": "devops_cloud", "aliases": ["docker"]},
    {"id": "aws", "name": "AWS", "category": "devops_cloud", "aliases": ["aws"]},
])
JOB = SimpleNamespace(id="job-1", required_skills=["python", "fastapi", "postgresql"],
                      preferred_skills=["docker", "aws"])

def test_full_coverage_strong():
    s = score_job({"python": ["l1"], "fastapi": ["l1"], "postgresql": ["l1"], "docker": ["l1"]}, JOB, sim=0.9, taxonomy=T)
    assert s.band == "strong" and s.score >= 85
    assert len(s.missing_required) == 0 and len(s.missing_preferred) == 1
    assert s.matched_required[0]["evidence"] == "l1"

def test_half_coverage_moderate_or_developing():
    s = score_job({"python": ["l1"]}, JOB, sim=0.5, taxonomy=T)
    assert s.score < 60 and {m["skill"] for m in s.missing_required} == {"fastapi", "postgresql"}

def test_bands():
    assert band_for(85) == "strong" and band_for(70) == "moderate"
    assert band_for(50) == "developing" and band_for(20) == "early"


# ===========================================================================
# Milestone 11 — ARCHITECTURE §19.2/§19.3 (Seniority + Weighted Scoring)
# ===========================================================================
# Charter §5 re-attack: the engine features were ALREADY LANDED at write time
# (verified by direct read — initial grep reported them absent, a false baseline;
# lesson logged in breadcrumbs). Measured contracts: SENIOR_TITLE_RE:36,
# FRESHER_TITLE_RE:41, is_senior_job:62, is_fresher_job:69, is_campus_fresher:78,
# SENIORITY_PENALTY=-25:33, depth-weighted coverage in score_job:165-209,
# senior gate in job_eligibility:126. Tests attack the LANDED features (§5).
#
# Anti-vacuity (charter §4): the weighted-scoring test seeds BOTH a senior and a
# junior job and asserts the Basic candidate scores higher on the junior one, so
# a flat-binary implementation (both equal) fails, and a degenerate all-zero
# scorer (both low) fails the junior-side floor too.
# ===========================================================================


def _mk_job(**kw):
    base = dict(id="j", title="Software Engineer", required_skills=["python", "fastapi", "postgresql"],
                preferred_skills=[], experience_years=0, min_gpa=0.0, allowed_branches=[])
    base.update(kw)
    return SimpleNamespace(**base)


class TestJobSeniorityClassification:
    """§19.2 — deterministic seniority / fresher title classification."""

    def test_job_seniority_classification(self):
        from app.matching.engine import is_fresher_job, is_senior_job

        senior_titles = [
            "Senior Data Scientist", "Lead Developer", "Staff Architect",
            "Principal Engineer", "Engineering Manager", "Head of Data",
        ]
        fresher_titles = ["Junior Developer", "Software Engineering Intern",
                          "Trainee Engineer", "Associate Analyst", "Entry Level SDE"]

        for t in senior_titles:
            job = _mk_job(title=t, experience_years=0)
            assert is_senior_job(job), f"{t!r} must be classified senior"
        for t in fresher_titles:
            job = _mk_job(title=t, experience_years=0)
            assert not is_senior_job(job), f"{t!r} must NOT be classified senior"
            assert is_fresher_job(job), f"{t!r} must be classified fresher"

    def test_seniority_by_experience_years(self):
        """§19.2: experience_years >= 3 is senior even without a senior title."""
        from app.matching.engine import is_fresher_job, is_senior_job

        exp = _mk_job(title="Data Scientist", experience_years=5)
        assert is_senior_job(exp), "5 YOE must be senior by experience rule"
        assert not is_fresher_job(exp)
        # fresher: not senior AND (<=1 YOE or fresher title)
        junior_exp = _mk_job(title="Backend Engineer", experience_years=0)
        assert not is_senior_job(junior_exp)
        assert is_fresher_job(junior_exp)

    def test_job_model_is_fresher_property(self):
        """§19.2: Job SQLModel exposes .is_fresher property."""
        from app.models import Job

        senior = Job(
            id="job-sr-1",
            title="Senior Backend Engineer",
            company="Acme Corp",
            description="Leading microservices design",
            experience_years=0,
        )
        assert senior.is_fresher is False

        fresher = Job(
            id="job-jr-1",
            title="Associate Software Engineer",
            company="Acme Corp",
            description="Entry level development",
            experience_years=0,
        )
        assert fresher.is_fresher is True

        exp_senior = Job(
            id="job-exp-1",
            title="Software Engineer",
            company="Acme Corp",
            description="Experienced developer",
            experience_years=4,
        )
        assert exp_senior.is_fresher is False

        campus_job = Job(
            id="job-local-1",
            title="Graduate Engineer Trainee",
            company="Campus Drive",
            description="GET program",
            experience_years=0,
            source="local",
        )
        assert campus_job.is_fresher is True

    def test_seniority_and_fresher_dict_support(self):
        """§19.2: is_senior_job and is_fresher_job support raw dictionary objects."""
        from app.matching.engine import is_fresher_job, is_senior_job

        sr_dict = {"title": "Staff Architect", "experience_years": 0}
        assert is_senior_job(sr_dict) is True
        assert is_fresher_job(sr_dict) is False

        jr_dict = {"title": "Junior Python Developer", "experience_years": 0}
        assert is_senior_job(jr_dict) is False
        assert is_fresher_job(jr_dict) is True

        exp_dict = {"title": "Fullstack Developer", "experience_years": 3}
        assert is_senior_job(exp_dict) is True
        assert is_fresher_job(exp_dict) is False

    def test_matching_and_scraper_package_reexports(self):
        """§19.2: Contract exports from app.matching and app.scraper."""
        import app.matching as matching_pkg
        import app.scraper as scraper_pkg

        for name in ["FRESHER_TITLE_RE", "SENIOR_TITLE_RE", "is_fresher_job", "is_senior_job"]:
            assert hasattr(matching_pkg, name), f"app.matching must re-export {name}"
            assert hasattr(scraper_pkg, name), f"app.scraper must re-export {name}"

        assert hasattr(matching_pkg, "is_campus_fresher")

    def test_title_regex_boundary_cases(self):
        """§19.2: regex boundary matches on specific Indian industry patterns."""
        from app.matching.engine import FRESHER_TITLE_RE, SENIOR_TITLE_RE

        # Senior patterns
        assert SENIOR_TITLE_RE.search("VP of Engineering")
        assert SENIOR_TITLE_RE.search("Director of Technology")
        assert SENIOR_TITLE_RE.search("Specialist III - Cloud")
        assert SENIOR_TITLE_RE.search("Specialist IV - Data")
        assert SENIOR_TITLE_RE.search("Sr. Frontend Engineer")
        assert SENIOR_TITLE_RE.search("Principal Consultant")

        # Fresher patterns
        assert FRESHER_TITLE_RE.search("Graduate Engineer Trainee")
        assert FRESHER_TITLE_RE.search("GET - Systems")
        assert FRESHER_TITLE_RE.search("Associate Software Engineer")
        assert FRESHER_TITLE_RE.search("SDE-1")
        assert FRESHER_TITLE_RE.search("SDE 1")
        assert FRESHER_TITLE_RE.search("SDE-I")
        assert FRESHER_TITLE_RE.search("Analyst 1")
        assert FRESHER_TITLE_RE.search("Engineer 1")
        assert FRESHER_TITLE_RE.search("Jr. Developer")
        assert FRESHER_TITLE_RE.search("Summer Internship 2026")


class TestProficiencyWeightedScoring:
    """§19.3 — depth-weighted required-skill coverage (not flat binary)."""

    def test_proficiency_weighted_scoring(self):
        """§19.3.1 grade bands: senior target=advanced (basic->0.1), junior
        target=intermediate (basic->0.4); same candidate, same skills."""
        from app.matching.engine import depth_weight, score_job, target_depth_for

        # Candidate holds python+fastapi+postgresql but only at 'basic'.
        basic_skills = {"python": ["l1"], "fastapi": ["l1"], "postgresql": ["l1"]}
        levels = {k: "basic" for k in basic_skills}

        senior_job = _mk_job(title="Senior Software Engineer", experience_years=4)
        junior_job = _mk_job(title="Junior Software Engineer", experience_years=0)

        s_senior = score_job(basic_skills, senior_job, sim=0.6, taxonomy=T, levels=levels)
        s_junior = score_job(basic_skills, junior_job, sim=0.6, taxonomy=T, levels=levels)

        # §19.3.1 grade bands themselves (the lever §19.6's band is built on).
        # Anti-vacuity proves each lever actually fires in THIS scoring path:
        # basic->advanced contributes 0.1, basic->intermediate contributes 0.4.
        assert depth_weight("basic", "advanced") == pytest.approx(0.1)
        assert depth_weight("basic", "intermediate") == pytest.approx(0.4)
        assert depth_weight("intermediate", "intermediate") == pytest.approx(1.0)
        structural = 100 * (0.6 * sum(depth_weight("basic", target_depth_for(senior_job))
                                      for _ in range(len(basic_skills))) / len(basic_skills)
                            + 0.2 * 0.6)
        assert s_senior.score == pytest.approx(structural), (
            f"senior score {s_senior.score} != structural band {structural} computed "
            f"from §19.3.1 grade weights — grade weighting lever not firing in score_job"
        )
        assert s_junior.score > s_senior.score, (
            f"junior score {s_junior.score} must exceed senior {s_senior.score} "
            f"(basic->intermediate 0.4 > basic->advanced 0.1) — depth-weighting not applied"
        )
        # §19.6 band: candidate with Basic skills scores significantly lower
        # (~35-40% or more reduction) on Senior roles than on Junior/Fresher roles.
        # Under §19.3.1 grade weights, basic-on-senior scores 18% vs junior 36%
        # (a 50% relative reduction), well exceeding the ~35% lower threshold.
        rel = (s_junior.score - s_senior.score) / max(1, s_junior.score)
        assert s_senior.score <= 40, f"senior score {s_senior.score} exceeds 40%"
        assert rel >= 0.35, (
            f"basic-on-senior {s_senior.score} (junior {s_junior.score}, "
            f"reduction {rel:.0%}) reduction less than 35%"
        )
        # Anti-vacuity: the SAME candidate on the junior role must clear the
        # senior score (a flat or degenerate scorer fails one side).
        assert s_junior.score >= 35, (
            f"junior score {s_junior.score} too low — scorer may be degenerate"
        )


class TestJobEligibilitySeniorRejection:
    """§19.3.3 — campus fresher ineligible for senior roles, with reason."""

    def test_job_eligibility_senior_role_rejection(self):
        from app.matching.engine import is_senior_job, job_eligibility

        # Anchor on title only — experience_years=5 would itself fail eligibility,
        # masking whether the TITLE rule caused the rejection (charter §4).
        # 'Headend Engineer' (junior broadband role): the substring 'head' must
        # not classify it senior — measured defect at M11 baseline (engine.py:37).
        campus_student = SimpleNamespace(gpa=9.0, branch="CSE", year=3)
        headline_job = _mk_job(title="Headend Engineer", experience_years=0, min_gpa=0.0)

        assert is_senior_job(headline_job) is False, (
            "'Headend Engineer' must not classify senior (substring 'head')"
        )
        eligible, reasons = job_eligibility(campus_student, headline_job)
        assert eligible is True, (
            f"campus student must be eligible for the junior 'Headend Engineer' "
            f"role — the senior gate misfired on a title substring: {reasons}"
        )

    def test_job_eligibility_senior_role_rejection_exact_reason(self):
        from app.matching.engine import is_senior_job, job_eligibility

        campus_student = SimpleNamespace(gpa=9.0, branch="CSE", year=3)
        senior_job = _mk_job(title="Staff Engineer", experience_years=5, min_gpa=0.0)

        assert is_senior_job(senior_job), "precondition: job must be senior"
        eligible, reasons = job_eligibility(campus_student, senior_job)
        assert eligible is False, "campus student must be ineligible for senior role"
        # §19.3.3 quotes the EXACT reason string — assert it verbatim (charter §3:
        # measured at M11 baseline: current reason is
        # "Role requires experienced professional (Senior title or 3+ years
        # experience required)", which is the spec's own text -> GREEN if the
        # gate carries it, RED if the gate or wording drifts).
        assert "Role requires experienced professional (Senior title or 3+ years experience required)" in reasons, (
            f"senior rejection must carry the spec's verbatim reason, got {reasons}"
        )

    def test_job_eligibility_fresher_role_allowed(self, ):
        """Anti-vacuity: the SAME campus student IS eligible for a fresher role,
        so the senior rejection proves a senior-specific gate, not blanket block."""
        from app.matching.engine import job_eligibility

        campus_student = SimpleNamespace(gpa=9.0, branch="CSE", year=3)
        junior_job = _mk_job(title="Junior Developer", experience_years=0, min_gpa=0.0)
        eligible, reasons = job_eligibility(campus_student, junior_job)
        assert eligible is True, (
            f"campus student must be eligible for fresher role, reasons={reasons}"
        )


class TestMilestone12StrictFresherDisambiguation:
    """§20.2.2 & §20.2.3 — Strict fresher experience gate and deterministic extraction."""

    def test_is_fresher_job_rejects_associate_with_exp_gt_one(self):
        """Verifies an Associate/Junior role with > 1.0 year experience returns False."""
        from app.matching.engine import is_fresher_job

        # Associate role with 2 years experience: must be rejected strictly
        assoc_job_2yr = _mk_job(title="Associate Software Engineer", experience_years=2.0)
        assert is_fresher_job(assoc_job_2yr) is False, (
            "Associate role with 2 years experience must not be classified as fresher"
        )

        # Junior role with 1.5 years experience: must be rejected strictly
        jr_job_1_5yr = _mk_job(title="Junior Developer", experience_years=1.5)
        assert is_fresher_job(jr_job_1_5yr) is False, (
            "Junior role with 1.5 years experience must not be classified as fresher"
        )

        # Dictionary input support
        assoc_dict_2yr = {"title": "Associate Backend Engineer", "experience_years": 2}
        assert is_fresher_job(assoc_dict_2yr) is False, (
            "Dict associate job with 2 years must return False"
        )

        # Anti-vacuity: Associate role with <= 1.0 year experience IS fresher
        assoc_job_1yr = _mk_job(title="Associate Software Engineer", experience_years=1.0)
        assert is_fresher_job(assoc_job_1yr) is True, (
            "Associate role with 1.0 year experience must be classified as fresher"
        )

        assoc_job_0yr = _mk_job(title="Associate Developer", experience_years=0)
        assert is_fresher_job(assoc_job_0yr) is True, (
            "Associate role with 0 years experience must be classified as fresher"
        )

    def test_extract_experience_years_from_description(self):
        """Verifies extract_experience_years parses lower bound from range and plus patterns."""
        from app.scraper.normalizer import extract_experience_years

        # Range patterns
        assert extract_experience_years("1-3 years") == 1
        assert extract_experience_years("Requires 1 to 3 years experience in Python") == 1
        assert extract_experience_years("Looking for candidates with 3 - 5 years of software engineering") == 3

        # Plus patterns
        assert extract_experience_years("2+ years") == 2
        assert extract_experience_years("Candidate should have 4+ years of relevant experience") == 4
        assert extract_experience_years("5+ yrs in distributed systems") == 5

        # Freshers / no requirement
        assert extract_experience_years("Great opportunity for freshers and entry-level talent") == 0
        assert extract_experience_years("") == 0
        assert extract_experience_years(None) == 0


class TestMilestone13StrictExternalFresherGateAndYOE:
    """§21.2.1, §21.2.2, §21.3 — Strict external fresher gating and YOE parsing."""

    def test_is_fresher_job_rejects_external_roles_with_one_year_and_no_fresher_title(self):
        """Verifies unbadged external roles with 1 YOE return False, while local on-campus
        and badged Associate/Junior roles return True (ARCHITECTURE §21.2.2 & §21.3)."""
        from app.matching.engine import is_fresher_job

        # External unbadged roles with 1 YOE (must be rejected)
        ge_job = _mk_job(
            id="job-ext-unstop-1",
            title="Software Engineering Specialist- Python Fullstack",
            experience_years=1.0,
            source="unstop",
        )
        assert is_fresher_job(ge_job) is False, (
            "GE Vernova Software Engineering Specialist (1 YOE) must not be fresher"
        )

        thyrocare_job = _mk_job(
            id="job-ext-naukri-2",
            title="Data Analyst",
            experience_years=1.0,
            source="naukri",
        )
        assert is_fresher_job(thyrocare_job) is False, (
            "Thyrocare Data Analyst (1 YOE) must not be fresher"
        )

        headout_job = _mk_job(
            id="job-ext-linkedin-3",
            title="Product Analyst",
            experience_years=1.0,
            source="linkedin",
        )
        assert is_fresher_job(headout_job) is False, (
            "Headout Product Analyst (1 YOE) must not be fresher"
        )

        # Dictionary format support for external unbadged
        ext_dict_unbadged = {
            "id": "job-ext-4",
            "title": "Backend Software Developer",
            "experience_years": 1,
            "source": "indeed",
        }
        assert is_fresher_job(ext_dict_unbadged) is False

        # External roles with 1 YOE (must be rejected as fresher even if title has Junior/Associate)
        badged_assoc = _mk_job(
            id="job-ext-unstop-5",
            title="Associate Software Engineer",
            experience_years=1.0,
            source="unstop",
        )
        assert is_fresher_job(badged_assoc) is False, (
            "Associate Software Engineer with 1 YOE must not be classified as fresher"
        )

        badged_junior = _mk_job(
            id="job-ext-foundit-6",
            title="Junior Software Developer",
            experience_years=1.0,
            source="foundit",
        )
        assert is_fresher_job(badged_junior) is False, (
            "Junior Software Developer with 1 YOE must not be classified as fresher"
        )

        badged_trainee = _mk_job(
            id="job-ext-naukri-7",
            title="Graduate Engineer Trainee",
            experience_years=1.0,
            source="naukri",
        )
        assert is_fresher_job(badged_trainee) is False, (
            "Graduate Engineer Trainee with 1 YOE must not be classified as fresher"
        )

        # Local on-campus roles with 1 YOE (unbadged title allowed because on-campus)
        local_job = _mk_job(
            id="job-local-8",
            title="Software Developer",
            experience_years=1.0,
            source="local",
        )
        assert is_fresher_job(local_job) is True, (
            "Local campus role with 1 YOE must be classified as fresher"
        )

        # 0 YOE roles (both external and local are fresher)
        ext_zero_exp = _mk_job(
            id="job-ext-9",
            title="Data Analyst",
            experience_years=0.0,
            source="naukri",
        )
        assert is_fresher_job(ext_zero_exp) is True, (
            "External 0 YOE role must be classified as fresher regardless of title"
        )

    def test_extract_experience_years_yoe_and_digit_boundaries(self):
        """Verifies extract_experience_years parses YOE syntax and respects digit boundaries (ARCHITECTURE §21.2.1)."""
        from app.scraper.normalizer import extract_experience_years

        # YOE range
        assert extract_experience_years("3-9 YOE") == 3
        assert extract_experience_years("Requirements: 1 to 2 YOE in backend") == 1
        assert extract_experience_years("1-3 yoe") == 1

        # YOE single / plus
        assert extract_experience_years("5+ YOE") == 5
        assert extract_experience_years("2 YOE required") == 2

        # Realistic digit boundary enforcement (years like 2024 or 2025 must not parse as experience)
        assert extract_experience_years("Batch of 2024 graduating students") == 0
        assert extract_experience_years("Passout year 2025") == 0
        assert extract_experience_years("Role requires 20 years experience") == 20


