"""Institutional SOP policy enforcement and defaults for campus drives."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SOP_FILE = DATA_DIR / "placement_sops.json"
DEFAULTS_FILE = DATA_DIR / "campus_drive_defaults.json"

VALID_WORK_MODES = ("On-site", "Remote", "Hybrid")
VALID_EXPERIENCE_LEVELS = ("Fresher", "Internship")
VALID_APPROVAL_STATUSES = ("DRAFT", "PENDING_TPO_APPROVAL", "PUBLISHED", "REJECTED")
VALID_POLICY_TIERS = ("super_dream", "dream", "standard")


def _load_sops_by_id() -> dict[str, dict[str, Any]]:
    if not SOP_FILE.exists():
        return {}
    rows = json.loads(SOP_FILE.read_text(encoding="utf-8"))
    return {str(r.get("id", "")).upper(): r for r in rows if isinstance(r, dict)}


def load_sop_params(sop_id: str) -> dict[str, Any]:
    sops = _load_sops_by_id()
    entry = sops.get(sop_id.upper(), {})
    params = entry.get("parameters")
    return dict(params) if isinstance(params, dict) else {}


def load_campus_defaults() -> dict[str, Any]:
    if not DEFAULTS_FILE.exists():
        return {
            "graduation_year": 2027,
            "work_mode": "On-site",
            "experience_level": "Fresher",
        }
    data = json.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


def tier_for(
    ctc_lpa_max: float | None,
    *,
    experience_level: str = "Fresher",
    salary_disclosed: bool = True,
) -> tuple[str | None, str | None]:
    """Classify placement tier from SOP-01 thresholds using the top of the CTC band.

    Must remain consistent with ``app/matching/engine.py::job_tier``.
    Internships or undisclosed/non-positive CTCs are never assigned an SOP tier.
    """
    if str(experience_level or "").strip().lower() == "internship":
        return (None, None)
    if not salary_disclosed or ctc_lpa_max is None:
        return (None, None)
    try:
        upper = float(ctc_lpa_max)
    except (TypeError, ValueError):
        return (None, None)
    if upper <= 0:
        return (None, None)

    sop01 = load_sop_params("SOP-01")
    super_dream_min = float(sop01.get("super_dream_min_lpa", 15.0))
    dream_min = float(sop01.get("dream_min_lpa", 8.0))

    if upper >= super_dream_min:
        return ("super_dream", "SOP-01")
    if upper >= dream_min:
        return ("dream", "SOP-01")
    return ("standard", "SOP-01")


def backlog_default_for_tier(
    policy_tier: str | None,
) -> tuple[int | None, list[str], dict[str, Any] | None]:
    """Return SOP-03 default max_active_backlogs, extra flags, and defaults_applied entry."""
    sop03 = load_sop_params("SOP-03")
    if policy_tier == "super_dream":
        val = int(sop03.get("super_dream_max_backlogs", 0))
        return (val, [], {"field": "max_active_backlogs", "value": val, "source": "SOP-03"})
    if policy_tier == "dream":
        val = int(sop03.get("dream_max_backlogs", 1))
        return (val, [], {"field": "max_active_backlogs", "value": val, "source": "SOP-03"})
    if policy_tier == "standard":
        return (None, ["departmental clearance"], None)
    return (None, ["max_active_backlogs"], None)


def _dedupe_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        s = str(item or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def compute_weighted_confidence(
    field_confidence: dict[str, float],
    *,
    has_required_skills: bool = False,
) -> float:
    """Compute weighted mean confidence across core campus drive fields."""
    weights = {
        "company": 1.5,
        "role_title": 1.5,
        "ctc_lpa_max": 1.5,
        "min_cgpa": 1.0,
        "allowed_branches": 1.0,
        "max_active_backlogs": 0.75,
        "graduation_year": 0.5,
        "location": 0.75,
        "work_mode": 0.5,
        "experience_level": 0.75,
        "skills": 1.25,
    }
    total_w = 0.0
    weighted_sum = 0.0
    for key, w in weights.items():
        if key == "skills":
            score = field_confidence.get("skills", 0.9 if has_required_skills else 0.3)
        elif key in field_confidence:
            score = float(field_confidence[key])
        else:
            continue
        weighted_sum += w * max(0.0, min(1.0, score))
        total_w += w
    if total_w <= 0:
        return 0.0
    return round(weighted_sum / total_w, 2)


def apply_policy_and_defaults(
    drive: dict[str, Any],
    *,
    explicitly_found: set[str] | None = None,
) -> dict[str, Any]:
    """Enforce SOP-01/SOP-03 rules and apply campus_drive_defaults.json transparently.

    Mutates and returns a copy of ``drive`` with up-to-date ``policy_tier``,
    ``policy_tier_source``, ``eligibility``, ``parser_metadata.defaults_applied``,
    and ``parser_metadata.missing_fields_flagged``.
    """
    out = dict(drive)
    elig = dict(out.get("eligibility") or {})
    meta = dict(out.get("parser_metadata") or {})
    field_conf = dict(meta.get("field_confidence") or {})
    missing = list(meta.get("missing_fields_flagged") or [])
    defaults_applied: list[dict[str, Any]] = []
    explicit = set(explicitly_found or ())

    # Re-build defaults_applied cleanly from current state so re-validation stays idempotent.
    cfg = load_campus_defaults()

    # 1. Experience level
    exp_lvl = out.get("experience_level")
    if exp_lvl not in VALID_EXPERIENCE_LEVELS:
        exp_lvl = None
    if not exp_lvl:
        exp_lvl = str(cfg.get("experience_level") or "Fresher")
        out["experience_level"] = exp_lvl
        field_conf.setdefault("experience_level", 0.3)
        defaults_applied.append({
            "field": "experience_level",
            "value": exp_lvl,
            "source": "campus_drive_defaults.json",
        })
        missing.append("experience_level")
    else:
        out["experience_level"] = exp_lvl

    # 2. Work mode
    wm = out.get("work_mode")
    if wm not in VALID_WORK_MODES:
        wm = None
    if not wm:
        wm = str(cfg.get("work_mode") or "On-site")
        out["work_mode"] = wm
        field_conf.setdefault("work_mode", 0.3)
        defaults_applied.append({
            "field": "work_mode",
            "value": wm,
            "source": "campus_drive_defaults.json",
        })
        missing.append("work_mode")
    else:
        out["work_mode"] = wm

    # 3. Graduation year
    grad_year = elig.get("graduation_year")
    if grad_year is None and "graduation_year" not in explicit:
        default_yr = cfg.get("graduation_year")
        if default_yr is not None:
            elig["graduation_year"] = int(default_yr)
            field_conf.setdefault("graduation_year", 0.3)
            defaults_applied.append({
                "field": "graduation_year",
                "value": int(default_yr),
                "source": "campus_drive_defaults.json",
            })
            missing.append("graduation_year")

    # 4. Salary & SOP-01 Policy Tier
    salary_disclosed = bool(out.get("salary_disclosed", False))
    ctc_min = out.get("ctc_lpa_min")
    ctc_max = out.get("ctc_lpa_max")
    if ctc_max is None and ctc_min is not None:
        ctc_max = ctc_min
        out["ctc_lpa_max"] = ctc_max
    if ctc_min is None and ctc_max is not None:
        ctc_min = ctc_max
        out["ctc_lpa_min"] = ctc_min
    if not salary_disclosed:
        out["ctc_lpa_min"] = None
        out["ctc_lpa_max"] = None
        ctc_max = None
        missing.append("salary_undisclosed")

    tier, tier_src = tier_for(
        ctc_max,
        experience_level=out["experience_level"],
        salary_disclosed=salary_disclosed,
    )
    out["policy_tier"] = tier
    out["policy_tier_source"] = tier_src
    if out["experience_level"] == "Internship":
        missing.append("internship_no_sop_tier")

    # 5. Min CGPA — NO SOP defines a default CGPA (stays None, flagged)
    if elig.get("min_cgpa") is None:
        field_conf.setdefault("min_cgpa", 0.3)
        missing.append("min_cgpa")

    # 6. Max active backlogs — SOP-03 defaults by tier when omitted from JD
    if elig.get("max_active_backlogs") is None and "max_active_backlogs" not in explicit:
        b_val, b_flags, b_default = backlog_default_for_tier(tier)
        if b_val is not None:
            elig["max_active_backlogs"] = b_val
        if b_default is not None:
            defaults_applied.append(b_default)
        field_conf.setdefault("max_active_backlogs", 0.3)
        missing.extend(b_flags)

    # 7. Allowed branches default is [] ("open to all"), flag if not explicit
    if "allowed_branches" not in elig or elig["allowed_branches"] is None:
        elig["allowed_branches"] = []
    if "allowed_branches" not in explicit and not elig["allowed_branches"]:
        field_conf.setdefault("allowed_branches", 0.3)

    # 8. Flag empty required metadata
    for req_field in ("company", "role_title", "location"):
        if not str(out.get(req_field) or "").strip():
            field_conf.setdefault(req_field, 0.3)
            missing.append(req_field)

    out["eligibility"] = elig
    meta["field_confidence"] = field_conf
    meta["defaults_applied"] = defaults_applied
    meta["missing_fields_flagged"] = _dedupe_list(missing)
    meta["confidence_score"] = compute_weighted_confidence(
        field_conf,
        has_required_skills=bool(out.get("required_skills")),
    )
    out["parser_metadata"] = meta
    return out
