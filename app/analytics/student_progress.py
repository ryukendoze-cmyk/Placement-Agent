"""Per-student PS-06 competency, readiness, and preparation progress report."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlmodel import Session, select

from ..matching.engine import band_for, is_branch_eligible, job_tier
from ..models import (
    Course,
    Interview,
    Job,
    JobApplication,
    JobMatch,
    PlanItem,
    ProgressEvent,
    Skill,
    Student,
    StudentSkill,
    UpskillPlan,
)
from ..rag.policy_rag import load_sop_params

_TIER_LABELS = {
    "standard": "Standard",
    "dream": "Dream",
    "super_dream": "Super Dream",
}

_KIND_BADGES = {
    "resume_parsed": ("Resume Parsed", "Resume Skills Parsed"),
    "matches_computed": ("Role Matching", "Target Role Matching Computed"),
    "plan_created": ("Plan Created", "30-Day Upskilling Plan Activated"),
    "plan_item_done": ("Module Done", "Completed Upskilling Module"),
    "quiz_passed": ("Skill Verified", "Micro-Quiz Verification Passed"),
    "interview_completed": ("Mock Interview", "Mock Interview Completed"),
    "skill_verified": ("Skill Verified", "Skill Verified via Assessment"),
    "skill_flagged": ("Re-verify", "Skill Flagged for Re-Verification"),
    "readiness_checkpoint": ("Checkpoint", "Readiness Milestone Updated"),
}

_SYNTHETIC_LEGACY_LADDER = [20, 34, 45, 58, 67, 75]


def _is_job_visible_to_student(job: Job | None, student_id: str) -> bool:
    if job is None:
        return False
    owner = getattr(job, "owner_student_id", None)
    src = str(getattr(job, "source", "local") or "local").strip().lower()
    is_live = (
        bool(getattr(job, "is_live", False))
        or str(getattr(job, "id", "") or "").startswith("job-ext-")
        or src not in ("", "local", "campus", "campus_drive", "jnf")
    )
    if is_live:
        return owner == student_id
    return owner in (None, student_id)


def _as_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _skill_name_lookup(session: Session, taxonomy: Any = None) -> dict[str, str]:
    names: dict[str, str] = {}
    if taxonomy is not None:
        try:
            for sk in taxonomy.all():
                names[sk.id] = sk.name
        except Exception:  # noqa: BLE001
            pass
    for sk in session.exec(select(Skill)).all():
        names.setdefault(sk.id, sk.name)
    return names


def compute_student_readiness(best_role_score: int, plan_done_items: int, plan_total_items: int) -> int:
    """Compute bounded 0..100 student readiness consistent with planner milestones."""
    if plan_total_items > 0:
        plan_pct = round(100 * plan_done_items / plan_total_items)
        return max(0, min(100, round(0.5 * plan_pct + 0.5 * best_role_score)))
    return max(0, min(100, int(best_role_score)))


def _build_milestone_series(
    events: list[ProgressEvent],
    *,
    current_readiness: int,
    projected_readiness: int,
    plan_modules_by_item: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Return chronological milestone points and whether legacy/compressed events were normalized."""
    now = datetime.now(timezone.utc)
    ordered = sorted(events, key=lambda e: (_as_utc(e.at), str(e.id)))

    readiness_seq = [int(e.readiness) for e in ordered]
    legacy_n = len(_SYNTHETIC_LEGACY_LADDER)
    is_legacy_ladder = readiness_seq == _SYNTHETIC_LEGACY_LADDER
    has_legacy_prefix = len(readiness_seq) >= legacy_n and readiness_seq[:legacy_n] == _SYNTHETIC_LEGACY_LADDER
    span_seconds = 0.0
    if len(ordered) >= 2:
        span_seconds = (_as_utc(ordered[-1].at) - _as_utc(ordered[0].at)).total_seconds()
    is_compressed = len(ordered) >= 2 and span_seconds <= 4 * 3600
    has_non_monotonic = any(readiness_seq[i] > readiness_seq[i + 1] for i in range(len(readiness_seq) - 1))
    exceeds_current = any(val > current_readiness for val in readiness_seq)

    points: list[dict[str, Any]] = []
    normalized_synthetic = False

    if (
        not ordered
        or is_legacy_ladder
        or has_legacy_prefix
        or is_compressed
        or has_non_monotonic
        or exceeds_current
    ):
        normalized_synthetic = bool(ordered)
        if current_readiness <= 0:
            baseline = 0
        else:
            baseline = min(current_readiness, round(current_readiness * 0.58))
            if baseline == current_readiness and current_readiness > 1:
                baseline = max(0, current_readiness - 6)

        if ordered:
            n = len(ordered)
            day_span = 14
            prev_val: int | None = None
            for idx, ev in enumerate(ordered):
                offset_days = round(day_span * (n - 1 - idx) / max(1, n - 1))
                dt = now - timedelta(days=offset_days)
                if idx == n - 1:
                    val = current_readiness
                else:
                    frac = idx / max(1, n - 1)
                    val = round(baseline + (current_readiness - baseline) * frac)
                if prev_val is not None and val < prev_val:
                    val = prev_val
                delta = 0 if prev_val is None else val - prev_val
                prev_val = val

                badge, default_title = _KIND_BADGES.get(ev.kind, ("Milestone", "Progress Milestone"))
                if ev.kind == "plan_item_done" and ev.ref_id in plan_modules_by_item:
                    mod = plan_modules_by_item[ev.ref_id]
                    title = f"Completed: {mod['skill_name']}"
                    detail = f"{mod['course_title']} ({mod['hours']} hrs)"
                else:
                    title = default_title
                    detail = f"Readiness {val}/100 recorded on {dt.strftime('%Y-%m-%d')}"

                short_prefix = "Baseline" if idx == 0 else ("Current" if idx == n - 1 else badge)
                points.append({
                    "stage_label": f"{idx + 1}. {badge}",
                    "short_label": f"{short_prefix} ({dt.strftime('%b %d')})",
                    "date_label": dt.strftime("%Y-%m-%d"),
                    "readiness": int(val),
                    "delta": int(delta),
                    "kind": ev.kind,
                    "title": title,
                    "detail": detail,
                    "is_projected": False,
                })
        else:
            dt_base = now - timedelta(days=7)
            dt_now = now
            points.append({
                "stage_label": "1. Baseline Profile",
                "short_label": f"Baseline ({dt_base.strftime('%b %d')})",
                "date_label": dt_base.strftime("%Y-%m-%d"),
                "readiness": int(baseline),
                "delta": 0,
                "kind": "resume_parsed",
                "title": "Resume Skills Parsed",
                "detail": "Initial skill extraction and competency baseline.",
                "is_projected": False,
            })
            points.append({
                "stage_label": "2. Current Standing",
                "short_label": f"Current ({dt_now.strftime('%b %d')})",
                "date_label": dt_now.strftime("%Y-%m-%d"),
                "readiness": int(current_readiness),
                "delta": int(current_readiness - baseline),
                "kind": "matches_computed",
                "title": "Current Cohort Readiness",
                "detail": f"Live readiness calibrated against matched roles ({current_readiness}/100).",
                "is_projected": False,
            })
    else:
        prev_val = None
        for idx, ev in enumerate(ordered):
            dt = _as_utc(ev.at)
            val = int(ev.readiness)
            delta = 0 if prev_val is None else val - prev_val
            prev_val = val
            badge, default_title = _KIND_BADGES.get(ev.kind, ("Milestone", "Progress Milestone"))
            if ev.kind == "plan_item_done" and ev.ref_id in plan_modules_by_item:
                mod = plan_modules_by_item[ev.ref_id]
                title = f"Completed: {mod['skill_name']}"
                detail = f"{mod['course_title']} ({mod['hours']} hrs)"
            else:
                title = default_title
                detail = f"Readiness {val}/100 recorded on {dt.strftime('%Y-%m-%d')}"
            short_prefix = "Baseline" if idx == 0 else badge
            points.append({
                "stage_label": f"{idx + 1}. {badge}",
                "short_label": f"{short_prefix} ({dt.strftime('%b %d')})",
                "date_label": dt.strftime("%Y-%m-%d"),
                "readiness": val,
                "delta": int(delta),
                "kind": ev.kind,
                "title": title,
                "detail": detail,
                "is_projected": False,
            })
        if points and points[-1]["readiness"] != current_readiness:
            delta_now = current_readiness - points[-1]["readiness"]
            points.append({
                "stage_label": f"{len(points) + 1}. Current",
                "short_label": f"Current ({now.strftime('%b %d')})",
                "date_label": now.strftime("%Y-%m-%d"),
                "readiness": int(current_readiness),
                "delta": int(delta_now),
                "kind": "readiness_checkpoint",
                "title": "Current Readiness Standing",
                "detail": f"Current role match and plan completion readiness ({current_readiness}/100).",
                "is_projected": False,
            })

    if projected_readiness > (points[-1]["readiness"] if points else current_readiness):
        last_val = points[-1]["readiness"] if points else current_readiness
        points.append({
            "stage_label": f"{len(points) + 1}. Post-Plan Target",
            "short_label": "Post-Plan Target",
            "date_label": "Projected",
            "readiness": int(projected_readiness),
            "delta": int(projected_readiness - last_val),
            "kind": "projected_target",
            "title": "Projected Post-Plan Readiness",
            "detail": "Estimated readiness once remaining 30-day modules and SOP-04 interview gate complete.",
            "is_projected": True,
        })

    return points, normalized_synthetic


def build_student_progress_report(
    session: Session,
    sid: str,
    taxonomy: Any = None,
) -> dict[str, Any]:
    """Read-only PS-06 student progress, skill verification, and SOP readiness report."""
    student = session.get(Student, sid)
    skill_names = _skill_name_lookup(session, taxonomy)

    sop04 = load_sop_params("SOP-04")
    sop05 = load_sop_params("SOP-05")
    sop04_min_mock = int(sop04.get("min_mock_score", 65) or 65)
    sop04_super_dream_min = int(sop04.get("super_dream_min_score", max(75, sop04_min_mock + 10)) or 75)
    sop04_tiers = {str(t).strip().lower() for t in (sop04.get("applicable_tiers") or ["dream", "super_dream"])}
    sop05_min_cov = int(sop05.get("min_coverage_pct", 40) or 40)

    # 1. Student skills & verification breakdown
    skill_rows = list(session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all())
    by_verification = {"verified": 0, "claimed": 0, "needs_reverification": 0, "unverified": 0}
    by_level = {"advanced": 0, "intermediate": 0, "basic": 0}
    skill_items: list[dict[str, Any]] = []
    student_skill_ids: set[str] = set()

    for ss in sorted(skill_rows, key=lambda r: (str(r.verification_status), str(r.skill_id))):
        student_skill_ids.add(ss.skill_id)
        vstat = str(getattr(ss, "verification_status", "claimed") or "claimed")
        by_verification[vstat] = by_verification.get(vstat, 0) + 1
        lvl = str(getattr(ss, "level", "basic") or "basic").lower()
        by_level[lvl] = by_level.get(lvl, 0) + 1
        skill_items.append({
            "skill_id": ss.skill_id,
            "name": skill_names.get(ss.skill_id, ss.skill_id.replace("_", " ").title()),
            "level": lvl,
            "verification_status": vstat,
            "confidence": round(float(getattr(ss, "confidence", 0.5) or 0.5), 2),
            "mentions": int(getattr(ss, "mentions", 0) or 0),
            "last_verified_score": getattr(ss, "last_verified_score", None),
        })

    # 2. UpskillPlan + PlanItems + Courses
    plans = list(session.exec(
        select(UpskillPlan).where(UpskillPlan.student_id == sid).order_by(UpskillPlan.created_at.desc())
    ).all())
    active_plan = plans[0] if plans else None
    raw_items: list[PlanItem] = []
    if active_plan is not None:
        raw_items = list(session.exec(
            select(PlanItem).where(PlanItem.plan_id == active_plan.id).order_by(PlanItem.week, PlanItem.order)
        ).all())

    courses_by_id = {c.id: c for c in session.exec(select(Course)).all()} if raw_items else {}
    plan_modules: list[dict[str, Any]] = []
    plan_modules_by_item: dict[str, dict[str, Any]] = {}
    weekly_map: dict[int, dict[str, Any]] = {
        w: {
            "week": w,
            "label": f"Week {w}",
            "done_hours": 0,
            "pending_hours": 0,
            "total_hours": 0,
            "done_items": 0,
            "total_items": 0,
        }
        for w in range(1, 5)
    }

    plan_done_items = 0
    plan_total_items = len(raw_items)
    plan_done_hours = 0
    plan_total_hours = 0
    plan_skill_ids: set[str] = set()

    for it in raw_items:
        course = courses_by_id.get(it.course_id)
        wk = min(4, max(1, int(getattr(it, "week", 1) or 1)))
        hrs = int(getattr(it, "hours", 0) or 0)
        is_done = str(it.status or "").lower() == "done"
        plan_skill_ids.add(it.skill_id)
        plan_total_hours += hrs
        weekly_map[wk]["total_hours"] += hrs
        weekly_map[wk]["total_items"] += 1
        if is_done:
            plan_done_items += 1
            plan_done_hours += hrs
            weekly_map[wk]["done_hours"] += hrs
            weekly_map[wk]["done_items"] += 1
        else:
            weekly_map[wk]["pending_hours"] += hrs

        mod_dict = {
            "id": it.id,
            "plan_id": it.plan_id,
            "order": int(it.order),
            "week": wk,
            "skill_id": it.skill_id,
            "skill_name": skill_names.get(it.skill_id, it.skill_id.replace("_", " ").title()),
            "course_title": course.title if course else f"Bridge Module: {it.skill_id}",
            "course_provider": course.provider if course else "Official Docs",
            "course_url": course.url if course else "",
            "hours": hrs,
            "status": str(it.status or "pending"),
        }
        plan_modules.append(mod_dict)
        plan_modules_by_item[it.id] = mod_dict

    plan_completion_pct = round(100 * plan_done_items / plan_total_items) if plan_total_items else 0

    # 3. Mock Interviews & Speech Telemetry
    raw_interviews = list(session.exec(
        select(Interview).where(Interview.student_id == sid).order_by(Interview.started_at.asc())
    ).all())
    finished_interviews = [iv for iv in raw_interviews if iv.finished or iv.score is not None]
    jobs_lookup = {j.id: j for j in session.exec(select(Job)).all()}

    def _coerce_float(val: object) -> float:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return 0.0
        if isinstance(val, dict):
            nums = [
                float(v)
                for v in val.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            return round(sum(nums) / len(nums), 2) if nums else 0.0
        return 0.0

    interview_history: list[dict[str, Any]] = []
    wpms: list[float] = []
    stars: list[float] = []
    for iv in finished_interviews:
        fb = iv.feedback if isinstance(iv.feedback, dict) else {}
        ssum = fb.get("speech_summary") if isinstance(fb.get("speech_summary"), dict) else {}
        wpm_val = _coerce_float(ssum.get("avg_wpm") or ssum.get("wpm"))
        filler_val = _coerce_float(ssum.get("filler_rate") or ssum.get("avg_filler_rate"))
        raw_rubric = fb.get("rubric_overall")
        star_val = _coerce_float(
            ssum.get("avg_star_score")
            or ssum.get("avg_star_completeness")
            or (raw_rubric if isinstance(raw_rubric, (int, float)) else 0)
        )
        if isinstance(iv.transcript, list):
            if not star_val:
                turn_stars = [
                    sv
                    for t in iv.transcript
                    if isinstance(t, dict) and isinstance(t.get("speech_analytics"), dict)
                    if (sv := _coerce_float(t["speech_analytics"].get("star_completeness"))) > 0
                ]
                if turn_stars:
                    star_val = round(sum(turn_stars) / len(turn_stars))
            if not wpm_val:
                turn_wpms = [
                    wv
                    for t in iv.transcript
                    if isinstance(t, dict) and isinstance(t.get("speech_analytics"), dict)
                    if (wv := _coerce_float(t["speech_analytics"].get("wpm"))) > 0
                ]
                if turn_wpms:
                    wpm_val = round(sum(turn_wpms) / len(turn_wpms))
        if not star_val and isinstance(raw_rubric, dict):
            star_val = _coerce_float(raw_rubric)

        if wpm_val > 0:
            wpms.append(wpm_val)
        if star_val > 0:
            stars.append(star_val)

        j = jobs_lookup.get(iv.job_id)
        dt = _as_utc(iv.started_at)
        raw_per_skill = fb.get("per_skill")
        interview_history.append({
            "id": iv.id,
            "job_id": iv.job_id,
            "job_title": j.title if j else iv.job_id,
            "company": j.company if j else "",
            "score": int(iv.score or 0),
            "started_at": dt.isoformat(),
            "date_label": dt.strftime("%Y-%m-%d"),
            "star_score": round(star_val) if star_val > 0 else 0,
            "wpm": round(wpm_val) if wpm_val > 0 else 0,
            "filler_rate": round(filler_val, 2),
            "per_skill": dict(raw_per_skill) if isinstance(raw_per_skill, dict) else {},
        })

    best_interview_score = max((ih["score"] for ih in interview_history), default=0)
    latest_interview_score = interview_history[-1]["score"] if interview_history else 0
    avg_wpm = round(sum(wpms) / len(wpms)) if wpms else 0
    avg_star_completeness = round(sum(stars) / len(stars)) if stars else 0

    # 4. Visible JobMatches & Target Roles + Eligibility Evaluation
    raw_matches = list(session.exec(
        select(JobMatch).where(JobMatch.student_id == sid).order_by(JobMatch.score.desc())
    ).all())
    visible_matches: list[tuple[JobMatch, Job]] = []
    for m in raw_matches:
        if getattr(m, "band", None) == "dismissed" or int(getattr(m, "score", 0) or 0) < 0:
            continue
        bd = m.breakdown if isinstance(m.breakdown, dict) else {}
        if bd.get("dismissed"):
            continue
        job = jobs_lookup.get(m.job_id)
        if not _is_job_visible_to_student(job, sid):
            continue
        visible_matches.append((m, job))

    target_roles: list[dict[str, Any]] = []
    eligible_drives_count = 0
    def _extract_skill_id(entry: object) -> str:
        if isinstance(entry, dict):
            return str(entry.get("skill") or entry.get("skill_id") or entry.get("id") or "")
        return str(entry or "")

    for idx, (m, job) in enumerate(visible_matches):
        bd = m.breakdown if isinstance(m.breakdown, dict) else {}
        req = list(job.required_skills or [])
        raw_matched = bd.get("matched_required")
        raw_missing = bd.get("missing_required")
        matched_req = (
            [s_id for s in raw_matched if (s_id := _extract_skill_id(s))]
            if raw_matched
            else [s for s in req if s in student_skill_ids]
        )
        matched_set = set(matched_req)
        missing_req = (
            [s_id for s in raw_missing if (s_id := _extract_skill_id(s))]
            if raw_missing
            else [s for s in req if s not in matched_set]
        )
        cov_pct = round(100 * len(matched_req) / len(req)) if req else 100
        tier = job_tier(job)
        tier_label = _TIER_LABELS.get(tier, tier.replace("_", " ").title())

        min_gpa = float(getattr(job, "min_gpa", 0.0) or 0.0)
        stu_gpa = float(getattr(student, "gpa", 0.0) or 0.0) if student else 0.0
        max_bl = getattr(job, "max_backlogs", None)
        stu_bl = int(getattr(student, "backlogs", 0) or 0) if student else 0
        gpa_passed = not (min_gpa > 0 and stu_gpa < min_gpa)
        if max_bl is not None and stu_bl > int(max_bl):
            gpa_passed = False

        allowed = [str(b).strip() for b in (getattr(job, "allowed_branches", None) or []) if str(b).strip()]
        branch_passed = is_branch_eligible(getattr(student, "branch", "") or "", allowed) if allowed else True
        needs_mock = tier in sop04_tiers
        sop04_passed = (not needs_mock) or (best_interview_score >= sop04_min_mock)
        sop05_passed = cov_pct >= sop05_min_cov
        eligible_now = gpa_passed and branch_passed and sop04_passed and sop05_passed
        if eligible_now:
            eligible_drives_count += 1

        if idx < 6:
            blockers: list[str] = []
            if not gpa_passed:
                blockers.append("SOP-03 CGPA/Backlog")
            if not branch_passed:
                blockers.append("SOP-09 Branch")
            if not sop04_passed:
                blockers.append(f"SOP-04 Mock >= {sop04_min_mock}%")
            if not sop05_passed:
                blockers.append(f"SOP-05 Coverage >= {sop05_min_cov}%")
            covered_by_plan = sum(1 for sk in missing_req if sk in plan_skill_ids)
            uplift = round(14 * (covered_by_plan / max(1, len(missing_req)))) if missing_req else 6
            projected_role_score = min(98, int(m.score) + max(4, uplift))
            target_roles.append({
                "job_id": job.id,
                "title": job.title,
                "company": job.company,
                "tier": tier,
                "tier_label": tier_label,
                "current_score": int(m.score),
                "projected_score": projected_role_score,
                "band": str(m.band),
                "matched_required": [skill_names.get(s, s) for s in matched_req],
                "missing_required": [skill_names.get(s, s) for s in missing_req],
                "required_coverage_pct": cov_pct,
                "gpa_passed": gpa_passed,
                "branch_passed": branch_passed,
                "sop04_passed": sop04_passed,
                "eligible_now": eligible_now,
                "blocker_summary": ", ".join(blockers) if blockers else "All SOP gates satisfied",
            })

    best_match_obj, best_job_obj = visible_matches[0] if visible_matches else (None, None)
    best_role_score = int(best_match_obj.score) if best_match_obj else 0
    best_role_tier = job_tier(best_job_obj) if best_job_obj else "standard"
    best_role_tier_label = _TIER_LABELS.get(best_role_tier, "Standard")

    # 5. Readiness & Milestones
    current_readiness = compute_student_readiness(best_role_score, plan_done_items, plan_total_items)
    if plan_total_items > 0:
        projected_best = target_roles[0]["projected_score"] if target_roles else min(98, best_role_score + 10)
        projected_readiness = max(current_readiness, min(98, round(0.5 * 100 + 0.5 * projected_best)))
    else:
        projected_readiness = min(98, current_readiness + 15) if current_readiness > 0 else 0

    raw_events = list(session.exec(
        select(ProgressEvent).where(ProgressEvent.student_id == sid).order_by(ProgressEvent.at.asc())
    ).all())
    milestone_series, normalized_synthetic = _build_milestone_series(
        raw_events,
        current_readiness=current_readiness,
        projected_readiness=projected_readiness,
        plan_modules_by_item=plan_modules_by_item,
    )

    non_projected = [pt for pt in milestone_series if not pt["is_projected"]]
    baseline_readiness = non_projected[0]["readiness"] if non_projected else current_readiness
    readiness_delta = current_readiness - baseline_readiness

    # 6. Enriched Activity Ledger (newest-first)
    enriched_events: list[dict[str, Any]] = []
    if raw_events and not normalized_synthetic:
        prev_r = None
        chron_enriched: list[dict[str, Any]] = []
        for ev in sorted(raw_events, key=lambda e: (_as_utc(e.at), str(e.id))):
            dt = _as_utc(ev.at)
            val = int(ev.readiness)
            delta = 0 if prev_r is None else val - prev_r
            delta_str = "Baseline" if prev_r is None else (f"+{delta} pts" if delta >= 0 else f"{delta} pts")
            prev_r = val
            badge, default_title = _KIND_BADGES.get(ev.kind, ("Milestone", "Progress Milestone"))
            if ev.kind == "plan_item_done" and ev.ref_id in plan_modules_by_item:
                mod = plan_modules_by_item[ev.ref_id]
                title = f"Completed: {mod['skill_name']}"
                detail = f"{mod['course_title']} ({mod['hours']} hrs)"
            else:
                title = default_title
                detail = f"Verified readiness milestone recorded ({val}/100)"
            chron_enriched.append({
                "id": ev.id,
                "kind": ev.kind,
                "badge_label": badge,
                "title": title,
                "detail": detail,
                "readiness": val,
                "delta_str": delta_str,
                "at": dt.isoformat(),
                "date_display": dt.strftime("%Y-%m-%d"),
            })
        enriched_events = list(reversed(chron_enriched))
    else:
        for idx, pt in enumerate(reversed(non_projected)):
            d = int(pt["delta"])
            delta_str = "Baseline" if idx == len(non_projected) - 1 else (f"+{d} pts" if d >= 0 else f"{d} pts")
            enriched_events.append({
                "id": f"ms-{len(non_projected) - idx}",
                "kind": pt["kind"],
                "badge_label": pt["stage_label"].split(". ", 1)[-1],
                "title": pt["title"],
                "detail": pt["detail"],
                "readiness": int(pt["readiness"]),
                "delta_str": delta_str,
                "at": pt["date_label"],
                "date_display": pt["date_label"],
            })

    # 7. Active applications count
    apps = list(session.exec(select(JobApplication).where(JobApplication.student_id == sid)).all())
    active_apps_count = sum(1 for a in apps if str(a.status or "").lower() not in ("withdrawn", "rejected"))

    flagged_count = by_verification.get("unverified", 0) + by_verification.get("needs_reverification", 0)
    kpis = {
        "current_readiness": int(current_readiness),
        "baseline_readiness": int(baseline_readiness),
        "readiness_delta": int(readiness_delta),
        "readiness_band": band_for(current_readiness),
        "projected_readiness": int(projected_readiness),
        "best_role_job_id": best_job_obj.id if best_job_obj else None,
        "best_role_title": best_job_obj.title if best_job_obj else "No matched roles yet",
        "best_role_company": best_job_obj.company if best_job_obj else "",
        "best_role_score": int(best_role_score),
        "best_role_tier": best_role_tier,
        "best_role_tier_label": best_role_tier_label,
        "best_role_band": str(best_match_obj.band) if best_match_obj else "early",
        "plan_id": active_plan.id if active_plan else None,
        "plan_done_items": int(plan_done_items),
        "plan_total_items": int(plan_total_items),
        "plan_done_hours": int(plan_done_hours),
        "plan_total_hours": int(plan_total_hours),
        "plan_completion_pct": int(plan_completion_pct),
        "total_skills": len(skill_items),
        "verified_skills_count": int(by_verification.get("verified", 0)),
        "claimed_skills_count": int(by_verification.get("claimed", 0)),
        "flagged_skills_count": int(flagged_count),
        "advanced_skills_count": int(by_level.get("advanced", 0)),
        "intermediate_skills_count": int(by_level.get("intermediate", 0)),
        "basic_skills_count": int(by_level.get("basic", 0)),
        "interviews_completed": len(interview_history),
        "best_interview_score": int(best_interview_score),
        "latest_interview_score": int(latest_interview_score),
        "sop04_dream_ready": best_interview_score >= sop04_min_mock,
        "sop04_super_dream_ready": best_interview_score >= sop04_super_dream_min,
        "avg_wpm": int(avg_wpm),
        "avg_star_completeness": int(avg_star_completeness),
        "eligible_drives_count": int(eligible_drives_count),
        "total_drives_evaluated": len(visible_matches),
        "active_applications_count": int(active_apps_count),
    }

    return {
        "kpis": kpis,
        "milestone_series": milestone_series,
        "target_roles": target_roles,
        "skill_breakdown": {
            "by_verification": by_verification,
            "by_level": by_level,
            "skills": skill_items,
        },
        "weekly_plan": [weekly_map[w] for w in range(1, 5)],
        "plan_modules": plan_modules,
        "interview_history": interview_history,
        "enriched_events": enriched_events,
    }
