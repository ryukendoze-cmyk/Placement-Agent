"""Cohort-level placement analytics. Pure functions over a SQLModel Session.

Each function takes a Session (and optional knobs) and returns JSON-safe
dict / list / dict[str,int] results, so they are fully unit-testable on
SQLite and directly serializable for the admin API.
"""
from collections import Counter, defaultdict

from sqlmodel import Session, select

from ..matching.engine import VERIFICATION_WEIGHTS, band_for, job_tier
from ..models import (
    Course,
    Interview,
    Job,
    JobMatch,
    PlanItem,
    Skill,
    Student,
    StudentSkill,
    UpskillPlan,
)
from ..plan.planner import _load_courses, pick_course_for_skill
from ..rag.policy_rag import load_sop_params
from ..taxonomy.branches import normalize_branch


def _clean_provider(provider: str | None) -> str:
    text = str(provider or "").strip()
    if not text or text.lower() == "youtube":
        return "Official Docs / Institutional Lab"
    return text


def _skill_names(db: Session) -> dict[str, str]:
    return {s.id: s.name for s in db.exec(select(Skill)).all()}


def _filter_students(db: Session, student_ids: set[str] | None, department: str | None) -> list[Student]:
    rows = db.exec(select(Student)).all()
    if student_ids is not None:
        rows = [s for s in rows if s.id in student_ids]
    if department:
        dept_norm = (normalize_branch(department) or department).strip().upper()
        rows = [
            s for s in rows
            if (normalize_branch(s.branch or "") or (s.branch or "")).strip().upper() == dept_norm
        ]
    return rows


def skill_gap_report(
    db: Session,
    top_k_jobs_per_student: int = 3,
    student_ids: set[str] | None = None,
    department: str | None = None,
) -> list[dict]:
    """For each skill demanded by top-matched jobs: demand (distinct such jobs),
    verification-weighted coverage, raw coverage, unverified claims count/details,
    and gap_score = (1 - coverage) * demand, sorted by gap_score desc.
    Only skills demanded by at least one top job appear.

    ``student_ids`` or ``department`` optionally restricts the cohort (used for
    faculty-mentor department scoping, §17.2.B.3); None keeps the full cohort.
    """
    students = _filter_students(db, student_ids, department)
    total = len(students)
    if total == 0:
        return []
    top_job_ids: set[str] = set()
    have_skill: dict[str, set[str]] = defaultdict(set)
    weighted_skill: dict[str, float] = defaultdict(float)
    unverified_by_skill: dict[str, list[dict]] = defaultdict(list)

    for s in students:
        top = db.exec(select(JobMatch).where(JobMatch.student_id == s.id)
                      .order_by(JobMatch.score.desc()).limit(top_k_jobs_per_student)).all()
        for m in top:
            top_job_ids.add(m.job_id)
        for ss in db.exec(select(StudentSkill).where(StudentSkill.student_id == s.id)).all():
            have_skill[ss.skill_id].add(s.id)
            vs = str(getattr(ss, "verification_status", "claimed") or "claimed").lower()
            w = VERIFICATION_WEIGHTS.get(vs, 0.85)
            weighted_skill[ss.skill_id] += w
            if vs in ("unverified", "needs_reverification"):
                unverified_by_skill[ss.skill_id].append({
                    "student_id": s.id,
                    "name": s.name,
                    "status": vs,
                    "last_verified_score": getattr(ss, "last_verified_score", None),
                })

    demand: dict[str, int] = defaultdict(int)
    for jid in top_job_ids:
        job = db.get(Job, jid)
        if job is None:
            continue
        for sk in set(job.required_skills):
            demand[sk] += 1
    names = _skill_names(db)
    report = []
    for sk, d in demand.items():
        coverage = weighted_skill.get(sk, 0.0) / total
        raw_coverage = len(have_skill.get(sk, ())) / total
        gap = round((1 - coverage) * d, 3)
        claims = unverified_by_skill.get(sk, [])
        report.append({
            "skill": sk,
            "name": names.get(sk, sk),
            "demand": d,
            "coverage": round(coverage, 3),
            "raw_coverage": round(raw_coverage, 3),
            "gap_score": gap,
            "unverified_claims": len(claims),
            "unverified_students": claims,
        })
    report.sort(key=lambda r: (-r["gap_score"], r["skill"]))
    return report


def readiness_distribution(
    db: Session,
    student_ids: set[str] | None = None,
    department: str | None = None,
) -> dict[str, int]:
    """Band of each student's mean top-3 match score -> student count.

    All four bands are initialized to 0 so an empty band still appears (keeps the
    Chart.js doughnut color mapping stable instead of shifting keys).

    ``student_ids`` or ``department`` optionally restricts the cohort (faculty
    department scoping, §17.2.B.3); None keeps the full cohort."""
    dist: dict[str, int] = {"strong": 0, "moderate": 0, "developing": 0, "early": 0}
    rows = _filter_students(db, student_ids, department)
    for s in rows:
        scores = [m.score for m in db.exec(select(JobMatch).where(JobMatch.student_id == s.id)
                   .order_by(JobMatch.score.desc()).limit(3)).all()]
        if not scores:
            continue
        band = band_for(round(sum(scores) / len(scores)))
        dist[band] = dist.get(band, 0) + 1
    return dist


def unverified_claims_report(
    db: Session,
    student_ids: set[str] | None = None,
    department: str | None = None,
) -> list[dict]:
    """Return unverified / needs_reverification skill claims across the cohort."""
    students = {s.id: s for s in _filter_students(db, student_ids, department)}
    if not students:
        return []
    names = _skill_names(db)
    rows = db.exec(
        select(StudentSkill).where(
            StudentSkill.verification_status.in_(["unverified", "needs_reverification"])
        )
    ).all()
    out = []
    for ss in rows:
        stu = students.get(ss.student_id)
        if stu is None:
            continue
        out.append({
            "student_id": stu.id,
            "student_name": stu.name,
            "branch": stu.branch,
            "skill_id": ss.skill_id,
            "skill_name": names.get(ss.skill_id, ss.skill_id),
            "claimed_level": ss.level,
            "verification_status": ss.verification_status,
            "confidence": round(float(getattr(ss, "confidence", 0.5) or 0.5), 2),
            "last_verified_score": getattr(ss, "last_verified_score", None),
        })
    out.sort(key=lambda r: (0 if r["verification_status"] == "unverified" else 1, r["student_name"], r["skill_id"]))
    return out


def top_requested_courses(db: Session, n: int = 5) -> list[dict]:
    """Most common courses across all plan items in the cohort."""
    counts = Counter(it.course_id for it in db.exec(select(PlanItem)).all())
    courses = {c.id: c for c in db.exec(select(Course)).all()}
    out = []
    for cid, cnt in counts.most_common(n):
        c = courses.get(cid)
        if c is not None:
            out.append({
                "course_id": c.id,
                "title": c.title,
                "provider": _clean_provider(c.provider),
                "count": cnt,
            })
    return out


def programme_recommendations(db: Session, n: int = 5, gap_report: list[dict] | None = None) -> list[dict]:
    """Top gap skills -> a recommended (best covering) course for each.

    Accepts a precomputed `gap_report` to avoid re-running the (n_students) queries
    when the caller already computed one for the same request."""
    out = []
    courses = _load_courses(db)
    for r in (gap_report if gap_report is not None else skill_gap_report(db)):
        if len(out) >= n:
            break
        course = pick_course_for_skill(r["skill"], courses)
        if course is None:
            continue
        out.append({
            "skill": r["skill"],
            "name": r["name"],
            "gap_score": r["gap_score"],
            "course_id": course["id"],
            "course_title": course["title"],
            "provider": _clean_provider(course.get("provider")),
        })
    return out


_ROLE_TRACKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Core ECE / Embedded", ("embedded", "firmware", "vlsi", "rtos", "iot", "hardware", "microcontroller", "fpga", "ece")),
    ("DevOps & SRE", ("devops", "sre", "kubernetes", "docker", "terraform", "ci/cd", "cicd", "linux", "ansible", "cloud engineer")),
    ("Data & AI/ML", ("data", "machine learning", "deep learning", "ml", "ai", "nlp", "pandas", "pytorch", "tensorflow", "analytics")),
    ("Frontend & Full-Stack", ("frontend", "front-end", "full-stack", "fullstack", "react", "angular", "vue", "typescript", "javascript", "ui")),
    ("Backend & Cloud", ("backend", "back-end", "api", "microservice", "fastapi", "django", "spring", "node", "cloud", "software", "sde")),
)


def _classify_job_track(job: Job) -> str:
    blob = " ".join([
        str(getattr(job, "title", "") or ""),
        str(getattr(job, "description", "") or "")[:300],
        " ".join(str(s) for s in (getattr(job, "required_skills", None) or [])),
    ]).lower()
    best_track = "Backend & Cloud"
    best_hits = 0
    for track, kws in _ROLE_TRACKS:
        hits = sum(1 for kw in kws if kw in blob)
        if hits > best_hits:
            best_hits = hits
            best_track = track
    return best_track


def _cohort_ctx(
    db: Session,
    precomputed_matches: list[JobMatch] | None = None,
    *,
    student_ids: set[str] | None = None,
    department: str | None = None,
) -> dict:
    students = _filter_students(db, student_ids, department)
    students.sort(key=lambda s: (str(s.name or "").lower(), str(s.id)))
    sids = {s.id for s in students}

    all_matches = (
        list(precomputed_matches)
        if precomputed_matches is not None
        else db.exec(select(JobMatch)).all()
    )
    matches_by_student: dict[str, list[JobMatch]] = defaultdict(list)
    for m in all_matches:
        if (
            m.student_id in sids
            and str(getattr(m, "band", "") or "").lower() != "dismissed"
            and not getattr(m, "dismissed", False)
        ):
            matches_by_student[m.student_id].append(m)
    for sid in matches_by_student:
        matches_by_student[sid].sort(key=lambda m: (-int(m.score or 0), str(m.job_id)))

    skills_by_student: dict[str, list[StudentSkill]] = defaultdict(list)
    if sids:
        for ss in db.exec(select(StudentSkill)).all():
            if ss.student_id in sids:
                skills_by_student[ss.student_id].append(ss)

    jobs_by_id = {j.id: j for j in db.exec(select(Job)).all()}
    skill_names = _skill_names(db)

    latest_iv_by_student: dict[str, Interview] = {}
    if sids:
        iv_rows = db.exec(
            select(Interview)
            .where(Interview.finished == True)  # noqa: E712
            .order_by(Interview.started_at.desc(), Interview.id.desc())
        ).all()
        for iv in iv_rows:
            if iv.student_id in sids and iv.score is not None and iv.student_id not in latest_iv_by_student:
                latest_iv_by_student[iv.student_id] = iv

    plans_by_student: dict[str, list[str]] = defaultdict(list)
    if sids:
        for p in db.exec(select(UpskillPlan)).all():
            if p.student_id in sids:
                plans_by_student[p.student_id].append(p.id)
    plan_to_student = {pid: sid for sid, pids in plans_by_student.items() for pid in pids}
    items_by_student: dict[str, list[PlanItem]] = defaultdict(list)
    if plan_to_student:
        for it in db.exec(select(PlanItem)).all():
            sid = plan_to_student.get(it.plan_id)
            if sid:
                items_by_student[sid].append(it)

    sop04 = load_sop_params("SOP-04")
    min_mock_score = int(sop04.get("min_mock_score", 65))
    sop05 = load_sop_params("SOP-05")
    min_cov_pct = int(sop05.get("min_coverage_pct", 40))

    return {
        "students": students,
        "matches_by_student": matches_by_student,
        "skills_by_student": skills_by_student,
        "jobs_by_id": jobs_by_id,
        "skill_names": skill_names,
        "latest_iv_by_student": latest_iv_by_student,
        "items_by_student": items_by_student,
        "min_mock_score": min_mock_score,
        "min_cov_pct": min_cov_pct,
    }


def _student_readiness_score(matches: list[JobMatch]) -> int:
    if not matches:
        return 0
    top3 = [int(m.score or 0) for m in matches[:3]]
    return round(sum(top3) / len(top3))


def cohort_executive_kpis(
    db: Session,
    precomputed_matches: list[JobMatch] | None = None,
    *,
    student_ids: set[str] | None = None,
    department: str | None = None,
) -> dict:
    """Compute the 6 PS-06 executive cohort KPIs from live database records."""
    c = _cohort_ctx(db, precomputed_matches, student_ids=student_ids, department=department)
    students = c["students"]
    total = len(students)
    branches = sorted({
        (normalize_branch(s.branch or "") or (s.branch or "")).strip().upper()
        for s in students if (s.branch or "").strip()
    })

    readiness_scores: list[int] = []
    eligible_count = 0
    verified_skills = 0
    total_skills = 0
    mock_passed = 0
    mock_attempted = 0
    plan_done = 0
    plan_total = 0

    for s in students:
        ms = c["matches_by_student"].get(s.id, [])
        r_score = _student_readiness_score(ms)
        readiness_scores.append(r_score)
        best_job = c["jobs_by_id"].get(ms[0].job_id) if ms else None
        min_gpa = float(getattr(best_job, "min_gpa", 0.0) or 0.0) if best_job else 0.0
        gpa_ok = (min_gpa == 0.0) or (float(s.gpa or 0.0) >= min_gpa)
        if ms and band_for(r_score) != "early" and gpa_ok:
            eligible_count += 1

        for ss in c["skills_by_student"].get(s.id, []):
            total_skills += 1
            if str(getattr(ss, "verification_status", "") or "").lower() == "verified":
                verified_skills += 1

        iv = c["latest_iv_by_student"].get(s.id)
        if iv is not None and iv.score is not None:
            mock_attempted += 1
            if int(iv.score) >= c["min_mock_score"]:
                mock_passed += 1

        for it in c["items_by_student"].get(s.id, []):
            plan_total += 1
            if str(getattr(it, "status", "") or "").lower() in ("done", "completed"):
                plan_done += 1

    return {
        "total_students": total,
        "active_branches": len(branches),
        "branches": branches,
        "mean_readiness_index": round(sum(readiness_scores) / total, 1) if total else 0.0,
        "drive_eligible_count": eligible_count,
        "drive_eligible_pct": round(100.0 * eligible_count / total, 1) if total else 0.0,
        "verified_skills_count": verified_skills,
        "total_skills_count": total_skills,
        "verified_authenticity_pct": round(100.0 * verified_skills / total_skills, 1) if total_skills else 0.0,
        "mock_passed_count": mock_passed,
        "mock_attempted_count": mock_attempted,
        "mock_pass_rate_pct": round(100.0 * mock_passed / total, 1) if total else 0.0,
        "min_mock_score": c["min_mock_score"],
        "plan_done_count": plan_done,
        "plan_total_count": plan_total,
        "plan_completion_pct": round(100.0 * plan_done / plan_total, 1) if plan_total else 0.0,
    }


def branch_readiness_breakdown(
    db: Session,
    precomputed_matches: list[JobMatch] | None = None,
    *,
    student_ids: set[str] | None = None,
    department: str | None = None,
) -> list[dict]:
    """Branch-wise readiness, CGPA x10, and mock interview index across JKLU branches."""
    c = _cohort_ctx(db, precomputed_matches, student_ids=student_ids, department=department)
    core_branches = ["CSE", "AI&DS", "ECE", "IT", "ME", "EE"]
    grouped: dict[str, list[Student]] = defaultdict(list)
    for s in c["students"]:
        b = (normalize_branch(s.branch or "") or str(s.branch or "OTHER").strip().upper()) or "OTHER"
        grouped[b].append(s)

    ordered = list(core_branches)
    for b in sorted(grouped):
        if b not in ordered:
            ordered.append(b)

    out: list[dict] = []
    for branch in ordered:
        rows = grouped.get(branch, [])
        if not rows and branch not in core_branches:
            continue
        n = len(rows)
        r_scores = [_student_readiness_score(c["matches_by_student"].get(s.id, [])) for s in rows]
        cgpas = [float(s.gpa or 0.0) for s in rows if s.gpa is not None]
        iv_scores = [
            int(c["latest_iv_by_student"][s.id].score)
            for s in rows
            if s.id in c["latest_iv_by_student"] and c["latest_iv_by_student"][s.id].score is not None
        ]
        sop04_pass = sum(1 for sc in iv_scores if sc >= c["min_mock_score"])
        mean_cgpa = round(sum(cgpas) / len(cgpas), 2) if cgpas else 0.0
        out.append({
            "branch": branch,
            "student_count": n,
            "mean_readiness": round(sum(r_scores) / n, 1) if n else 0.0,
            "mean_cgpa": mean_cgpa,
            "cgpa_x10": round(mean_cgpa * 10.0, 1),
            "mock_interview_index": round(sum(iv_scores) / len(iv_scores), 1) if iv_scores else 0.0,
            "sop04_pass_pct": round(100.0 * sop04_pass / n, 1) if n else 0.0,
        })
    return out


def role_track_readiness(
    db: Session,
    precomputed_matches: list[JobMatch] | None = None,
    *,
    student_ids: set[str] | None = None,
    department: str | None = None,
) -> list[dict]:
    """Readiness & eligibility across the 5 PS-06 career role tracks."""
    c = _cohort_ctx(db, precomputed_matches, student_ids=student_ids, department=department)
    track_names = [
        "Backend & Cloud",
        "Frontend & Full-Stack",
        "Data & AI/ML",
        "DevOps & SRE",
        "Core ECE / Embedded",
    ]
    jobs_by_track: dict[str, list[Job]] = {t: [] for t in track_names}
    job_to_track: dict[str, str] = {}
    for j in c["jobs_by_id"].values():
        t = _classify_job_track(j)
        job_to_track[j.id] = t
        jobs_by_track.setdefault(t, []).append(j)

    stu_by_id = {s.id: s for s in c["students"]}
    track_best_by_stu: dict[str, dict[str, tuple[int, Job]]] = {t: {} for t in track_names}
    for sid, ms in c["matches_by_student"].items():
        for m in ms:
            t = job_to_track.get(m.job_id)
            j = c["jobs_by_id"].get(m.job_id)
            if not t or j is None:
                continue
            prev = track_best_by_stu[t].get(sid)
            if prev is None or int(m.score or 0) > prev[0]:
                track_best_by_stu[t][sid] = (int(m.score or 0), j)

    out: list[dict] = []
    total_students = len(c["students"])
    for t in track_names:
        best_map = track_best_by_stu.get(t, {})
        scores = [pair[0] for pair in best_map.values()]
        eligible = 0
        for sid, (sc, job) in best_map.items():
            stu = stu_by_id.get(sid)
            min_gpa = float(getattr(job, "min_gpa", 0.0) or 0.0)
            gpa_ok = (min_gpa == 0.0) or (stu is not None and float(stu.gpa or 0.0) >= min_gpa)
            if band_for(sc) != "early" and gpa_ok:
                eligible += 1
        out.append({
            "track": t,
            "job_count": len(jobs_by_track.get(t, [])),
            "matched_students": len(best_map),
            "mean_readiness": round(sum(scores) / len(scores), 1) if scores else 0.0,
            "eligible_pct": round(100.0 * eligible / total_students, 1) if total_students else 0.0,
            "sample_roles": [j.title for j in jobs_by_track.get(t, [])[:3]],
        })
    return out


def student_readiness_matrix(
    db: Session,
    precomputed_matches: list[JobMatch] | None = None,
    *,
    student_ids: set[str] | None = None,
    department: str | None = None,
) -> list[dict]:
    """Per-student placement readiness, verification breakdown, SOP-04 gate, and gap summary."""
    c = _cohort_ctx(db, precomputed_matches, student_ids=student_ids, department=department)
    out: list[dict] = []
    for s in c["students"]:
        ms = c["matches_by_student"].get(s.id, [])
        r_score = _student_readiness_score(ms)
        band = band_for(r_score) if ms else "early"

        best_match = ms[0] if ms else None
        best_job = c["jobs_by_id"].get(best_match.job_id) if best_match else None
        best_tier = job_tier(best_job) if best_job else "standard"

        ss_rows = c["skills_by_student"].get(s.id, [])
        have_ids = {r.skill_id for r in ss_rows}
        verified = sum(1 for r in ss_rows if str(r.verification_status or "").lower() == "verified")
        claimed = sum(1 for r in ss_rows if str(r.verification_status or "").lower() in ("claimed", "needs_reverification"))
        unverified = sum(1 for r in ss_rows if str(r.verification_status or "").lower() == "unverified")

        iv = c["latest_iv_by_student"].get(s.id)
        mock_score = int(iv.score) if (iv is not None and iv.score is not None) else None
        sop04_passed = mock_score is not None and mock_score >= c["min_mock_score"]

        items = c["items_by_student"].get(s.id, [])
        done_items = sum(1 for it in items if str(it.status or "").lower() in ("done", "completed"))
        plan_pct = round(100 * done_items / len(items)) if items else 0

        missing_names: list[str] = []
        if best_job is not None:
            for sk in (getattr(best_job, "required_skills", None) or []):
                if sk not in have_ids:
                    missing_names.append(c["skill_names"].get(sk, sk))
                if len(missing_names) >= 3:
                    break

        branch_norm = normalize_branch(s.branch or "") or (s.branch or "")
        out.append({
            "student_id": s.id,
            "name": s.name,
            "branch": branch_norm,
            "year": s.year,
            "gpa": round(float(s.gpa or 0.0), 2),
            "backlogs": int(getattr(s, "backlogs", 0) or 0),
            "readiness_score": r_score,
            "readiness_band": band,
            "best_role": best_job.title if best_job else "No matched role",
            "best_company": best_job.company if best_job else "",
            "best_tier": best_tier,
            "verified_skills": verified,
            "claimed_skills": claimed,
            "unverified_skills": unverified,
            "mock_score": mock_score,
            "sop04_passed": sop04_passed,
            "plan_completion_pct": plan_pct,
            "top_missing_skills": missing_names,
        })
    return out


def enriched_skill_gap_matrix(
    db: Session,
    precomputed_matches: list[JobMatch] | None = None,
    *,
    student_ids: set[str] | None = None,
    department: str | None = None,
) -> list[dict]:
    """Cohort skill demand vs verified/claimed mastery, blocked students, and institutional intervention."""
    c = _cohort_ctx(db, precomputed_matches, student_ids=student_ids, department=department)
    students = c["students"]
    total_students = len(students)
    if total_students == 0:
        return []

    top_job_ids: set[str] = set()
    student_top_req: dict[str, set[str]] = {}
    for s in students:
        top3 = c["matches_by_student"].get(s.id, [])[:3]
        reqs: set[str] = set()
        for m in top3:
            top_job_ids.add(m.job_id)
            j = c["jobs_by_id"].get(m.job_id)
            if j is not None:
                reqs.update(j.required_skills or [])
        student_top_req[s.id] = reqs

    target_jobs = [c["jobs_by_id"][jid] for jid in top_job_ids if jid in c["jobs_by_id"]]
    if not target_jobs:
        target_jobs = list(c["jobs_by_id"].values())
    total_jobs = max(1, len(target_jobs))

    demand: dict[str, int] = defaultdict(int)
    for j in target_jobs:
        for sk in set(j.required_skills or []):
            demand[sk] += 1

    verified_students: dict[str, set[str]] = defaultdict(set)
    claimed_students: dict[str, set[str]] = defaultdict(set)
    unverified_students: dict[str, set[str]] = defaultdict(set)
    weighted_sum: dict[str, float] = defaultdict(float)

    for s in students:
        for ss in c["skills_by_student"].get(s.id, []):
            vs = str(getattr(ss, "verification_status", "claimed") or "claimed").lower()
            weighted_sum[ss.skill_id] += VERIFICATION_WEIGHTS.get(vs, 0.85)
            if vs == "verified":
                verified_students[ss.skill_id].add(s.id)
            elif vs == "unverified":
                unverified_students[ss.skill_id].add(s.id)
            else:
                claimed_students[ss.skill_id].add(s.id)

    rows: list[dict] = []
    for sk, d in demand.items():
        v_cnt = len(verified_students.get(sk, ()))
        c_cnt = len(claimed_students.get(sk, ()))
        u_cnt = len(unverified_students.get(sk, ()))
        v_pct = round(100.0 * v_cnt / total_students, 1)
        c_pct = round(100.0 * c_cnt / total_students, 1)
        gap_pct = round(max(0.0, 100.0 - v_pct - c_pct), 1)
        demand_pct = round(100.0 * d / total_jobs, 1)
        w_cov = weighted_sum.get(sk, 0.0) / total_students
        gap_score = round((1.0 - min(1.0, w_cov)) * d, 3)

        blocked = sum(
            1 for s in students
            if sk in student_top_req.get(s.id, set())
            and s.id not in verified_students.get(sk, set())
            and s.id not in claimed_students.get(sk, set())
        )
        if u_cnt > 0 and u_cnt >= blocked:
            intervention = "Supervised Verification Lab & Code Assessment"
        elif gap_pct >= 60.0:
            intervention = "Institutional Hands-On Bootcamp & Project Lab"
        else:
            intervention = "Guided Technical Workshop & Official Docs Sprint"

        rows.append({
            "skill_id": sk,
            "name": c["skill_names"].get(sk, sk),
            "roles_demanding_count": d,
            "role_demand_pct": demand_pct,
            "verified_pct": v_pct,
            "claimed_pct": c_pct,
            "unverified_or_missing_pct": gap_pct,
            "students_blocked": blocked,
            "gap_score": gap_score,
            "intervention_type": intervention,
        })
    rows.sort(key=lambda r: (-r["gap_score"], -r["role_demand_pct"], r["skill_id"]))
    return rows
