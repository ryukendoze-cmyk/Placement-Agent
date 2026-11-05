"""Cohort-level placement analytics. Pure functions over a SQLModel Session.

Each function takes a Session (and optional knobs) and returns JSON-safe
dict / list / dict[str,int] results, so they are fully unit-testable on
SQLite and directly serializable for the admin API.
"""
from collections import Counter, defaultdict

from sqlmodel import Session, select

from ..matching.engine import band_for
from ..models import Course, Job, JobMatch, PlanItem, Skill, Student, StudentSkill
from ..plan.planner import _load_courses, pick_course_for_skill


def _skill_names(db: Session) -> dict[str, str]:
    return {s.id: s.name for s in db.exec(select(Skill)).all()}


def skill_gap_report(db: Session, top_k_jobs_per_student: int = 3,
                     student_ids: set[str] | None = None) -> list[dict]:
    """For each skill demanded by top-matched jobs: demand (distinct such jobs),
    coverage (fraction of students who have the skill), and
    gap_score = (1 - coverage) * demand, sorted by gap_score desc.
    Only skills demanded by at least one top job appear.

    ``student_ids`` optionally restricts the cohort to those students (used for
    faculty-mentor department scoping, §17.2.B.3); None keeps the full cohort.
    """
    all_students = db.exec(select(Student)).all()
    if student_ids is not None:
        students = [s for s in all_students if s.id in student_ids]
    else:
        students = all_students
    total = len(students)
    if total == 0:
        return []
    top_job_ids: set[str] = set()
    have_skill: dict[str, set[str]] = defaultdict(set)
    for s in students:
        top = db.exec(select(JobMatch).where(JobMatch.student_id == s.id)
                      .order_by(JobMatch.score.desc()).limit(top_k_jobs_per_student)).all()
        for m in top:
            top_job_ids.add(m.job_id)
        for ss in db.exec(select(StudentSkill).where(StudentSkill.student_id == s.id)).all():
            have_skill[ss.skill_id].add(s.id)
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
        coverage = len(have_skill.get(sk, ())) / total
        gap = round((1 - coverage) * d, 3)
        report.append({"skill": sk, "name": names.get(sk, sk), "demand": d,
                       "coverage": round(coverage, 3), "gap_score": gap})
    report.sort(key=lambda r: (-r["gap_score"], r["skill"]))
    return report


def readiness_distribution(db: Session, student_ids: set[str] | None = None) -> dict[str, int]:
    """Band of each student's mean top-3 match score -> student count.

    All four bands are initialized to 0 so an empty band still appears (keeps the
    Chart.js doughnut color mapping stable instead of shifting keys).

    ``student_ids`` optionally restricts the cohort to those students (faculty
    department scoping, §17.2.B.3); None keeps the full cohort."""
    dist: dict[str, int] = {"strong": 0, "moderate": 0, "developing": 0, "early": 0}
    rows = db.exec(select(Student)).all()
    if student_ids is not None:
        rows = [s for s in rows if s.id in student_ids]
    for s in rows:
        scores = [m.score for m in db.exec(select(JobMatch).where(JobMatch.student_id == s.id)
                   .order_by(JobMatch.score.desc()).limit(3)).all()]
        if not scores:
            continue
        band = band_for(round(sum(scores) / len(scores)))
        dist[band] = dist.get(band, 0) + 1
    return dist


def top_requested_courses(db: Session, n: int = 5) -> list[dict]:
    """Most common courses across all plan items in the cohort."""
    counts = Counter(it.course_id for it in db.exec(select(PlanItem)).all())
    courses = {c.id: c for c in db.exec(select(Course)).all()}
    out = []
    for cid, cnt in counts.most_common(n):
        c = courses.get(cid)
        if c is not None:
            out.append({"course_id": c.id, "title": c.title, "provider": c.provider, "count": cnt})
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
        out.append({"skill": r["skill"], "name": r["name"], "gap_score": r["gap_score"],
                    "course_id": course["id"], "course_title": course["title"],
                    "provider": course["provider"]})
    return out
