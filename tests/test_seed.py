
from sqlmodel import Session, select

from app.db import init_db, make_engine
from app.models import Course, Job, Skill
from app.seed import _json, seed
from app.taxonomy.taxonomy import load_taxonomy


def test_seed_populates_all_tables():
    eng = make_engine("sqlite://")
    init_db(eng)
    with Session(eng) as db:
        seed(db, load_taxonomy())
        assert len(db.exec(select(Skill)).all()) >= 30
        assert len(db.exec(select(Job)).all()) >= 20
        assert len(db.exec(select(Course)).all()) >= 20


def test_every_required_skill_has_a_course():
    jobs = _json("jobs.json")
    courses = _json("courses.json")
    covered = {s for c in courses for s in c["skill_ids"]}
    required = {s for j in jobs for s in j["required_skills"]}
    assert not (required - covered), f"missing courses for: {required - covered}"


def test_required_skills_exist_in_taxonomy():
    tax = load_taxonomy()
    valid = {s.id for s in tax.all()}
    jobs = _json("jobs.json")
    unknown = {s for j in jobs for s in j["required_skills"] + j["preferred_skills"] if s not in valid}
    assert not unknown, f"unknown skill ids in jobs.json: {unknown}"
