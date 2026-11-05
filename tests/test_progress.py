from sqlmodel import Session, select

from app.models import ProgressEvent


def _setup_plan(client_engine):
    client, eng = client_engine
    r = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker\n", "text/plain")},
        data={"name": "Progress Probe"},
    )
    sid = r.json()["id"]
    client.get(f"/api/students/{sid}/matches")
    r2 = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"})
    plan = r2.json()
    return client, eng, sid, plan


def test_completing_items_raises_readiness_monotonically(client_engine):
    client, eng, sid, plan = _setup_plan(client_engine)
    items = plan["items"]
    assert len(items) >= 1

    prev = -1
    for it in items:
        r = client.post(f"/api/plans/{plan['id']}/items/{it['id']}/complete")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["readiness"] >= prev
        prev = body["readiness"]
        assert 0 <= body["readiness"] <= 100

    # events persisted
    with Session(eng) as db:
        events = db.exec(select(ProgressEvent).where(ProgressEvent.student_id == sid)).all()
        assert len(events) == len(items)
        assert all(e.kind == "plan_item_done" for e in events)


def test_progress_endpoint_returns_ordered_series(client_engine):
    client, eng, sid, plan = _setup_plan(client_engine)
    for it in plan["items"]:
        client.post(f"/api/plans/{plan['id']}/items/{it['id']}/complete")

    r = client.get(f"/api/students/{sid}/progress")
    assert r.status_code == 200
    body = r.json()
    assert "readiness" in body and "events" in body and "best_match" in body
    series = body["readiness"]
    assert len(series) == len(plan["items"])
    # ordered by at ascending
    times = [p["at"] for p in series]
    assert times == sorted(times)
    # best_match present
    assert body["best_match"] is None or "score" in body["best_match"]
