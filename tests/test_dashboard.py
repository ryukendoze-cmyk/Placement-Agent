def test_index_lists_students(client_engine):
    client, _ = client_engine
    client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI\n", "text/plain")},
                data={"name": "Asha"})
    h = client.get("/")
    assert h.status_code == 200
    assert "Asha" in h.text
    assert "upload" in h.text.lower()


def test_profile_renders_name_and_skill_chip(client_engine):
    client, _ = client_engine
    sid = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI, Docker\n", "text/plain")},
                      data={"name": "Bharat", "branch": "CSE", "year": 3, "gpa": 8.1}).json()["id"]
    h = client.get(f"/students/{sid}")
    assert h.status_code == 200
    assert "Bharat" in h.text
    assert "Python" in h.text
    assert "skill-chips" in h.text


def test_resume_reupload_refreshes_skill_chips(client_engine):
    client, _ = client_engine
    sid = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python\n", "text/plain")},
                      data={"name": "Swap"}).json()["id"]
    r = client.post(f"/students/{sid}/resume",
                    files={"file": ("r.txt", b"Skills: Python, Kubernetes\n", "text/plain")},
                    headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert "kubernetes" in r.text.lower()


def _seed_with_matches(client, name="Meera"):
    sid = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL\n", "text/plain")},
                      data={"name": name}).json()["id"]
    client.get(f"/api/students/{sid}/matches")
    return sid


def test_matches_page_renders_explainable_cards(client_engine):
    client, _ = client_engine
    sid = _seed_with_matches(client)
    h = client.get(f"/students/{sid}/matches")
    assert h.status_code == 200
    assert "Missing" in h.text
    assert "Python" in h.text
    assert "Generate plan" in h.text


def test_matches_page_no_matches_state(client_engine):
    client, _ = client_engine
    sid = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python\n", "text/plain")},
                      data={"name": "NoMatch"}).json()["id"]
    h = client.get(f"/students/{sid}/matches")
    assert h.status_code == 200
    assert "Run matching" in h.text


def test_plan_page_and_htmx_complete(client_engine):
    client, _ = client_engine
    sid = _seed_with_matches(client, name="Planner")
    plan = client.post(f"/api/students/{sid}/plans", json={}).json()
    assert plan["items"], "expected a non-empty plan"
    h = client.get(f"/students/{sid}/plan")
    assert h.status_code == 200
    assert "plan-bar" in h.text
    assert plan["items"][0]["course_title"] in h.text or plan["items"][0]["skill_id"] in h.text
    item = plan["items"][0]
    r = client.post(f"/api/plans/{plan['id']}/items/{item['id']}/complete", headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert "readiness" in r.json()


def test_progress_page_renders_chart(client_engine):
    client, _ = client_engine
    sid = _seed_with_matches(client, name="Progress")
    plan = client.post(f"/api/students/{sid}/plans", json={}).json()
    client.post(f"/api/plans/{plan['id']}/items/{plan['items'][0]['id']}/complete")
    h = client.get(f"/students/{sid}/progress")
    assert h.status_code == 200
    assert "readinessChart" in h.text
    assert "chart.js" in h.text or "Chart" in h.text


def test_interviews_page_renders(client_engine):
    client, _ = client_engine
    sid = _seed_with_matches(client, name="Intv")
    h = client.get(f"/students/{sid}/interviews")
    assert h.status_code == 200
    assert "interview" in h.text.lower()
    assert "Start" in h.text
