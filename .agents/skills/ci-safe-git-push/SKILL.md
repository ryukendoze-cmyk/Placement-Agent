---
name: ci-safe-git-push
description: >-
  Pre-push CI verification and post-push GitHub Actions check workflow.
  Use before and after every `git push` to guarantee CI (`quality` and `integration`)
  passes on the first try without local-environment false positives.
---

# CI-Safe Git Push

Always run this checklist before and after `git push` in this repository.

## 1. Pre-Push Local Hygiene

1. Ensure no secrets, build artifacts, or local databases are staged:
   - Never commit `.env` (only `.env.example`), `node_modules/`, `.venv/`, `*.db`, or `.pytest_temp/`.
2. Sync to the GitHub mirror folder (`ps06-placement-agent-github`) if working in `ps06-placement-agent`, excluding `.git`, `.venv`, `node_modules`, `.pytest_cache`, `.pytest_temp`, `.ruff_cache`, `__pycache__`, `dist`, `build`, `.gradle`, `*.pyc`, `*.db`, `*.log`.

## 2. Exact CI Parity Checks (Simulate `ubuntu-latest` Without Local DB)

Run all checks from `.github/workflows/ci.yml`:

1. **Lockfile & Lint**:
   ```powershell
   uv lock --check
   uv run ruff check .
   ```
2. **Catch Import-Time / Local-DB Leakage (Critical)**:
   Local machines often have PostgreSQL running on `127.0.0.1:5432/ps06`, whereas CI `quality` has **no** Postgres and CI `integration` only has `ps06_test`. Never allow tests or `app/api/main.py` to import `app.api.__main__` (which executes `_build_ctx()` -> `init_db()` on `settings.database_url` at import time).
   Verify with an unreachable `DATABASE_URL`:
   ```powershell
   $env:DATABASE_URL = "postgresql+psycopg://postgres@127.0.0.1:59999/nonexistent"
   uv run pytest -m "not integration" -q
   Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
   ```
3. **Integration Tests** (when local `ps06_test` Postgres is running):
   ```powershell
   uv run pytest -m integration -q
   ```

## 3. Post-Push GitHub Actions Verification

After `git push origin main`:

1. Query the GitHub Actions API for the pushed commit SHA until `status == "completed"`:
   ```powershell
   python -c "
   import time, urllib.request, json
   sha = '<COMMIT_SHA>'
   url = 'https://api.github.com/repos/ryukendoze-cmyk/Placement-Agent/actions/runs?per_page=5'
   for _ in range(30):
       runs = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'ci-check'})).read())['workflow_runs']
       match = next((r for r in runs if r['head_sha'].startswith(sha)), None)
       if match and match['status'] == 'completed':
           print(match['head_sha'][:7], match['status'], match['conclusion'], match['html_url'])
           break
       time.sleep(10)
   "
   ```
2. Only report the push as successful once `conclusion == "success"`. If any job fails, inspect the failed step and fix it before finishing.
