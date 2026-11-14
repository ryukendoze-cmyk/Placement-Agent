"""Skill-verification micro-quiz (Phase 5).

Serves 2 short-answer questions per plan item from data/questions.json plus
domain-specific supplemental banks for catalog/job skills
(fallback cascade: skill_id exact -> _PARENT_MAP canonical parent -> "generic").
Grading uses deterministic word-boundary, negation-aware keyword matching
against the question's `keywords` (no LLM, offline-safe).

GET never serializes the grading key (`keywords`) or any `explanation`; POST
grades server-side only and, on pass (>=70% of questions meeting the
keyword threshold), triggers the shared closed-loop completion path.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Canonical parent for a skill when the exact skill has no quiz questions;
# aligned with app/plan/prereq.py (PREREQUISITE_GRAPH).
_PARENT_MAP = {
    "fastapi": "python",
    "django": "python",
    "flask": "python",
    "pandas": "python",
    "numpy": "python",
    "pytest": "python",
    "celery": "python",
    "postgresql": "sql",
    "mysql": "sql",
    "sqlite": "sql",
    "dbt": "sql",
    "sql_analytics": "sql",
    "bigquery": "sql",
    "etl": "sql",
    "data_modeling": "sql",
    "css": "html",
    "javascript_web": "html",
    "javascript": "html",
    "typescript": "javascript_web",
    "react": "javascript_web",
    "nextjs": "react",
    "vue": "javascript_web",
    "angular": "typescript",
    "tailwind": "css",
    "bootstrap": "css",
    "docker": "linux",
    "kubernetes": "docker",
    "bash": "linux",
    "nginx": "linux",
    "ci_cd": "git",
    "github": "git",
    "terraform": "aws",
    "ansible": "linux",
    "monitoring": "linux",
    "deep_learning": "machine_learning",
    "pytorch": "deep_learning",
    "tensorflow": "deep_learning",
    "nlp": "deep_learning",
    "llms": "nlp",
    "computer_vision": "deep_learning",
    "machine_learning": "statistics",
    "sklearn": "statistics",
    "ab_testing": "statistics",
    "playwright": "test_automation",
    "selenium": "test_automation",
    "jest": "test_automation",
}

PASSING_SCORE = 70          # percent of questions graded correct (requires 2/2 on 2-question quiz)
QUESTIONS_PER_QUIZ = 2
HIT_RATIO = 0.5             # a question is correct when >=50% of its non-negated keywords appear
EXPLANATION_TEMPLATES = {
    "git": "Answer should cover the core git concepts referenced by the prompt.",
    "default": "A strong answer demonstrates mastery of core concepts and "
               "principles with concrete reasoning.",
}


_SUPPLEMENTAL_SKILL_QUESTIONS: tuple[dict, ...] = (
    {"id": "q-supp-ab_testing-1", "skill_id": "ab_testing", "difficulty": 1, "question": "How do you randomize users and define a null hypothesis in an A/B test?", "keywords": ["randomization", "control", "hypothesis", "metric"]},
    {"id": "q-supp-ab_testing-2", "skill_id": "ab_testing", "difficulty": 2, "question": "How do statistical power, sample size, and p-value guard against false positives?", "keywords": ["power", "sample", "significance", "p-value"]},
    {"id": "q-supp-agile-1", "skill_id": "agile", "difficulty": 1, "question": "What are the core roles and ceremonies in a Scrum sprint?", "keywords": ["sprint", "backlog", "retrospective", "standup"]},
    {"id": "q-supp-agile-2", "skill_id": "agile", "difficulty": 2, "question": "How do story points and velocity help a team plan sprint scope?", "keywords": ["velocity", "estimate", "iteration", "priority"]},
    {"id": "q-supp-ansible-1", "skill_id": "ansible", "difficulty": 1, "question": "How do Ansible playbooks, inventory files, and SSH configure hosts without agents?", "keywords": ["playbook", "inventory", "yaml", "ssh"]},
    {"id": "q-supp-ansible-2", "skill_id": "ansible", "difficulty": 2, "question": "Why does idempotency matter in Ansible roles and task modules?", "keywords": ["idempotent", "role", "module", "state"]},
    {"id": "q-supp-bootstrap-1", "skill_id": "bootstrap", "difficulty": 1, "question": "How does Bootstrap's 12-column responsive grid use breakpoints?", "keywords": ["grid", "column", "breakpoint", "responsive"]},
    {"id": "q-supp-bootstrap-2", "skill_id": "bootstrap", "difficulty": 2, "question": "How do you customize Bootstrap components and utility classes via Sass variables?", "keywords": ["utility", "component", "sass", "variable"]},
    {"id": "q-supp-chrome_devtools-1", "skill_id": "chrome_devtools", "difficulty": 1, "question": "How do you inspect failed API calls and payload headers in Chrome DevTools?", "keywords": ["network", "status", "header", "request"]},
    {"id": "q-supp-chrome_devtools-2", "skill_id": "chrome_devtools", "difficulty": 2, "question": "How do you diagnose layout shift or long tasks with the Performance panel?", "keywords": ["performance", "flamegraph", "breakpoint", "console"]},
    {"id": "q-supp-communication-1", "skill_id": "communication", "difficulty": 1, "question": "How do you tailor technical trade-offs for non-technical stakeholders?", "keywords": ["audience", "clarity", "trade-off", "impact"]},
    {"id": "q-supp-communication-2", "skill_id": "communication", "difficulty": 2, "question": "What structure makes design docs and incident postmortems actionable?", "keywords": ["context", "decision", "summary", "action"]},
    {"id": "q-supp-data_visualization-1", "skill_id": "data_visualization", "difficulty": 1, "question": "How do you choose between bar, line, and scatter charts for a dataset?", "keywords": ["distribution", "trend", "comparison", "axis"]},
    {"id": "q-supp-data_visualization-2", "skill_id": "data_visualization", "difficulty": 2, "question": "How do color encoding and hierarchy prevent misleading charts?", "keywords": ["scale", "contrast", "label", "clarity"]},
    {"id": "q-supp-dbt-1", "skill_id": "dbt", "difficulty": 1, "question": "How do ref() and staging/mart SQL models structure a dbt project?", "keywords": ["model", "ref", "staging", "mart"]},
    {"id": "q-supp-dbt-2", "skill_id": "dbt", "difficulty": 2, "question": "How do schema tests and incremental materializations improve warehouse reliability?", "keywords": ["test", "incremental", "unique", "materialization"]},
    {"id": "q-supp-excel-1", "skill_id": "excel", "difficulty": 1, "question": "When would you use XLOOKUP/INDEX-MATCH instead of nested IF formulas?", "keywords": ["lookup", "formula", "range", "match"]},
    {"id": "q-supp-excel-2", "skill_id": "excel", "difficulty": 2, "question": "How do Pivot Tables summarize multi-dimensional metrics cleanly?", "keywords": ["pivot", "aggregate", "filter", "group"]},
    {"id": "q-supp-figma-1", "skill_id": "figma", "difficulty": 1, "question": "How do Figma Components, Variants, and Auto Layout streamline responsive UI specs?", "keywords": ["component", "variant", "layout", "constraint"]},
    {"id": "q-supp-figma-2", "skill_id": "figma", "difficulty": 2, "question": "How do Design Tokens and Dev Mode handoff CSS spacing and colors accurately?", "keywords": ["token", "style", "inspect", "spacing"]},
    {"id": "q-supp-github-1", "skill_id": "github", "difficulty": 1, "question": "How do Pull Requests, branch protection, and code reviews maintain repo quality?", "keywords": ["pull request", "review", "branch", "protection"]},
    {"id": "q-supp-github-2", "skill_id": "github", "difficulty": 2, "question": "How do GitHub Actions workflows automate tests and status checks on push?", "keywords": ["workflow", "action", "ci", "check"]},
    {"id": "q-supp-hadoop-1", "skill_id": "hadoop", "difficulty": 1, "question": "How do HDFS block replication and NameNode/DataNode roles provide fault tolerance?", "keywords": ["hdfs", "block", "replication", "namenode"]},
    {"id": "q-supp-hadoop-2", "skill_id": "hadoop", "difficulty": 2, "question": "How do MapReduce and YARN schedule distributed batch jobs across nodes?", "keywords": ["mapreduce", "yarn", "shuffle", "cluster"]},
    {"id": "q-supp-jira-1", "skill_id": "jira", "difficulty": 1, "question": "How do Epics, Stories, and Subtasks organize backlog prioritization in Jira?", "keywords": ["epic", "story", "backlog", "workflow"]},
    {"id": "q-supp-jira-2", "skill_id": "jira", "difficulty": 2, "question": "How do JQL filters and sprint burndown charts track delivery blockers?", "keywords": ["jql", "sprint", "status", "blocker"]},
    {"id": "q-supp-kotlin-1", "skill_id": "kotlin", "difficulty": 1, "question": "How does Kotlin's nullable type system prevent NullPointerExceptions at compile time?", "keywords": ["nullable", "safe call", "elvis", "type"]},
    {"id": "q-supp-kotlin-2", "skill_id": "kotlin", "difficulty": 2, "question": "How do data classes and coroutines simplify asynchronous domain modeling?", "keywords": ["coroutine", "suspend", "data class", "immutable"]},
    {"id": "q-supp-manual_testing-1", "skill_id": "manual_testing", "difficulty": 1, "question": "What belongs in a reproducible test case and bug report?", "keywords": ["reproduce", "expected", "actual", "step"]},
    {"id": "q-supp-manual_testing-2", "skill_id": "manual_testing", "difficulty": 2, "question": "How do boundary-value analysis and exploratory testing uncover edge cases?", "keywords": ["boundary", "edge", "regression", "exploratory"]},
    {"id": "q-supp-matplotlib-1", "skill_id": "matplotlib", "difficulty": 1, "question": "Why should you prefer Matplotlib's object-oriented Figure/Axes API over pyplot state?", "keywords": ["figure", "axes", "plot", "subplot"]},
    {"id": "q-supp-matplotlib-2", "skill_id": "matplotlib", "difficulty": 2, "question": "How do you format axis ticks, legends, and DPI exports for publication figures?", "keywords": ["legend", "label", "tick", "dpi"]},
    {"id": "q-supp-mysql-1", "skill_id": "mysql", "difficulty": 1, "question": "How does InnoDB locking and ACID transaction isolation behave under concurrent writes?", "keywords": ["innodb", "transaction", "acid", "isolation"]},
    {"id": "q-supp-mysql-2", "skill_id": "mysql", "difficulty": 2, "question": "How do you read EXPLAIN output and composite B-Tree indexes to tune slow MySQL queries?", "keywords": ["explain", "index", "btree", "scan"]},
    {"id": "q-supp-playwright-1", "skill_id": "playwright", "difficulty": 1, "question": "Why do Playwright auto-waiting locators reduce flakiness compared to sleep timers?", "keywords": ["locator", "auto-wait", "assertion", "flaky"]},
    {"id": "q-supp-playwright-2", "skill_id": "playwright", "difficulty": 2, "question": "How do BrowserContexts and trace viewer isolate and debug E2E test failures?", "keywords": ["context", "trace", "isolation", "headless"]},
    {"id": "q-supp-postman-1", "skill_id": "postman", "difficulty": 1, "question": "How do Postman environments, variables, and collections keep API tests portable?", "keywords": ["collection", "environment", "variable", "request"]},
    {"id": "q-supp-postman-2", "skill_id": "postman", "difficulty": 2, "question": "How do Postman pre-request and test scripts assert status codes and JSON schemas via Newman?", "keywords": ["assertion", "script", "status", "newman"]},
    {"id": "q-supp-powerbi-1", "skill_id": "powerbi", "difficulty": 1, "question": "How do Power Query transformations and star-schema relationships clean source tables?", "keywords": ["power query", "schema", "relationship", "model"]},
    {"id": "q-supp-powerbi-2", "skill_id": "powerbi", "difficulty": 2, "question": "What is the difference between a DAX measure and a calculated column under filter context?", "keywords": ["dax", "measure", "filter", "context"]},
    {"id": "q-supp-presentation-1", "skill_id": "presentation", "difficulty": 1, "question": "How do you structure a technical demo around a clear problem, architecture, and outcome?", "keywords": ["narrative", "problem", "outcome", "visual"]},
    {"id": "q-supp-presentation-2", "skill_id": "presentation", "difficulty": 2, "question": "How do concise slides and live verification metrics engage decision-makers?", "keywords": ["metric", "concise", "takeaway", "evidence"]},
    {"id": "q-supp-rails-1", "skill_id": "rails", "difficulty": 1, "question": "How does Rails MVC routing map HTTP verbs to controller actions and ActiveRecord models?", "keywords": ["mvc", "route", "controller", "activerecord"]},
    {"id": "q-supp-rails-2", "skill_id": "rails", "difficulty": 2, "question": "How do database migrations and eager loading avoid N+1 query bottlenecks in Rails?", "keywords": ["migration", "includes", "n+1", "association"]},
    {"id": "q-supp-ruby-1", "skill_id": "ruby", "difficulty": 1, "question": "How do blocks, Procs, and Enumerable methods express transformations in Ruby?", "keywords": ["block", "enumerable", "yield", "symbol"]},
    {"id": "q-supp-ruby-2", "skill_id": "ruby", "difficulty": 2, "question": "How do modules and mixins share behavior cleanly across Ruby classes?", "keywords": ["module", "mixin", "include", "class"]},
    {"id": "q-supp-selenium-1", "skill_id": "selenium", "difficulty": 1, "question": "Why should WebDriver tests use explicit WebDriverWait conditions instead of implicit waits?", "keywords": ["webdriver", "explicit", "wait", "locator"]},
    {"id": "q-supp-selenium-2", "skill_id": "selenium", "difficulty": 2, "question": "How does the Page Object Model keep Selenium test suites maintainable?", "keywords": ["page object", "selector", "maintainability", "session"]},
    {"id": "q-supp-sql_analytics-1", "skill_id": "sql_analytics", "difficulty": 1, "question": "How do window functions like ROW_NUMBER, RANK, and OVER() partition analytical rows?", "keywords": ["window", "partition", "rank", "over"]},
    {"id": "q-supp-sql_analytics-2", "skill_id": "sql_analytics", "difficulty": 2, "question": "How do CTEs and cohort aggregations compute retention or rolling 7-day metrics?", "keywords": ["cte", "cohort", "aggregation", "rolling"]},
    {"id": "q-supp-sqlite-1", "skill_id": "sqlite", "difficulty": 1, "question": "What are SQLite's embedded serverless trade-offs for local apps vs concurrent writes?", "keywords": ["embedded", "file", "lock", "concurrency"]},
    {"id": "q-supp-sqlite-2", "skill_id": "sqlite", "difficulty": 2, "question": "How does WAL (Write-Ahead Logging) mode improve concurrent read/write throughput in SQLite?", "keywords": ["wal", "journal", "pragma", "transaction"]},
    {"id": "q-supp-swift-1", "skill_id": "swift", "difficulty": 1, "question": "How do Swift Optionals, guard let, and protocols enforce type safety?", "keywords": ["optional", "guard", "protocol", "unwrap"]},
    {"id": "q-supp-swift-2", "skill_id": "swift", "difficulty": 2, "question": "How do Swift value-type structs and async/await manage state safely on iOS?", "keywords": ["struct", "value", "async", "arc"]},
    {"id": "q-supp-tableau-1", "skill_id": "tableau", "difficulty": 1, "question": "How do Dimensions vs Measures and calculated fields control Tableau aggregations?", "keywords": ["dimension", "measure", "calculated", "aggregation"]},
    {"id": "q-supp-tableau-2", "skill_id": "tableau", "difficulty": 2, "question": "How do Level of Detail (LOD) expressions fix granularity independently of view filters?", "keywords": ["lod", "fixed", "granularity", "dashboard"]},
    {"id": "q-supp-unittest-1", "skill_id": "unittest", "difficulty": 1, "question": "How do TestCase setUp fixtures and assertion methods isolate unit behavior?", "keywords": ["testcase", "setup", "assert", "fixture"]},
    {"id": "q-supp-unittest-2", "skill_id": "unittest", "difficulty": 2, "question": "When should you use unittest.mock.patch to isolate external network dependencies?", "keywords": ["mock", "patch", "isolate", "dependency"]},
    # Second-question top-ups for skills with only 1 question in data/questions.json
    {"id": "q-supp-airflow-2", "skill_id": "airflow", "difficulty": 2, "question": "How do Airflow XComs, sensors, and catchup/backfill control DAG scheduling?", "keywords": ["dag", "sensor", "backfill", "xcom"]},
    {"id": "q-supp-azure-2", "skill_id": "azure", "difficulty": 2, "question": "How do Azure Managed Identities and Resource Groups isolate RBAC permissions?", "keywords": ["managed identity", "resource group", "rbac", "tenant"]},
    {"id": "q-supp-bigquery-2", "skill_id": "bigquery", "difficulty": 2, "question": "How do table partitioning and clustering cut BigQuery bytes scanned?", "keywords": ["partition", "cluster", "slot", "bytes"]},
    {"id": "q-supp-computer_vision-2", "skill_id": "computer_vision", "difficulty": 2, "question": "How do anchor boxes, IoU thresholds, and Non-Maximum Suppression work in object detection?", "keywords": ["iou", "bounding box", "nms", "anchor"]},
    {"id": "q-supp-data_modeling-2", "skill_id": "data_modeling", "difficulty": 2, "question": "When do you choose a star schema (facts/dimensions) over normalized 3NF tables?", "keywords": ["star schema", "fact", "dimension", "normalization"]},
    {"id": "q-supp-etl-2", "skill_id": "etl", "difficulty": 2, "question": "How do watermarks, CDC, and dead-letter queues make incremental ETL idempotent?", "keywords": ["cdc", "watermark", "idempotent", "incremental"]},
    {"id": "q-supp-express-2", "skill_id": "express", "difficulty": 2, "question": "How does Express middleware ordering and 4-arg error-handling middleware work?", "keywords": ["middleware", "next", "router", "error"]},
    {"id": "q-supp-grpc-2", "skill_id": "grpc", "difficulty": 2, "question": "When should you pick bidirectional gRPC streams and Protobuf schemas over JSON REST?", "keywords": ["protobuf", "http2", "stream", "contract"]},
    {"id": "q-supp-jest-2", "skill_id": "jest", "difficulty": 2, "question": "How do jest.spyOn, module mocks, and fake timers isolate async UI tests?", "keywords": ["spy", "mock", "timer", "async"]},
    {"id": "q-supp-kafka-2", "skill_id": "kafka", "difficulty": 2, "question": "How do consumer groups, partition keys, and offsets preserve ordering and replayability?", "keywords": ["partition", "offset", "consumer group", "broker"]},
    {"id": "q-supp-mlops-2", "skill_id": "mlops", "difficulty": 2, "question": "How do model registries, canary rollouts, and data drift alerts protect production ML?", "keywords": ["registry", "drift", "canary", "monitoring"]},
    {"id": "q-supp-monitoring-2", "skill_id": "monitoring", "difficulty": 2, "question": "How do SLIs, SLO error budgets, and Prometheus histogram percentiles drive alerting?", "keywords": ["sli", "slo", "percentile", "histogram"]},
    {"id": "q-supp-nginx-2", "skill_id": "nginx", "difficulty": 2, "question": "How do upstream pools, TLS termination, and proxy timeouts keep reverse proxies resilient?", "keywords": ["upstream", "proxy_pass", "tls", "timeout"]},
    {"id": "q-supp-nlp-2", "skill_id": "nlp", "difficulty": 2, "question": "How do subword tokenization, attention masks, and padding affect transformer inference?", "keywords": ["token", "attention", "padding", "subword"]},
    {"id": "q-supp-numpy-2", "skill_id": "numpy", "difficulty": 2, "question": "How do NumPy broadcasting rules and vectorized views avoid Python loop copies?", "keywords": ["broadcasting", "vectorized", "shape", "view"]},
    {"id": "q-supp-sklearn-2", "skill_id": "sklearn", "difficulty": 2, "question": "Why must scalers/imputers sit inside a scikit-learn Pipeline during cross-validation?", "keywords": ["pipeline", "leakage", "cross-validation", "fit"]},
    {"id": "q-supp-tensorflow-2", "skill_id": "tensorflow", "difficulty": 2, "question": "How do tf.data pipelines and @tf.function graph tracing speed up training?", "keywords": ["tf.data", "prefetch", "graph", "gradient"]},
    {"id": "q-supp-test_automation-2", "skill_id": "test_automation", "difficulty": 2, "question": "How does the test pyramid balance fast unit checks, API contracts, and selective E2E tests?", "keywords": ["pyramid", "contract", "unit", "e2e"]},
    {"id": "q-supp-vite-2", "skill_id": "vite", "difficulty": 2, "question": "How does Vite use native ESM in dev and Rollup code-splitting in production builds?", "keywords": ["esm", "hmr", "rollup", "bundle"]},
)


@lru_cache(maxsize=1)
def _load_quiz_bank() -> tuple[dict, ...]:
    raw = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for q in raw:
        sk = q.get("skill_id")
        if sk:
            counts[sk] = counts.get(sk, 0) + 1
    combined = list(raw)
    for q in _SUPPLEMENTAL_SKILL_QUESTIONS:
        sk = q["skill_id"]
        if counts.get(sk, 0) < QUESTIONS_PER_QUIZ:
            combined.append(q)
            counts[sk] = counts.get(sk, 0) + 1
    return tuple(combined)


def questions_for_skill(skill_id: str) -> list[dict]:
    """All bank questions matching skill_id (exact, then parent, then generic)."""
    bank = _load_quiz_bank()
    exact = [q for q in bank if q.get("skill_id") == skill_id]
    if exact:
        return exact
    parent = _PARENT_MAP.get(skill_id)
    if parent:
        parent_qs = [q for q in bank if q.get("skill_id") == parent]
        if parent_qs:
            return parent_qs
    return [q for q in bank if q.get("skill_id") == "generic"]


def generate_quiz(skill_id: str, n: int = QUESTIONS_PER_QUIZ) -> list[dict]:
    """Deterministically pick n questions (stable by difficulty then id)."""
    pool = questions_for_skill(skill_id)
    if not pool:
        return []
    pool = sorted(pool, key=lambda q: (q.get("difficulty", 0), q.get("id", "")))
    picked: list[dict] = []
    i = 0
    while len(picked) < n and pool:
        picked.append(pool[i % len(pool)])
        i += 1
        if i >= len(pool):  # exhausted pool without hitting n
            break
    return picked


def _answer_key(skill_id: str) -> dict[str, list[str]]:
    """{question_id: keywords} for grading - never leaves the server on GET."""
    return {q["id"]: list(q.get("keywords", [])) for q in generate_quiz(skill_id)}


def _answers_for_item(item_id: str, skill_id: str) -> dict[str, str]:
    """A passing answer per question (for tests/seeded demos): the keywords
    joined into one short-answer string that always meets the hit threshold."""
    return {qid: " ".join(kws) for qid, kws in _answer_key(skill_id).items()}


def _keyword_present(text: str, kw: str) -> bool:
    """Match keyword on whole-token boundaries while rejecting directly negated mentions."""
    kw_l = re.sub(r"\s+", " ", str(kw).strip().lower())
    if not kw_l:
        return False
    plural_suffix = r"(?:s|es)?" if (len(kw_l) >= 4 and kw_l[-1].isalnum() and not kw_l.endswith("s")) else ""
    pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(kw_l)}{plural_suffix}(?![a-z0-9_])")
    for match in pattern.finditer(text):
        prefix = text[:match.start()]
        last_clause = re.split(r"[\n.,;:!?]", prefix)[-1]
        words_before = re.findall(r"[a-z0-9_']+", last_clause)[-4:]
        joined_tail = " ".join(words_before) + (" " if words_before else "")
        if not re.search(
            r"(?:^|\s)(?:not|never|no|without|avoid|anti|don't|doesn't|isn't|aren't|cannot|can't)\s+"
            r"(?:a\s+|an\s+|the\s+|any\s+|using\s+|to\s+)?$",
            joined_tail,
        ):
            return True
    return False


def _grade_question(answer: str, keywords: list[str]) -> tuple[bool, float]:
    """Deterministic word-boundary, negation-aware keyword hit ratio >= HIT_RATIO -> correct."""
    if not keywords:
        return True, 1.0
    text = re.sub(r"\s+", " ", (answer or "").lower()).strip()
    if not text:
        return False, 0.0
    hits = sum(1 for k in keywords if _keyword_present(text, str(k)))
    ratio = hits / len(keywords)
    return ratio >= HIT_RATIO, ratio


def check_answers(skill_id: str, answers: dict) -> tuple[bool, int, list[dict]]:
    """Grade submitted answers. Returns (passed, score, feedback_rows).

    `answers` maps question_id -> free-text answer.
    """
    quiz = generate_quiz(skill_id)
    if not quiz:
        return True, 100, []  # no questions = vacuously verified
    correct = 0
    rows = []
    for q in quiz:
        qid = q["id"]
        keywords = list(q.get("keywords", []))
        is_correct, ratio = _grade_question(str(answers.get(qid, "")), keywords)
        if is_correct:
            correct += 1
        rows.append({
            "question_id": qid,
            "correct": is_correct,
            "hit_ratio": round(ratio, 2),
            "explanation": EXPLANATION_TEMPLATES.get(skill_id, EXPLANATION_TEMPLATES["default"]),
        })
    total = len(quiz)
    score = round(100 * correct / total)
    passed = score >= PASSING_SCORE
    return passed, score, rows
