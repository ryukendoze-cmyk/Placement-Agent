"""Skill prerequisite DAG (Phase 5).

Edge semantics: skill A -> [B, ...] means "A needs B first" (B is a
prerequisite of A). `topological_order` expands a target-skill set with its
transitive prerequisites and returns a deterministic consumer-before-
prerequisite order (consumers have their prerequisites scheduled earlier),
breaking ties by degree then alphabetically. Cycles are tolerated: remaining
nodes are appended in deterministic order rather than raising.
"""
from __future__ import annotations

PREREQUISITE_GRAPH: dict[str, list[str]] = {
    # backend & languages
    "fastapi": ["python"],
    "django": ["python"],
    "flask": ["python"],
    "spring": ["java"],
    "rails": [],
    "express": ["nodejs"],
    "grpc": ["rest_api"],
    "graphql": ["rest_api"],
    "rest_api": ["javascript_web"],
    "microservices": ["rest_api", "docker"],
    "kafka": ["java"],
    "rabbitmq": ["microservices"],
    "celery": ["python"],
    "auth_jwt": ["rest_api"],
    "websockets": ["javascript_web"],
    # frontend & web
    "css": ["html"],
    "javascript_web": ["html"],
    "javascript": ["javascript_web"],
    "typescript": ["javascript_web"],
    "bootstrap": ["css"],
    "tailwind": ["css"],
    "vite": ["javascript_web"],
    "webpack": ["javascript_web"],
    "redux": ["javascript_web"],
    "responsive_design": ["css", "html"],
    "accessibility": ["html", "css"],
    "react": ["javascript_web", "html", "css"],
    "vue": ["javascript_web", "html", "css"],
    "angular": ["typescript", "html", "css"],
    "nextjs": ["react", "typescript"],
    "nodejs": ["javascript_web"],
    # databases & storage
    "postgresql": ["sql"],
    "mysql": ["sql"],
    "mongodb": ["data_modeling"],
    "redis": ["sql"],
    "sqlite": ["sql"],
    "elasticsearch": ["data_modeling"],
    "data_modeling": ["sql"],
    "dbt": ["sql", "data_modeling"],
    "bigquery": ["sql"],
    "sql_analytics": ["sql"],
    "spark": ["sql"],
    "airflow": ["etl"],
    "etl": ["sql"],
    "hadoop": ["java"],
    # devops & infrastructure
    "bash": ["linux"],
    "docker": ["linux"],
    "kubernetes": ["docker"],
    "ci_cd": ["git"],
    "github": ["git"],
    "nginx": ["linux"],
    "terraform": ["aws"],
    "ansible": ["linux"],
    "monitoring": ["linux"],
    "aws": [],
    "azure": [],
    "gcp": [],
    "linux": [],
    # data science & machine learning
    "pandas": ["python", "numpy"],
    "numpy": ["python"],
    "matplotlib": ["python"],
    "statistics": ["python"],
    "ab_testing": ["statistics"],
    "excel": [],
    "powerbi": ["excel"],
    "tableau": ["excel"],
    "machine_learning": ["python", "statistics", "numpy"],
    "sklearn": ["python", "statistics", "numpy"],
    "deep_learning": ["machine_learning", "numpy"],
    "pytorch": ["python", "deep_learning"],
    "tensorflow": ["python", "deep_learning"],
    "nlp": ["deep_learning", "python"],
    "llms": ["nlp", "deep_learning"],
    "computer_vision": ["deep_learning", "numpy"],
    "mlops": ["machine_learning"],
    # quality assurance & tooling
    "test_automation": ["unittest"],
    "playwright": ["test_automation", "javascript_web"],
    "selenium": ["test_automation", "python"],
    "jest": ["test_automation", "javascript_web"],
    "postman": ["rest_api"],
}


def reachable(graph: dict[str, list[str]], skill: str) -> set[str]:
    """All skills reachable from `skill` by following prerequisite edges (incl. itself)."""
    seen: set[str] = set()
    stack = [skill]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, []))
    return seen


def topological_order(
    graph: dict[str, list[str]],
    targets: list[str],
    *,
    exclude: set[str] | None = None,
    priority: dict[str, tuple[int, int]] | None = None,
) -> list[str]:
    """Order `targets` so every prerequisite precedes its consumers.

    1. Expand `targets` with all transitive prerequisites reachable via graph
       edges - a prerequisite the student already owns (in `exclude`) is never
       emitted as a schedulable gap; it still constrains ordering if it is also
       a target, but is dropped from the result.
    2. Kahn's algorithm: repeatedly emit the ready node with
       (fewest outstanding prerequisites, then `priority` - defaults to
       (1, 999), then alphabetical tie-break).
    3. Cycles (or prerequisites outside the subgraph) fall back to
       alphabetical emission instead of raising - plan generation never 500s.

    `priority` maps a skill to `(rank, hours)` where lower rank = higher
    urgency (0 = required gap, 1 = preferred) and lower hours wins ties, so
    required/short courses are scheduled before optional/long ones at the
    same dependency level.
    """
    targets = [t for t in dict.fromkeys(targets or [])]
    if not targets:
        return []
    excluded = set(exclude or ())
    prio = priority or {}

    # closure: expand targets with all transitive prerequisites reachable via
    # PREREQUISITE_GRAPH edges - a prerequisite that is itself another target's
    # prereq must be scheduled, even if it wasn't requested directly.
    all_nodes: set[str] = set()
    for t in targets:
        all_nodes |= reachable(graph, t)
    sub: dict[str, list[str]] = {
        node: [p for p in graph.get(node, []) if p in all_nodes]
        for node in all_nodes
    }

    remaining = set(all_nodes)
    ordered: list[str] = []
    while remaining:
        ready = [n for n in remaining if all(p not in remaining for p in sub[n])]
        if not ready:
            # cycle guard: emit deterministically, keep only edges among emitted later
            ordered.extend(sorted(remaining))
            break
        ready.sort(key=lambda n: (len(sub[n]), prio.get(n, (1, 999)), n))
        pick = ready[0]
        ordered.append(pick)
        remaining.discard(pick)
    return [n for n in ordered if n not in excluded]

