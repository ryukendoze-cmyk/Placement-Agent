"""Job ranking pipeline shared by the API, the CLI demo and (Phase 2) the planner.

Pure layer: no FastAPI, no SQLModel imports — jobs can be SQLModel rows or any
object exposing .id/.description/.required_skills/.preferred_skills (the CLI
passes SimpleNamespace built from data/jobs.json).
"""


from ..embeddings.embedder import Embedder
from ..embeddings.store import VectorStore, cosine
from ..taxonomy.taxonomy import Taxonomy
from .engine import JobScore, score_job


def rank_jobs(
    jobs: list,
    student_vec: list[float],
    student_skills: dict[str, list[str]],
    embedder: Embedder,
    taxonomy: Taxonomy,
    *,
    vector_store: VectorStore | None = None,
    job_vec_cache: dict[str, list[float]] | None = None,
    top: int | None = None,
    levels: dict[str, str] | None = None,
    student=None,
    ann_pre_filter_k: int | None = None,
) -> list[tuple[object, JobScore]]:
    """Score and rank jobs for a student, highest score first.

    vector_store / job_vec_cache: optional caching of per-job vectors. The API
    passes its app-lifetime PgVectorStore/MemoryVectorStore plus the
    AppContext.job_vectors dict so repeated requests skip re-embedding; the CLI
    passes neither and embeds on the fly. Embedding text is `job.description`
    everywhere (seed, API lazy path, CLI) - keep them in sync if it changes.

    ann_pre_filter_k (M21): when set - or automatically when the catalog has
    more than 500 jobs and the vector store supports search() - run a two-stage
    rerank: retrieve top-k candidate job ids by ANN cosine similarity, union
    them with every job sharing at least one required skill the student has,
    and run exact score_job only on that candidate set. Catalogs <= 500 jobs
    always score every job directly so results stay byte-for-byte identical.
    """
    supports_search = (
        vector_store is not None
        and callable(getattr(vector_store, "search", None))
    )
    use_ann = supports_search and (
        ann_pre_filter_k is not None or len(jobs) > 500
    )

    candidates = jobs
    if use_ann:
        effective_k = (ann_pre_filter_k if ann_pre_filter_k is not None
                       else min(len(jobs), 200))
        candidate_ids = {
            ref for ref, _ in vector_store.search("job", student_vec, k=effective_k)
        }
        owned = set(student_skills.keys())
        for job in jobs:
            if job.id in candidate_ids:
                continue
            for skill in (getattr(job, "required_skills", None) or []):
                tax_skill = taxonomy.get_opt(skill)
                sid = tax_skill.id if tax_skill else skill
                if sid in owned or skill in owned:
                    candidate_ids.add(job.id)
                    break
        candidates = [job for job in jobs if job.id in candidate_ids]

    results: list[tuple[object, JobScore]] = []
    for job in candidates:
        jv = None
        if job_vec_cache is not None:
            jv = job_vec_cache.get(job.id)
        if jv is None and vector_store is not None:
            jv = vector_store.get("job", job.id)
        if jv is None:
            jv = embedder.embed(job.description)
            if vector_store is not None:
                vector_store.upsert("job", job.id, jv)
            if job_vec_cache is not None:
                job_vec_cache[job.id] = jv
        sim = cosine(student_vec, jv)
        results.append((job, score_job(student_skills, job, sim, taxonomy,
                                       levels=levels, student=student)))
    results.sort(key=lambda pair: -pair[1].score)
    if top is not None:
        results = results[: max(top, 0)]
    return results
