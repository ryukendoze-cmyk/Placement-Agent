import math

from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore


def test_hashing_embedder_is_deterministic_normalized_384():
    e = HashingEmbedder()
    a, b = e.embed("python fastapi postgres"), e.embed("python fastapi postgres")
    assert a == b
    assert len(a) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-6)

def test_similar_text_scores_higher():
    e = HashingEmbedder()
    near = cosine(e.embed("python fastapi postgresql rest api"), e.embed("python fastapi postgres"))
    far = cosine(e.embed("react css html styling"), e.embed("python fastapi postgres"))
    assert near > far

def test_memory_store_search_topk():
    e = HashingEmbedder()
    s = MemoryVectorStore()
    s.upsert("job", "j1", e.embed("python fastapi"))
    s.upsert("job", "j2", e.embed("react css"))
    res = s.search("job", e.embed("python fastapi sql"), k=2)
    assert res[0][0] == "j1" and len(res) == 2

def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))
