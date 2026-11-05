import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic, dependency-free embedder used by unit tests and offline mode."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9+#.]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


class LocalEmbedder:
    """sentence-transformers all-MiniLM-L6-v2 (384-dim). Loaded lazily."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model_name)
        self.dim = getattr(self._m, "get_embedding_dimension", self._m.get_sentence_embedding_dimension)()

    def embed(self, text: str) -> list[float]:
        return self._m.encode([text], normalize_embeddings=True)[0].tolist()
