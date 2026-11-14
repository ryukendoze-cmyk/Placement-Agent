import hashlib
import logging
import math
import os
import re
from typing import Protocol

logger = logging.getLogger(__name__)


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
    """sentence-transformers all-MiniLM-L6-v2 (384-dim) with offline HashingEmbedder fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._m = None
        self._fallback: HashingEmbedder | None = None
        offline = os.getenv("HF_HUB_OFFLINE") == "1"
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {"local_files_only": True} if offline else {}
            self._m = SentenceTransformer(model_name, **kwargs)
            self.dim = getattr(
                self._m,
                "get_embedding_dimension",
                self._m.get_sentence_embedding_dimension,
            )()
        except Exception as exc:
            logger.warning(
                "LocalEmbedder failed to load %s (HF_HUB_OFFLINE=%s): %s; "
                "falling back to HashingEmbedder(dim=384)",
                model_name,
                os.getenv("HF_HUB_OFFLINE", "0"),
                exc,
            )
            self._fallback = HashingEmbedder(dim=384)
            self.dim = 384

    def embed(self, text: str) -> list[float]:
        if self._fallback is not None:
            return self._fallback.embed(text)
        return self._m.encode([text], normalize_embeddings=True)[0].tolist()
