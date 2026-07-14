"""Embedding adapters for production BGE retrieval and deterministic tests."""

from __future__ import annotations

from hashlib import blake2b
import re
from typing import Protocol, Sequence

import numpy as np


DEFAULT_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
DEFAULT_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(array / norms, dtype=np.float32)


class Encoder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, query: str) -> np.ndarray: ...


class BGEEncoder:
    """Lazy sentence-transformers adapter used by production indexing."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self.query_prefix = query_prefix
        self.device = device
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for production BGE encoding; "
                    "install requirements.txt"
                ) from exc
            kwargs = {"device": self.device} if self.device else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    @property
    def dimension(self) -> int:
        value = self._load_model().get_sentence_embedding_dimension()
        if value is None:
            raise RuntimeError("the embedding model did not report its dimension")
        return int(value)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        values = self._load_model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return normalize_rows(values)

    def encode_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query must not be blank")
        value = self._load_model().encode(
            [self.query_prefix + query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return normalize_rows(value)[0]


class HashingEncoder:
    """Deterministic lightweight encoder restricted to tests and local smoke runs."""

    def __init__(self, dimension: int = 512) -> None:
        if dimension < 64:
            raise ValueError("test encoder dimension must be at least 64")
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return f"fixture-hashing-{self.dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _features(text: str) -> list[str]:
        normalized = text.lower()
        features = re.findall(r"[a-z]+\d*|\d+(?:\.\d+)?%?", normalized)
        for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
            features.extend(run)
            features.extend(run[index : index + 2] for index in range(len(run) - 1))
        return features

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in self._features(text):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.dimension
            vector[index] += 1.0
        return normalize_rows(vector)[0]

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([self._encode(text) for text in texts]).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query must not be blank")
        return self._encode(query.strip())

