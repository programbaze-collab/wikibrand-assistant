"""Эмбеддинги через fastembed: ONNX вместо PyTorch, модель весит 0,22 ГБ и работает на CPU."""

from functools import cache

import numpy as np
from fastembed import TextEmbedding

from .config import EMBEDDING_MODEL


@cache
def _model() -> TextEmbedding:
    return TextEmbedding(EMBEDDING_MODEL)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    # После нормализации косинусная близость — это просто скалярное произведение.
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed_passages(texts: list[str]) -> np.ndarray:
    return _normalize(np.array(list(_model().embed(texts, batch_size=64)), dtype=np.float32))


def embed_query(text: str) -> np.ndarray:
    return _normalize(np.array(list(_model().embed([text])), dtype=np.float32))[0]
