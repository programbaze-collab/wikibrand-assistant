"""Гибридный поиск: python -m assistant.search "как отличить подделку casio"

Векторный поиск находит куски по смыслу («как распознать фейк» ~ «как отличить подделку»),
но плохо — по точным названиям вроде «GA-2100». Поиск по словам — наоборот.
Результаты объединяются через Reciprocal Rank Fusion: важен ранг куска в каждом списке,
а не сами оценки — у косинусной близости и BM25 разные шкалы, складывать их нельзя.
"""

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import EMBEDDING_MODEL, INDEX_DIR
from .embeddings import embed_query
from .text import to_fts_query

RRF_K = 60  # Стандартная константа RRF: сглаживает разницу между первыми местами.
CANDIDATES = 30  # Сколько кандидатов берёт каждый из двух поисков до объединения.


@dataclass(frozen=True)
class Hit:
    rowid: int
    brand: str
    section: str
    text: str
    url: str
    score: float


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: сумма 1 / (k + ранг) по всем спискам, где есть кусок."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, rowid in enumerate(ranking, start=1):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + rank)
    return scores


class Index:
    def __init__(self, index_dir: Path = INDEX_DIR):
        model = (index_dir / "model.txt").read_text(encoding="utf-8").strip()
        if model != EMBEDDING_MODEL:
            raise RuntimeError(f"Индекс построен моделью {model}, а настроена {EMBEDDING_MODEL}: пересоберите индекс")
        self.vectors = np.load(index_dir / "embeddings.npy")
        self.db = sqlite3.connect(index_dir / "chunks.db", check_same_thread=False)

    def vector_search(self, question: str, limit: int = CANDIDATES) -> list[int]:
        scores = self.vectors @ embed_query(question)
        top = np.argpartition(-scores, limit)[:limit]
        return [int(i) for i in top[np.argsort(-scores[top])]]

    def keyword_search(self, question: str, limit: int = CANDIDATES) -> list[int]:
        query = to_fts_query(question)
        if not query:
            return []
        rows = self.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (query, limit),
        )
        return [rowid for (rowid,) in rows]

    def search(self, question: str, limit: int = 5, mode: str = "hybrid") -> list[Hit]:
        rankings = []
        if mode in ("hybrid", "vector"):
            rankings.append(self.vector_search(question))
        if mode in ("hybrid", "keyword"):
            rankings.append(self.keyword_search(question))
        scores = rrf(rankings)
        best = sorted(scores, key=scores.get, reverse=True)[:limit]
        return [self._hit(rowid, scores[rowid]) for rowid in best]

    def _hit(self, rowid: int, score: float) -> Hit:
        brand, section, text, url = self.db.execute(
            "SELECT brand, section, text, url FROM chunks WHERE rowid = ?", (rowid,)
        ).fetchone()
        return Hit(rowid, brand, section, text, url, score)


if __name__ == "__main__":
    index = Index()
    for hit in index.search(" ".join(sys.argv[1:])):
        print(f"{hit.score:.4f}  {hit.brand} — {hit.section}\n        {hit.url}")
