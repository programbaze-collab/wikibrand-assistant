"""Гибридный поиск: python -m assistant.search "как отличить подделку casio"

Векторный поиск находит куски по смыслу («как распознать фейк» ~ «как отличить подделку»),
но плохо — по точным названиям вроде «GA-2100». Поиск по словам — наоборот.
Результаты объединяются через Reciprocal Rank Fusion: важен ранг куска в каждом списке,
а не сами оценки — у косинусной близости и BM25 разные шкалы, складывать их нельзя.
"""

import sqlite3
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .brands import BrandMatcher
from .config import EMBEDDING_MODEL, INDEX_DIR
from .embeddings import embed_query
from .text import to_fts_query

RRF_K = 60  # Стандартная константа RRF: сглаживает разницу между первыми местами.
CANDIDATES = 30  # Сколько кандидатов берёт каждый из двух поисков до объединения.


@dataclass(frozen=True)
class Hit:
    rowid: int
    slug: str
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
        self._db = sqlite3.connect(index_dir / "chunks.db", check_same_thread=False)
        self._lock = threading.Lock()
        self.brands = BrandMatcher(self.query)
        self.slugs = np.array([slug for (slug,) in self.query("SELECT slug FROM chunks ORDER BY rowid")])

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Все запросы к SQLite — через блокировку: одно соединение нельзя использовать
        из нескольких потоков одновременно (FastAPI и evals обращаются параллельно)."""
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def vector_search(self, question: str, limit: int = CANDIDATES) -> list[int]:
        scores = self.vectors @ embed_query(question)
        top = np.argpartition(-scores, limit)[:limit]
        return [int(i) for i in top[np.argsort(-scores[top])]]

    def brand_search(self, question: str, limit: int = CANDIDATES) -> list[int]:
        """Куски брендов, названных в вопросе, по близости к вопросу.

        Это дополнительный список для RRF, а не фильтр: в сравнении «Seiko или Orient»
        Orient может не быть в базе, и тогда нужны и другие куски — например, аналоги Seiko.
        """
        slugs = self.brands.find(question)
        if not slugs:
            return []
        query = embed_query(question)
        per_brand = []
        for slug in slugs:
            rows = np.flatnonzero(self.slugs == slug)
            scores = self.vectors[rows] @ query
            per_brand.append([int(rows[i]) for i in np.argsort(-scores)])
        # По очереди от каждого бренда: в «Honor или Realme» оба бренда должны попасть
        # в контекст, даже если куски одного ближе к вопросу (это показали evals ответов).
        interleaved = [row for group in zip(*per_brand) for row in group] if len(per_brand) > 1 else per_brand[0]
        return interleaved[:limit]

    def keyword_search(self, question: str, limit: int = CANDIDATES) -> list[int]:
        query = to_fts_query(question)
        if not query:
            return []
        rows = self.query(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (query, limit),
        )
        return [rowid for (rowid,) in rows]

    def search(self, question: str, limit: int = 5, mode: str = "hybrid+brand") -> list[Hit]:
        rankings = []
        if mode in ("hybrid+brand", "hybrid", "vector"):
            rankings.append(self.vector_search(question))
        if mode in ("hybrid+brand", "hybrid", "keyword"):
            rankings.append(self.keyword_search(question))
        brand_ranking: list[int] = []
        if mode == "hybrid+brand":
            brand_ranking = self.brand_search(question)
            rankings.append(brand_ranking)
        scores = rrf(rankings)
        best = sorted(scores, key=scores.get, reverse=True)[:limit]
        best = self._ensure_each_brand(best, brand_ranking, limit)
        return [self._hit(rowid, scores[rowid]) for rowid in best]

    def _ensure_each_brand(self, best: list[int], brand_ranking: list[int], limit: int) -> list[int]:
        """У каждого бренда из вопроса — хотя бы один кусок в выдаче.

        В «чем отличается Honor от Realme» куски Honor стояли высоко во всех трёх списках
        и занимали все места, а Realme не попадал в контекст (это показали evals ответов).
        Лучший кусок недостающего бренда заменяет последний кусок бренда, у которого их больше одного.
        """
        brands_in_question = list(dict.fromkeys(self.slugs[row] for row in brand_ranking))
        for slug in brands_in_question:
            if any(self.slugs[row] == slug for row in best):
                continue
            candidate = next(row for row in brand_ranking if self.slugs[row] == slug)
            counts = {s: sum(self.slugs[row] == s for row in best) for s in {self.slugs[row] for row in best}}
            replace = next((i for i in range(len(best) - 1, -1, -1) if counts[self.slugs[best[i]]] > 1), None)
            if replace is None and len(best) < limit:
                best.append(candidate)
            elif replace is not None:
                best[replace] = candidate
        return best

    def _hit(self, rowid: int, score: float) -> Hit:
        [(slug, brand, section, text, url)] = self.query(
            "SELECT slug, brand, section, text, url FROM chunks WHERE rowid = ?", (rowid,)
        )
        return Hit(rowid, slug, brand, section, text, url, score)


if __name__ == "__main__":
    index = Index()
    for hit in index.search(" ".join(sys.argv[1:])):
        print(f"{hit.score:.4f}  {hit.brand} — {hit.section}\n        {hit.url}")
