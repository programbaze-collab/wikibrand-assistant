"""Построение индекса: python -m assistant.index

Куски и полнотекстовый индекс — в SQLite (chunks.db), векторы — в numpy (embeddings.npy).
Кусков около 14 тысяч: точный перебор всех векторов занимает миллисекунды, поэтому
отдельная векторная база (Qdrant, pgvector) здесь не нужна — она окупается на миллионах.
"""

import sqlite3
import time
from pathlib import Path

import numpy as np

from .chunking import Chunk, load_chunks
from .config import EMBEDDING_MODEL, INDEX_DIR, LOCALE, WIKIBRAND_DIR
from .embeddings import embed_passages
from .text import stem_words


def write_db(chunks: list[Chunk], path: Path) -> None:
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, brand TEXT, slug TEXT,"
            " category TEXT, section TEXT, text TEXT, url TEXT)"
        )
        # rowid куска совпадает с номером строки в embeddings.npy.
        db.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(stems)")
        db.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(i, c.id, c.brand, c.slug, c.category, c.section, c.text, c.url) for i, c in enumerate(chunks)],
        )
        db.executemany(
            "INSERT INTO chunks_fts (rowid, stems) VALUES (?, ?)",
            [(i, " ".join(stem_words(c.text))) for i, c in enumerate(chunks)],
        )


def build(brands_dir: Path = WIKIBRAND_DIR / "brands" / LOCALE, index_dir: Path = INDEX_DIR) -> None:
    started = time.perf_counter()
    chunks = list(load_chunks(brands_dir))
    print(f"кусков: {len(chunks)}, брендов: {len({c.slug for c in chunks})}")

    index_dir.mkdir(parents=True, exist_ok=True)
    write_db(chunks, index_dir / "chunks.db")
    print(f"SQLite и полнотекстовый индекс: {time.perf_counter() - started:.0f} с")

    vectors = embed_passages([c.text for c in chunks])
    np.save(index_dir / "embeddings.npy", vectors)
    (index_dir / "model.txt").write_text(EMBEDDING_MODEL, encoding="utf-8")
    print(f"векторы {vectors.shape}: {time.perf_counter() - started:.0f} с")


if __name__ == "__main__":
    build()
