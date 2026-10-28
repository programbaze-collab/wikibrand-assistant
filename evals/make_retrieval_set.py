"""Синтетический набор для оценки поиска: python -m evals.make_retrieval_set

Берёт случайные куски (поровну из каждого раздела) и просит модель написать вопрос,
на который этот кусок отвечает. Набор сохраняется в evals/retrieval_set.json и дальше
не пересоздаётся: иначе результаты разных прогонов нельзя сравнивать.
"""

import json
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from assistant.config import INDEX_DIR
from assistant.llm import chat_json

PER_SECTION = 9
SEED = 42
OUT = Path(__file__).with_name("retrieval_set.json")

PROMPT = """Ты пишешь вопросы для проверки поиска по энциклопедии брендов.

Ниже фрагмент статьи. Напиши один вопрос, который мог бы задать обычный пользователь и на который этот фрагмент отвечает.

Правила:
- Назови бренд в вопросе, как это сделал бы человек: иногда точно, иногда строчными буквами или кириллицей.
- Не копируй фразы из фрагмента: пиши своими словами, как пишут в поиске или чате.
- Вопрос должен быть про суть фрагмента, а не про случайную деталь.
- Одно предложение, до 15 слов.

Ответь JSON: {"question": "..."}"""


def main() -> None:
    db = sqlite3.connect(INDEX_DIR / "chunks.db")
    sections = [s for (s,) in db.execute("SELECT DISTINCT section FROM chunks ORDER BY section")]
    rng = random.Random(SEED)
    sample = []
    for section in sections:
        rows = db.execute("SELECT id, slug, brand, section, text FROM chunks WHERE section = ?", (section,)).fetchall()
        sample += rng.sample(rows, PER_SECTION)

    def ask(row):
        chunk_id, slug, brand, section, text = row
        question = chat_json(
            [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}], temperature=0.7
        )["question"]
        return {"question": question, "chunk_id": chunk_id, "slug": slug, "brand": brand, "section": section}

    with ThreadPoolExecutor(6) as pool:
        items = list(pool.map(ask, sample))
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} вопросов → {OUT}")


if __name__ == "__main__":
    main()
