"""Оценка поиска: python -m evals.retrieval

Для каждого вопроса из retrieval_set.json известен кусок, из которого он сделан.
Попадание засчитывается, если в выдаче есть кусок того же бренда и того же раздела:
раздел бывает разрезан на несколько кусков, и любой из них отвечает на вопрос.

Метрики:
- hit@1, hit@5 — доля вопросов, где нужный раздел на первом месте / в первых пяти;
- MRR — среднее 1/место нужного раздела (0, если его нет в первых 10): чем выше, тем ближе к началу;
- бренд@5 — доля вопросов, где в первых пяти есть хоть что-то про нужный бренд.
"""

import json
from pathlib import Path

from assistant.search import Index

SET = Path(__file__).with_name("retrieval_set.json")
MODES = ["keyword", "vector", "hybrid"]


def evaluate(index: Index, items: list[dict], mode: str) -> dict:
    hit1 = hit5 = brand5 = mrr = 0.0
    misses = []
    for item in items:
        hits = index.search(item["question"], limit=10, mode=mode)
        keys = [index.db.execute("SELECT slug, section FROM chunks WHERE rowid = ?", (h.rowid,)).fetchone() for h in hits]
        target = (item["slug"], item["section"])
        rank = next((i for i, key in enumerate(keys, start=1) if key == target), None)
        hit1 += rank == 1
        hit5 += rank is not None and rank <= 5
        mrr += 1 / rank if rank else 0
        brand5 += any(slug == item["slug"] for slug, _ in keys[:5])
        if not rank or rank > 5:
            misses.append((item, keys[:3]))
    n = len(items)
    return {"hit@1": hit1 / n, "hit@5": hit5 / n, "MRR": mrr / n, "бренд@5": brand5 / n, "misses": misses}


def main() -> None:
    items = json.loads(SET.read_text(encoding="utf-8"))
    index = Index()
    results = {mode: evaluate(index, items, mode) for mode in MODES}

    print(f"Вопросов: {len(items)}\n")
    print(f"{'режим':<10}" + "".join(f"{m:>10}" for m in ["hit@1", "hit@5", "MRR", "бренд@5"]))
    for mode, r in results.items():
        print(f"{mode:<10}" + "".join(f"{r[m]:>10.2f}" for m in ["hit@1", "hit@5", "MRR", "бренд@5"]))

    print("\nПромахи гибридного поиска (нужный раздел не в первых пяти):")
    for item, top in results["hybrid"]["misses"]:
        found = ", ".join(f"{slug}/{section}" for slug, section in top)
        print(f"- [{item['slug']}/{item['section']}] {item['question']}\n    первые три: {found}")


if __name__ == "__main__":
    main()
