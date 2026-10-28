"""Оценка ответов: python -m evals.answers [режим поиска ...]

Набор answer_set.json написан вручную. Типы вопросов:
- answer — ответ есть в базе: среди источников ответа должен быть нужный бренд;
- not_in_base — бренда нет в энциклопедии: ассистент должен сказать, что статьи нет;
- not_in_article — бренд есть, но ответа в статье нет: ассистент должен так и сказать.

Для всех вопросов судья (DeepSeek) проверяет достоверность: каждое утверждение ответа
должно подтверждаться фрагментами, на которые он ссылается. Результат — в evals/answers_report.md.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from assistant.answer import answer, format_context
from assistant.llm import chat_json
from assistant.search import Index

SET = Path(__file__).with_name("answer_set.json")
REPORT = Path(__file__).with_name("answers_report.md")

NOT_IN_BASE = re.compile(r"нет (отдельной )?стать|нет информации о|не нашл|отсутству", re.I)
NOT_IN_ARTICLE = re.compile(r"не сказано|не указан|нет (данных|информации|сведений)|не упомина|не приводит", re.I)

JUDGE_PROMPT = """Ты проверяешь достоверность ответа ассистента энциклопедии брендов.

Даны фрагменты статей с номерами и ответ ассистента со ссылками [n] на фрагменты.
Найди утверждения ответа, которые НЕ подтверждаются фрагментами: факт, которого во фрагментах нет,
искажённый факт, или ссылка [n] на фрагмент, где этого нет. Пересказ своими словами — не ошибка.
Фразы «в энциклопедии нет статьи о …» и «в статье об этом не сказано» — не утверждения о фактах, их не проверяй.

Ответь JSON: {"unsupported": ["утверждение — почему не подтверждается", ...]} — пустой список, если всё подтверждается."""


def judge(hits, text: str) -> list[str]:
    result = chat_json(
        [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Фрагменты:\n\n{format_context(hits)}\n\nОтвет ассистента:\n{text}"},
        ],
        temperature=0,
    )
    return result.get("unsupported", [])


def check(item: dict, index: Index, mode: str) -> dict:
    result = answer(item["question"], index, mode=mode)
    cited_slugs = {hit.slug for hit in result.sources}
    checks = {}
    if item["type"] == "answer":
        checks["источник нужного бренда"] = all(slug in cited_slugs for slug in item["brands"])
    elif item["type"] == "not_in_base":
        checks["говорит, что статьи нет"] = bool(NOT_IN_BASE.search(result.text))
    else:
        checks["говорит, что в статье не сказано"] = bool(NOT_IN_ARTICLE.search(result.text))
    unsupported = judge(result.hits, result.text)
    checks["нет неподтверждённых утверждений"] = not unsupported
    return {**item, "text": result.text, "cited": sorted(cited_slugs), "checks": checks, "unsupported": unsupported}


def run(index: Index, items: list[dict], mode: str) -> list[dict]:
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(lambda item: check(item, index, mode), items))


def main() -> None:
    modes = sys.argv[1:] or ["hybrid", "hybrid+brand"]
    items = json.loads(SET.read_text(encoding="utf-8"))
    index = Index()
    lines = ["# Оценка ответов", ""]
    for mode in modes:
        results = run(index, items, mode)
        passed = sum(all(r["checks"].values()) for r in results)
        names = sorted({name for r in results for name in r["checks"]})
        summary = f"**{mode}**: вопросов пройдено {passed} из {len(results)}. " + "; ".join(
            f"{name} — {sum(r['checks'].get(name, False) for r in results)} из {sum(name in r['checks'] for r in results)}"
            for name in names
        )
        print(summary.replace("**", ""))
        lines += [f"## Режим поиска {mode}", "", summary, ""]
        for r in results:
            if all(r["checks"].values()):
                continue
            failed = ", ".join(name for name, ok in r["checks"].items() if not ok)
            lines += [f"### ✗ {r['question']}", "", f"Провалено: {failed}. Источники: {', '.join(r['cited']) or 'нет'}", "", r["text"], ""]
            lines += [f"- {u}" for u in r["unsupported"]] + [""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
