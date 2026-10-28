"""Ответ на вопрос по найденным кускам: python -m assistant.answer "как отличить подделку boya"

Модель получает пронумерованные фрагменты и отвечает только по ним, ставя номера источников.
Ссылки собирает код по номерам, а не модель: так модель не может выдумать адрес страницы.
"""

import re
import sys
from dataclasses import dataclass

from .llm import chat
from .search import Hit, Index

SYSTEM_PROMPT = """Ты — ассистент энциклопедии брендов WikiBrand. Отвечаешь на вопросы о брендах по-русски, только по фрагментам статей энциклопедии, которые тебе даны.

Правила:
- Используй только факты из фрагментов. Не добавляй ничего из своих знаний, даже если уверен.
- После каждого утверждения ставь номер фрагмента, на который опираешься: [1], [2]. Номер — только существующего фрагмента.
- Перед фрагментами указано, о каких брендах из вопроса в энциклопедии точно есть статьи. О них никогда не говори, что статьи нет.
- Если вопрос о бренде, которого нет в этом списке и нет среди фрагментов (фрагменты о других брендах), ответь: «В энциклопедии WikiBrand нет статьи о <бренд>» — и больше ничего не добавляй о нём. Можно сказать, о каких похожих брендах статьи есть, если они есть во фрагментах.
- Если фрагменты о нужном бренде, но ответа на вопрос в них нет, так и скажи: «В статье о <бренд> об этом не сказано».
- Если в энциклопедии сказано, что информации мало или она противоречива, передай это, а не додумывай.
- Отвечай коротко: 2–5 предложений или короткий список. Без вступлений и без Markdown-заголовков."""


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Hit]  # Только те фрагменты, на которые модель сослалась.
    hits: list[Hit]  # Все найденные фрагменты — для отладки и evals.


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {hit.text}" for i, hit in enumerate(hits, start=1))


def cited_numbers(text: str, total: int) -> list[int]:
    """Номера фрагментов, на которые сослалась модель, без выдуманных номеров и повторов."""
    numbers = [int(n) for n in re.findall(r"\[(\d+)\]", text)]
    return list(dict.fromkeys(n for n in numbers if 1 <= n <= total))


def answer(question: str, index: Index, limit: int = 5, mode: str = "hybrid+brand") -> Answer:
    hits = index.search(question, limit=limit, mode=mode)
    # Без этой строки модель говорила «нет статьи о Realme», когда поиск не принёс
    # кусков Realme, хотя статья есть (это показали evals ответов).
    known = [index.brands.names[slug] for slug in index.brands.find(question)]
    known_line = f"В энциклопедии есть статьи о брендах из вопроса: {', '.join(known)}." if known else (
        "Брендов из вопроса в списке статей энциклопедии не найдено."
    )
    text = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{known_line}\n\nФрагменты:\n\n{format_context(hits)}\n\nВопрос: {question}"},
        ],
        temperature=0.1,
    )
    return Answer(text=text, sources=[hits[n - 1] for n in cited_numbers(text, len(hits))], hits=hits)


if __name__ == "__main__":
    result = answer(" ".join(sys.argv[1:]), Index())
    print(result.text, "\n")
    for i, hit in enumerate(result.hits, start=1):
        mark = "→" if hit in result.sources else " "
        print(f"{mark} [{i}] {hit.brand} — {hit.section}: {hit.url}")
