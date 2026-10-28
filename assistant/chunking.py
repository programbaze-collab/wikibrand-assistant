"""Нарезка статей WikiBrand на куски для поиска.

Статья бренда уже разделена по смыслу (история, продукты, подделки, аналоги…),
поэтому кусок — это раздел, а длинный раздел режется по абзацам или пунктам списка.
В начало каждого куска добавляется заголовок «Бренд — раздел»: без него кусок
«Проверьте гравировку на крышке» непонятно к какому бренду относится, и поиск его не найдёт.
"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import MAX_CHUNK_CHARS, SITE_URL


@dataclass(frozen=True)
class Chunk:
    id: str
    brand: str
    slug: str
    category: str
    section: str
    text: str
    url: str


# Раздел data.json → (название для человека, id страницы в pages.json).
SECTIONS = {
    "history": ("История", "history"),
    "products": ("Продукция", "products"),
    "technologies": ("Технологии", "technology"),
    "stores": ("Где купить", "where-to-buy"),
    "alternatives": ("Аналоги", "alternatives"),
    "authenticity": ("Как отличить подделку", "authenticity"),
    "reviews": ("Отзывы", "reviews"),
    "faq": ("Вопросы и ответы", "index"),
    "interesting_fact": ("Интересный факт", "index"),
}

# На какие вопросы отвечает раздел — словами пользователя. Без этого «какие бренды похожи
# на Benro» находит «Отзывы» и «Историю» Benro, где конкуренты тоже упоминаются,
# а не раздел «Аналоги» (это показали evals поиска).
SECTION_TOPICS = {
    "history": "история бренда, кто основал, откуда бренд, где производят, чем известен",
    "products": "какие товары выпускает, модели, линейки, ассортимент",
    "technologies": "технологии, материалы, характеристики, как устроено",
    "stores": "где купить, магазины, маркетплейсы, официальные продавцы",
    "alternatives": "аналоги, похожие бренды, чем заменить, альтернативы, конкуренты",
    "authenticity": "подделки, как отличить оригинал, как не купить фейк",
    "reviews": "отзывы покупателей, плюсы и минусы, за что хвалят и ругают",
    "faq": "частые вопросы",
    "interesting_fact": "интересный факт",
}

_TRANSLIT = [
    ("shch", "щ"), ("sch", "ш"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"), ("ts", "ц"),
    ("ya", "я"), ("yu", "ю"), ("yo", "ё"), ("ee", "и"), ("oo", "у"), ("ph", "ф"), ("th", "т"),
    ("ei", "ай"), ("ai", "ай"), ("ou", "ау"), ("ay", "ей"), ("ey", "ей"),
    ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"), ("h", "х"),
    ("i", "и"), ("j", "дж"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"),
    ("q", "к"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"), ("v", "в"), ("w", "в"), ("x", "кс"),
    ("y", "и"), ("z", "з"),
]


def transliterate(name: str) -> str:
    """Грубая транслитерация латиницы в кириллицу: «Boya» → «боя», «Hongdian» → «хонгдиан».

    Пользователи часто пишут бренд кириллицей, а в статье он латиницей — поиск по словам
    их не сопоставит. Точность не нужна: важно, чтобы совпала основа слова.
    """
    result, text = "", name.lower()
    while text:
        for latin, cyrillic in _TRANSLIT:
            if text.startswith(latin):
                result, text = result + cyrillic, text[len(latin):]
                break
        else:
            result, text = result + text[0], text[1:]
    return result


def _section_items(data: dict, key: str) -> list[str]:
    """Текст раздела как список смысловых единиц: абзацев или пунктов."""
    value = data.get(key)
    if key == "history":
        return [p.strip() for p in re.split(r"\n\s*\n", value or "") if p.strip()]
    if key == "products":
        return [
            f"{p.get('name', '')}: {p.get('summary', '')} {'; '.join(p.get('features', []))}".strip()
            for p in value or []
        ]
    if key == "technologies":
        return [f"{t.get('name', '')}: {t.get('description', '')}" for t in value or []]
    if key == "stores":
        items = [f"{s.get('type', '')}: {s.get('description', '')}" for s in value or []]
        return items + ([data["where_to_buy_notes"]] if data.get("where_to_buy_notes") else [])
    if key == "alternatives":
        return [f"{a.get('name', '')}: {a.get('comparison', '')}" for a in value or []]
    if key == "authenticity":
        value = value or {}
        return ([value["summary"]] if value.get("summary") else []) + list(value.get("tips", []))
    if key == "reviews":
        # Отзывы в базе — обобщённый опыт покупателей, а не отзывы конкретных людей.
        items = [f"{r.get('role', '')}: {r.get('text', '')}" for r in value or []]
        return ([data["reviews_summary"]] if data.get("reviews_summary") else []) + items
    if key == "faq":
        return [f"Вопрос: {q.get('q', '')} Ответ: {q.get('a', '')}" for q in value or []]
    if key == "interesting_fact":
        value = value or {}
        return [f"{value.get('title', '')}. {value.get('text', '')}"] if value.get("text") else []
    return []


def _pack(items: list[str], limit: int) -> list[str]:
    """Собирает пункты в куски не длиннее limit, не разрывая пункт посередине."""
    chunks, current = [], ""
    for item in items:
        if current and len(current) + 1 + len(item) > limit:
            chunks.append(current)
            current = item
        else:
            current = f"{current}\n{item}" if current else item
    if current:
        chunks.append(current)
    return chunks


def brand_chunks(data: dict, page_slugs: dict[str, str], category: str, slug: str) -> list[Chunk]:
    brand = data.get("brand", {}).get("name") or slug
    base_url = f"{SITE_URL}/{category}/{slug}"
    cyrillic = transliterate(brand)
    brand_label = f"{brand} ({cyrillic})" if cyrillic != brand.lower() else brand
    chunks = []
    for key, (title, page_id) in SECTIONS.items():
        items = [item for item in _section_items(data, key) if item.strip(" :.")]
        page_slug = page_slugs.get(page_id, "")
        url = f"{base_url}/{page_slug}" if page_slug else base_url
        header = f"{brand_label} — {title}: {SECTION_TOPICS[key]}"
        for i, body in enumerate(_pack(items, MAX_CHUNK_CHARS)):
            chunks.append(
                Chunk(
                    id=f"{slug}:{key}:{i}",
                    brand=brand,
                    slug=slug,
                    category=category,
                    section=title,
                    text=f"{header}\n{body}",
                    url=url,
                )
            )
    return chunks


def load_chunks(brands_dir: Path) -> Iterator[Chunk]:
    """Все куски всех брендов одной локали: brands_dir = brands/{locale}."""
    for data_file in sorted(brands_dir.glob("*/*/data.json")):
        category, slug = data_file.parent.parent.name, data_file.parent.name
        try:
            data = json.loads(data_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Битый файл не должен останавливать индексацию остальных 1 254 брендов.
            print(f"пропущен {data_file}: некорректный JSON")
            continue
        pages_file = data_file.with_name("pages.json")
        page_slugs = {}
        if pages_file.exists():
            pages = json.loads(pages_file.read_text(encoding="utf-8")).get("pages", [])
            page_slugs = {p["id"]: p.get("slug", "") for p in pages if "id" in p}
        yield from brand_chunks(data, page_slugs, category, slug)
