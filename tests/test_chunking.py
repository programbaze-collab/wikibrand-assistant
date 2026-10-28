from assistant.chunking import MAX_CHUNK_CHARS, _pack, brand_chunks, transliterate

DATA = {
    "brand": {"name": "Boya"},
    "history": "Первый абзац истории.\n\nВторой абзац истории.",
    "products": [{"name": "BY-M1", "summary": "Петличный микрофон", "features": ["3,5 мм", "6 м кабель"]}],
    "stores": [{"type": "Маркетплейсы", "description": "Ozon, Wildberries"}],
    "where_to_buy_notes": "Проверяйте продавца.",
    "authenticity": {"summary": "Подделки встречаются.", "tips": ["Проверьте упаковку"]},
    "faq": [{"q": "Где производят?", "a": "В Китае."}],
    "alternatives": [],
}
PAGES = {"history": "istoriya", "where-to-buy": "gde-kupit", "authenticity": "poddelka"}


def chunks_by_section():
    return {c.section: c for c in brand_chunks(DATA, PAGES, "electronics", "boya")}


def test_section_becomes_chunk_with_brand_header_and_topics():
    chunk = chunks_by_section()["Как отличить подделку"]
    assert chunk.text.splitlines()[0].startswith("Boya (боя) — Как отличить подделку: подделки")
    assert "Проверьте упаковку" in chunk.text
    assert chunk.id == "boya:authenticity:0"


def test_url_points_to_section_page_or_brand_root():
    sections = chunks_by_section()
    assert sections["Как отличить подделку"].url == "https://wiki-brand.com/ru/electronics/boya/poddelka"
    # Для продукции страницы нет в pages.json — ссылка на главную бренда.
    assert sections["Продукция"].url == "https://wiki-brand.com/ru/electronics/boya"


def test_where_to_buy_notes_join_stores_section():
    assert "Проверяйте продавца." in chunks_by_section()["Где купить"].text


def test_empty_sections_are_skipped():
    assert "Аналоги" not in chunks_by_section()


def test_pack_never_splits_an_item_and_respects_limit():
    items = ["а" * 500, "б" * 500, "в" * 500]
    packed = _pack(items, MAX_CHUNK_CHARS)
    assert packed == ["а" * 500 + "\n" + "б" * 500, "в" * 500]
    # Пункт длиннее лимита остаётся целым, а не режется посередине.
    assert _pack(["г" * 2000], MAX_CHUNK_CHARS) == ["г" * 2000]


def test_transliterate():
    assert transliterate("Boya") == "боя"
    assert transliterate("Hongdian") == "хонгдиан"
    assert transliterate("Shanling") == "шанлинг"
    assert transliterate("Тест") == "тест"
