import sqlite3

import pytest

from assistant.brands import BrandMatcher
from assistant.text import stem_words

BRANDS = [("soundcore", "Soundcore"), ("honor", "Honor"), ("realme", "Realme"), ("chasy-reward", "часы reward")]
# «наушники» встречается во многих кусках — это обычное слово, а не искажённый бренд.
TEXTS = ["наушники с шумоподавлением"] * 10 + ["зарядка для телефона"]


@pytest.fixture
def matcher():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE chunks (slug TEXT, brand TEXT)")
    db.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(stems)")
    db.executemany("INSERT INTO chunks VALUES (?, ?)", BRANDS)
    db.executemany("INSERT INTO chunks_fts VALUES (?)", [(" ".join(stem_words(t)),) for t in TEXTS])
    return BrandMatcher(lambda sql, params: db.execute(sql, params).fetchall())


def test_exact_latin_and_case(matcher):
    assert matcher.find("Где купить SOUNDCORE?") == ["soundcore"]


def test_several_brands_in_order(matcher):
    assert matcher.find("чем отличается Honor от Realme") == ["honor", "realme"]


def test_misspelled_cyrillic_brand(matcher):
    # Транслитерация даёт «саундкоре», пользователь пишет на слух.
    assert matcher.find("наушники саундкор") == ["soundcore"]


def test_common_words_do_not_match(matcher):
    assert matcher.find("какие наушники выбрать для бега") == []


def test_generic_words_are_ignored_in_brand_names(matcher):
    assert matcher.find("что за бренд reward") == ["chasy-reward"]
