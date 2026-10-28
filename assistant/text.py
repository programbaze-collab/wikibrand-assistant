"""Подготовка текста для полнотекстового поиска.

FTS5 в SQLite не знает русской морфологии: «подделка» не совпадёт с «подделку».
Поэтому в индекс кладётся текст, где каждое слово приведено к основе (стемминг
Snowball), а запрос обрабатывается так же.
"""

import re

import snowballstemmer

_ru = snowballstemmer.stemmer("russian")
_en = snowballstemmer.stemmer("english")
_WORD = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+")

# Слишком частые слова только шумят в поиске по словам.
STOP_WORDS = {
    "и", "в", "во", "на", "с", "со", "по", "к", "у", "о", "об", "от", "до", "из", "за", "для",
    "не", "ли", "что", "как", "это", "а", "но", "или", "же", "бы", "то", "the", "a", "of", "and",
    "чем", "какой", "какая", "какие", "каких", "есть", "ли", "мне", "можно",
}


def stem_words(text: str) -> list[str]:
    words = []
    for word in _WORD.findall(text.lower().replace("ё", "е")):
        if word in STOP_WORDS:
            continue
        is_cyrillic = bool(re.search("[а-я]", word))
        words.append((_ru if is_cyrillic else _en).stemWord(word))
    return words


def to_fts_query(question: str) -> str:
    """Запрос FTS5: основы слов через OR, каждое в кавычках — так спецсимволы не ломают синтаксис."""
    words = list(dict.fromkeys(stem_words(question)))
    return " OR ".join(f'"{word}"' for word in words)
