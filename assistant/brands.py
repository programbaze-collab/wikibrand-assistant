"""Распознавание брендов из базы в тексте вопроса.

Название ищется как целое слово (или несколько слов подряд) — латиницей, как в базе,
и в грубой транслитерации кириллицей: «boya» и «боя» → Boya.
"""

import re
from collections.abc import Callable
from difflib import SequenceMatcher

from .chunking import transliterate
from .text import stem_words

_WORD = re.compile(r"[0-9a-zа-яё]+")

# Слова, которые входят в некоторые названия в базе («часы reward»), но не отличают бренд.
GENERIC = {"часы", "chasy", "brand", "бренд", "official"}

# Транслитерация короче этого слишком часто совпадает с обычными русскими словами.
MIN_CYRILLIC_ALIAS = 4

# Нечёткое сравнение: пользователи пишут бренд на слух («саундкор»), а транслитерация
# даёт «соундкоре». Только для длинных слов и при совпадении первой буквы — иначе
# обычные слова начинают совпадать с названиями брендов.
FUZZY_MIN_LENGTH = 6
FUZZY_MIN_RATIO = 0.75
# Слово, которое встречается в базе чаще этого, — обычное слово, а не искажённое название:
# «наушниках» есть в сотнях кусков, «саундкор» — ни в одном. Без этой проверки
# «просто» совпадало с Prostormer, а «руками» — с RunCam.
FUZZY_MAX_DOC_FREQ = 5


def _words(text: str) -> tuple[str, ...]:
    return tuple(w for w in _WORD.findall(text.lower().replace("ё", "е")) if w not in GENERIC)


class BrandMatcher:
    def __init__(self, query: Callable[[str, tuple], list[tuple]]):
        """query — функция запроса к индексу (Index.query)."""
        self.query = query
        self.aliases: dict[tuple[str, ...], set[str]] = {}
        self.names: dict[str, str] = {}
        for slug, brand in query("SELECT DISTINCT slug, brand FROM chunks", ()):
            self.names[slug] = brand
            for name in {brand, slug.replace("-", " ")}:
                self._add(_words(name), slug)
                cyrillic = _words(transliterate(name))
                if len("".join(cyrillic)) >= MIN_CYRILLIC_ALIAS:
                    self._add(cyrillic, slug)
        self.max_words = max(len(alias) for alias in self.aliases)
        self.fuzzy_aliases = [
            (alias[0], slugs)
            for alias, slugs in self.aliases.items()
            if len(alias) == 1 and len(alias[0]) >= FUZZY_MIN_LENGTH and re.match("[а-я]", alias[0])
        ]

    def _is_common_word(self, word: str) -> bool:
        stems = stem_words(word)
        if not stems:
            return True
        [(count,)] = self.query(
            "SELECT count(*) FROM (SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?)",
            (f'"{stems[0]}"', FUZZY_MAX_DOC_FREQ + 1),
        )
        return count > FUZZY_MAX_DOC_FREQ

    def _fuzzy(self, word: str) -> set[str]:
        if len(word) < FUZZY_MIN_LENGTH or not re.match("[а-я]", word) or self._is_common_word(word):
            return set()
        best, slugs = 0.0, set()
        for alias, alias_slugs in self.fuzzy_aliases:
            if alias[0] != word[0]:
                continue
            ratio = SequenceMatcher(None, word, alias).ratio()
            if ratio > best:
                best, slugs = ratio, alias_slugs
        return slugs if best >= FUZZY_MIN_RATIO else set()

    def _add(self, alias: tuple[str, ...], slug: str) -> None:
        if alias:
            self.aliases.setdefault(alias, set()).add(slug)

    def find(self, question: str) -> list[str]:
        """Слаги брендов, названных в вопросе, в порядке появления. Длинные названия важнее коротких."""
        words = _words(question)
        found: list[str] = []
        i = 0
        while i < len(words):
            for size in range(min(self.max_words, len(words) - i), 0, -1):
                slugs = self.aliases.get(words[i : i + size])
                if slugs:
                    found += sorted(slugs - set(found))
                    i += size
                    break
            else:
                found += sorted(self._fuzzy(words[i]) - set(found))
                i += 1
        return found
