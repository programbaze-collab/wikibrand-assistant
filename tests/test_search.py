from assistant.search import rrf
from assistant.text import stem_words, to_fts_query


def test_rrf_rewards_items_found_by_both_searches():
    scores = rrf([[1, 2, 3], [3, 4]], k=60)
    # 3 есть в обоих списках — выше, чем 1, который первый только в одном.
    assert scores[3] > scores[1] > scores[2]
    assert scores[4] == 1 / 62


def test_stemming_matches_word_forms():
    assert stem_words("подделка") == stem_words("подделку") == stem_words("подделки")


def test_stop_words_are_dropped():
    assert stem_words("как и что") == []
    # «где» не стоп-слово: совпадает с «где купить» в заголовке раздела.
    assert stem_words("где") == ["где"]


def test_fts_query_quotes_words_so_symbols_do_not_break_syntax():
    query = to_fts_query('Где купить "GA-2100" OR *?')
    assert query == '"где" OR "куп" OR "ga" OR "2100" OR "or"'


def test_empty_question_gives_empty_query():
    assert to_fts_query("и в на") == ""
