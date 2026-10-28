"""Пути и настройки. Всё, что зависит от окружения, — через переменные окружения."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Репозиторий WikiBrand с данными брендов: brands/{locale}/{category}/{slug}/data.json
WIKIBRAND_DIR = Path(os.environ.get("WIKIBRAND_DIR", ROOT.parent / "WikiBrand"))
LOCALE = "ru"
SITE_URL = "https://wiki-brand.com/ru"

INDEX_DIR = Path(os.environ.get("INDEX_DIR", ROOT / "data" / "index"))

# Мультиязычная модель на 384 измерения: понимает русский и помещается на VPS с 1 ГБ памяти.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Кусок длиннее этого режется по абзацам или пунктам списка.
MAX_CHUNK_CHARS = 1200
