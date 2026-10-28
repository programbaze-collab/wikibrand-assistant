FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models \
    INDEX_DIR=/data/index

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Модель эмбеддингов скачивается при сборке, а не при каждом запуске контейнера.
COPY assistant/config.py assistant/__init__.py assistant/
RUN python -c "from fastembed import TextEmbedding; from assistant.config import EMBEDDING_MODEL; TextEmbedding(EMBEDDING_MODEL)"

COPY assistant assistant

# Индекс (data/index) не входит в образ: он собирается из данных WikiBrand
# (python -m assistant.index) и подключается томом в /data/index.
RUN useradd --create-home app && chown -R app /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "assistant.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
