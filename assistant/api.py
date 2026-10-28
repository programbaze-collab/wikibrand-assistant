"""HTTP API и страница чата: uvicorn assistant.api:app

POST /ask      {"question": "..."} → ответ и источники
GET  /health   для проверки контейнера
GET  /         страница чата
"""

import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answer import Answer, answer
from .search import Index

log = logging.getLogger(__name__)
STATIC = Path(__file__).with_name("static")

# Каждый вопрос — платный запрос к DeepSeek, а сервис публичный.
RATE_LIMIT = 10  # вопросов
RATE_WINDOW = 60  # за столько секунд с одного адреса


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    number: int  # номер, которым ответ ссылается на источник: [1], [2]
    brand: str
    section: str
    url: str


class Reply(BaseModel):
    answer: str
    sources: list[Source]


class RateLimiter:
    """Скользящее окно в памяти процесса: хватает для одного экземпляра сервиса."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.calls: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = self.clock()
        calls = self.calls[key]
        while calls and now - calls[0] > self.window:
            calls.popleft()
        if len(calls) >= self.limit:
            return False
        calls.append(now)
        return True


def to_reply(result: Answer) -> Reply:
    sources = [
        Source(number=result.hits.index(hit) + 1, brand=hit.brand, section=hit.section, url=hit.url)
        for hit in result.sources
    ]
    return Reply(answer=result.text, sources=sources)


def create_app(
    load_index: Callable[[], Index] = Index,
    ask: Callable[[str, Index], Answer] = answer,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    """Фабрика приложения: в тестах подставляются свой индекс и своя функция ответа."""
    state: dict = {}
    limiter = limiter or RateLimiter(RATE_LIMIT, RATE_WINDOW)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Индекс и модель эмбеддингов загружаются один раз при старте, а не на каждый запрос.
        state["index"] = load_index()
        yield

    app = FastAPI(title="Ассистент WikiBrand", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # Обычная (не async) функция: FastAPI выполнит её в пуле потоков,
    # и долгий запрос к DeepSeek не заблокирует остальные запросы.
    @app.post("/ask")
    def ask_question(body: Question, request: Request) -> Reply:
        # Последний адрес в X-Forwarded-For дописал наш прокси (Caddy) — это настоящий клиент.
        # Первые значения клиент может подставить сам, чтобы обойти ограничение.
        client = request.headers.get("x-forwarded-for", request.client.host if request.client else "").split(",")[-1]
        if not limiter.allow(client.strip()):
            raise HTTPException(429, "Слишком много вопросов. Подождите минуту.")
        try:
            return to_reply(ask(body.question.strip(), state["index"]))
        except httpx.HTTPError:
            log.exception("DeepSeek недоступен")
            raise HTTPException(502, "Не удалось получить ответ от модели. Попробуйте позже.")

    @app.get("/", include_in_schema=False)
    def chat_page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
