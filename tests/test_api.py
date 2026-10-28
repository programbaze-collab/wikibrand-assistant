import httpx
import pytest
from fastapi.testclient import TestClient

from assistant.answer import Answer
from assistant.api import RateLimiter, create_app
from assistant.search import Hit

HITS = [
    Hit(1, "boya", "Boya", "Как отличить подделку", "текст", "https://wiki-brand.com/ru/technology/boya/poddelka", 0.03),
    Hit(2, "boya", "Boya", "Продукция", "текст", "https://wiki-brand.com/ru/technology/boya", 0.02),
]


def fake_answer(question, index):
    # Модель сослалась только на второй фрагмент.
    return Answer(text=f"Ответ на «{question}» [2]", sources=[HITS[1]], hits=HITS)


def client(ask=fake_answer, limiter=None):
    app = create_app(load_index=lambda: object(), ask=ask, limiter=limiter)
    return TestClient(app)


def test_ask_returns_answer_with_numbered_sources():
    with client() as c:
        response = c.post("/ask", json={"question": "  подделка boya  "})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Ответ на «подделка boya» [2]",
        "sources": [{"number": 2, "brand": "Boya", "section": "Продукция", "url": "https://wiki-brand.com/ru/technology/boya"}],
    }


@pytest.mark.parametrize("question", ["", "аб", "я" * 501])
def test_question_length_is_validated(question):
    with client() as c:
        assert c.post("/ask", json={"question": question}).status_code == 422


def test_model_failure_gives_502_with_message():
    def broken(question, index):
        raise httpx.ConnectError("нет связи")

    with client(ask=broken) as c:
        response = c.post("/ask", json={"question": "подделка boya"})
    assert response.status_code == 502
    assert "Попробуйте позже" in response.json()["detail"]


def test_rate_limit_per_client():
    with client(limiter=RateLimiter(limit=2, window=60)) as c:
        codes = [c.post("/ask", json={"question": "подделка boya"}).status_code for _ in range(3)]
        other = c.post("/ask", json={"question": "подделка boya"}, headers={"X-Forwarded-For": "10.0.0.2"})
    assert codes == [200, 200, 429]
    assert other.status_code == 200


def test_spoofed_forwarded_for_does_not_bypass_limit():
    # Клиент подставляет свой адрес в начало, прокси дописывает настоящий в конец.
    with client(limiter=RateLimiter(limit=1, window=60)) as c:
        first = c.post("/ask", json={"question": "подделка boya"}, headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.9"})
        spoofed = c.post("/ask", json={"question": "подделка boya"}, headers={"X-Forwarded-For": "2.2.2.2, 10.0.0.9"})
    assert first.status_code == 200
    assert spoofed.status_code == 429


def test_rate_limiter_window_slides():
    now = [0.0]
    limiter = RateLimiter(limit=1, window=60, clock=lambda: now[0])
    assert limiter.allow("a") and not limiter.allow("a")
    now[0] = 61
    assert limiter.allow("a")


def test_health_and_chat_page():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}
        page = c.get("/")
    assert page.status_code == 200
    assert "Ассистент WikiBrand" in page.text
