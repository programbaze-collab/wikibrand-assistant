"""Вызов DeepSeek. Ключ — в переменной окружения DEEPSEEK_API_KEY."""

import json
import os

import httpx

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


def chat(messages: list[dict], *, temperature: float = 0.2, json_mode: bool = False, timeout: float = 120) -> str:
    response = httpx.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def chat_json(messages: list[dict], **kwargs) -> dict:
    return json.loads(chat(messages, json_mode=True, **kwargs))
