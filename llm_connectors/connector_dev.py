import os
from typing import List, Dict, Any

import httpx


OLLAMA_CHAT_URL = os.getenv(
    "OLLAMA_CHAT_URL",
    "http://127.0.0.1:4004/api/chat",
)


class LLMError(Exception):
    pass


async def call_llm(
    messages: List[Dict[str, Any]],
    model: str | None = None,
) -> Dict[str, Any]:
    """
    Дев-коннектор к локальному сервису ollama_chat_4b.
    messages: список сообщений в формате OpenAI (role/content).
    model: "llama3.2:1b" или "llama3.2:3b" (если None — дефолт в сервисе).
    """
    payload: Dict[str, Any] = {"messages": messages}
    if model:
        payload["model"] = model

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(OLLAMA_CHAT_URL, json=payload)
        if r.status_code != 200:
            raise LLMError(f"ollama_chat_4b error {r.status_code}: {r.text}")
        data = r.json()
        # ожидаем {"reply": "...", "raw": {...}}
        reply = data.get("reply", "")
        return {
            "model": model or "default",
            "choices": [
                {"message": {"role": "assistant", "content": reply}}
            ],
            "raw": data.get("raw"),
        }
