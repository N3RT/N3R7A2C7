from fastapi import FastAPI
from fastapi.responses import JSONResponse

from llm_connectors.connector_dev import call_llm, LLMError

app = FastAPI(title="RAG Service", version="0.1.0")


@app.get("/api/v1/health")
async def health():
    return JSONResponse(
        {
            "status": "ok",
            "environment": "dev",
            "chromadb": "unknown",
            "sqlite": "unknown",
            "llm_mode": "ollama_chat_4b",
            "uptime_seconds": 0,
        }
    )


@app.get("/api/v1/llm_test")
async def llm_test():
    try:
        resp = await call_llm(
            [{"role": "user", "content": "Скажи одно слово: тест."}],
            model="llama3.2:1b",
        )
        content = resp["choices"][0]["message"]["content"]
        return {"ok": True, "answer": content}
    except LLMError as e:
        return {"ok": False, "error": str(e)}
