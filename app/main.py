from fastapi import FastAPI
from fastapi.responses import JSONResponse

from llm_connectors.connector_dev import call_llm, LLMError
from app.api.v1.routes import router as api_v1_router
from app.core.config_loader import get_environment, get_llm_mode


app = FastAPI(title="RAG Service", version="0.1.0")

app.include_router(api_v1_router)


@app.get("/api/v1/health")
async def health():
    env = get_environment()
    llm_mode = get_llm_mode()
    return JSONResponse(
        {
            "status": "ok",
            "environment": env,
            "chromadb": "unknown",
            "sqlite": "unknown",
            "llm_mode": llm_mode or "unknown",
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
