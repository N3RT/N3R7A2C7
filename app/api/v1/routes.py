from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config_loader import get_environment
from llm_connectors.connector_dev import call_llm, LLMError


router = APIRouter(prefix="/api/v1", tags=["tasks"])


class TaskQueryRequest(BaseModel):
    task_id: str = Field(..., description="Идентификатор задачи, например demo_hello")
    task_type: str = Field(
        "demo",
        description="Тип задачи: demo или corporate",
        pattern="^(demo|corporate)$",
    )
    query: str = Field(..., description="Текст пользовательского запроса")


class TaskQueryResponse(BaseModel):
    ok: bool
    answer: Optional[str] = None
    error: Optional[str] = None


@router.post("/task/query", response_model=TaskQueryResponse)
async def task_query(request: TaskQueryRequest) -> TaskQueryResponse:
    """
    Минимальный демо-эндпоинт по ТЗ:
    - проверяет режим/тип задачи;
    - для demo-задач в dev/test прокидывает запрос в LLM с простым тех.промптом;
    - для corporate в dev/test блокирует.
    """
    env = get_environment()

    if request.task_type == "corporate" and env != "prod":
        raise HTTPException(
            status_code=403,
            detail="Корпоративные задачи разрешены только в режиме prod",
        )

    if request.task_type == "demo":
        system_prompt = (
            "Ты помощник демо-RAG системы. Отвечай кратко и по делу, "
            "без выдуманных фактов. Если не хватает информации, честно скажи об этом."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.query},
        ]
        try:
            resp = await call_llm(messages, model="llama3.2:1b")
            answer = resp["choices"][0]["message"]["content"]
            return TaskQueryResponse(ok=True, answer=answer)
        except LLMError as e:
            return TaskQueryResponse(ok=False, error=str(e))

    raise HTTPException(status_code=400, detail="Unsupported task_type")
