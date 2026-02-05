from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config_loader import get_environment
from app.core.task_config import load_task_config, TaskConfigError
from app.core.rag_pipeline import retrieve_text_chunks, build_llm_prompt
from llm_connectors.connector_dev import call_llm, LLMError


router = APIRouter(prefix="/api/v1", tags=["tasks"])


class TaskQueryRequest(BaseModel):
    task_id: str = Field(..., description="Идентификатор задачи, например demo_hello")
    task_type: str = Field(
        "demo",
        description="Тип задачи: demo или corporate (для валидации, должен совпадать с TaskConfig)",
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
    Эндпоинт по ТЗ:
    - грузит TaskConfig по task_id;
    - проверяет согласованность task_type и environment;
    - для demo-задач выполняет минимальный RAG по текстам (через ChromaDB) и отправляет контекст в LLM;
    - для corporate в dev/test блокирует.
    """
    env = get_environment()

    try:
        task_cfg = load_task_config(request.task_id)
    except TaskConfigError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if task_cfg.task_type != request.task_type:
        raise HTTPException(
            status_code=400,
            detail=f"Request task_type={request.task_type} does not match TaskConfig.task_type={task_cfg.task_type}",
        )

    if task_cfg.task_type == "corporate" and env != "prod":
        raise HTTPException(
            status_code=403,
            detail="Корпоративные задачи разрешены только в режиме prod",
        )

    if task_cfg.task_type == "demo":
        chunks = retrieve_text_chunks(task_cfg, request.query, top_k=3)
        system_prompt = build_llm_prompt(task_cfg, request.query, chunks)

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
