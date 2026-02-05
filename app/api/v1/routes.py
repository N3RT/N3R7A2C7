from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.task_registry import task_registry, TaskRegistryError
from app.core.access_control import check_task_access
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


class TaskInfo(BaseModel):
    task_id: str
    task_type: str
    name: str
    description: str


@router.get("/tasks", response_model=List[TaskInfo])
async def list_tasks() -> List[TaskInfo]:
    """
    Список всех задач из config/tasks/*.yaml.
    """
    tasks = task_registry.list_registered_tasks()
    return [
        TaskInfo(
            task_id=t.task_id,
            task_type=t.task_type,
            name=t.name,
            description=t.description,
        )
    for t in tasks
    ]


@router.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str) -> TaskInfo:
    """
    Информация по одной задаче.
    """
    try:
        cfg = task_registry.get_task_config(task_id)
    except TaskRegistryError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return TaskInfo(
        task_id=cfg.task_id,
        task_type=cfg.task_type,
        name=cfg.name,
        description=cfg.description,
    )


@router.post("/task/query", response_model=TaskQueryResponse)
async def task_query(request: TaskQueryRequest) -> TaskQueryResponse:
    """
    Эндпоинт по ТЗ:
    - получает TaskConfig через TaskRegistry;
    - проверяет согласованность task_type;
    - делегирует решение по доступу модулю AccessControl;
    - для demo-задач делает минимальный RAG по текстам и вызывает LLM;
    - corporate-задачи в dev/test блокируются.
    """
    try:
        task_cfg = task_registry.get_task_config(request.task_id)
    except TaskRegistryError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if task_cfg.task_type != request.task_type:
        raise HTTPException(
            status_code=400,
            detail=f"Request task_type={request.task_type} does not match TaskConfig.task_type={task_cfg.task_type}",
        )

    decision = check_task_access(task_cfg)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

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
