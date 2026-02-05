from typing import Optional, List, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.task_registry import task_registry, TaskRegistryError
from app.core.access_control import check_task_access
from app.core.rag_pipeline import retrieve_text_chunks, build_llm_prompt
from llm_connectors.connector_dev import call_llm, LLMError


router = APIRouter(prefix="/api/v1", tags=["tasks"])


class TaskQueryRequest(BaseModel):
    task_id: str = Field(..., description="Идентификатор задачи, например demo_hello")
    task_type: Literal["demo", "corporate"] = Field(
        "demo",
        description="Тип задачи: demo или corporate (должен совпадать с TaskConfig)",
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


class GenericQueryRequest(BaseModel):
    query: str = Field(..., description="Текст пользовательского запроса")
    debug: bool = Field(False, description="Возвращать служебную информацию по маршрутизации")


class GenericQueryResponse(BaseModel):
    ok: bool
    answer: Optional[str] = None
    routed_task_id: Optional[str] = None
    routed_task_type: Optional[str] = None
    error: Optional[str] = None


async def _run_task(task_id: str, task_type: str, query: str) -> TaskQueryResponse:
    from app.core.task_config import TaskConfig

    try:
        task_cfg: TaskConfig = task_registry.get_task_config(task_id)
    except TaskRegistryError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if task_cfg.task_type != task_type:
        raise HTTPException(
            status_code=400,
            detail=f"Request task_type={task_type} does not match TaskConfig.task_type={task_cfg.task_type}",
        )

    decision = check_task_access(task_cfg)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    if task_cfg.task_type == "demo":
        chunks = retrieve_text_chunks(task_cfg, query, top_k=3)
        system_prompt = build_llm_prompt(task_cfg, query, chunks)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            resp = await call_llm(messages, model="llama3.2:1b")
            answer = resp["choices"][0]["message"]["content"]
            return TaskQueryResponse(ok=True, answer=answer)
        except LLMError as e:
            return TaskQueryResponse(ok=False, error=str(e))

    if task_cfg.task_type == "corporate":
        system_prompt = (
            "Ты корпоративный помощник по данным сотрудников. "
            "Не выдавай реальные персональные данные. "
            "Объясни, какие типы информации доступны в системе и как сотрудник может получить к ним доступ через официальные каналы."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            resp = await call_llm(messages, model="llama3.2:1b")
            answer = resp["choices"][0]["message"]["content"]
            return TaskQueryResponse(ok=True, answer=answer)
        except LLMError as e:
            return TaskQueryResponse(ok=False, error=str(e))

    raise HTTPException(status_code=400, detail="Unsupported task_type")


@router.get("/tasks", response_model=List[TaskInfo])
async def list_tasks() -> List[TaskInfo]:
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
    return await _run_task(
        task_id=request.task_id,
        task_type=request.task_type,
        query=request.query,
    )


@router.post("/query", response_model=GenericQueryResponse)
async def generic_query(request: GenericQueryRequest) -> GenericQueryResponse:
    """
    Общая точка входа:
    - по тексту запроса LLM выбирает подходящий task_id и task_type;
    - затем вызывается общий флоу _run_task.
    """
    user_query = request.query

    system_prompt = (
        "Ты маршрутизатор запросов на задачи RAG-сервиса.\n"
        "У тебя есть задачи:\n"
        "- demo_hello (demo): общие приветственные вопросы про демо-сервис;\n"
        "- demo_rules (demo): вопросы про внутренние правила и регламенты компании;\n"
        "- employee_data (corporate): вопросы про данные сотрудников.\n\n"
        "Тебе приходит текст запроса пользователя.\n"
        "Твоя задача — выбрать ОДНУ задачу и вернуть СТРОГО такой JSON:\n"
        "{\"task_id\": \"demo_rules\", \"task_type\": \"demo\"}\n\n"
        "Жёсткие правила:\n"
        "- НИКАКОГО дополнительного текста, объяснений, Markdown, примеров.\n"
        "- НИКАКИХ других полей, только task_id и task_type.\n"
        "- НИКАКИХ значений null: task_id и task_type ВСЕГДА должны быть строками.\n"
        "- Если сложно решить, используй по умолчанию task_id=\"demo_rules\", task_type=\"demo\".\n"
        "Верни ровно один JSON-объект, без обрамления в массив."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    try:
        resp = await call_llm(messages, model="llama3.2:1b")
        content = resp["choices"][0]["message"]["content"]
    except LLMError as e:
        return GenericQueryResponse(ok=False, error=str(e))

    import json

    try:
        routed = json.loads(content)
        task_id = routed.get("task_id")
        task_type = routed.get("task_type")
        if not isinstance(task_id, str) or not isinstance(task_type, str):
            raise ValueError("task_id/task_type must be strings")
    except Exception:
        # fallback: demo_rules
        task_id = "demo_rules"
        task_type = "demo"

    try:
        task_resp = await _run_task(task_id=task_id, task_type=task_type, query=user_query)
    except HTTPException as e:
        return GenericQueryResponse(
            ok=False,
            error=str(e.detail),
            routed_task_id=task_id,
            routed_task_type=task_type,
        )

    return GenericQueryResponse(
        ok=task_resp.ok,
        answer=task_resp.answer,
        error=task_resp.error,
        routed_task_id=task_id,
        routed_task_type=task_type,
    )
