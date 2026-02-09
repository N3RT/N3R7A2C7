from typing import Optional, List, Literal, Any, Dict
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.task_registry import task_registry, TaskRegistryError
from app.core.access_control import check_task_access
from app.core.rag_pipeline import (
    retrieve_text_chunks,
    retrieve_text_chunks_for_research,
    build_llm_prompt,
)
from app.core.event_logger import log_event
from app.core.glossary import normalize_query
from app.core.profanity_filter import check_profanity
from app.core.config_loader import get_environment
from llm_connectors.connector_dev import call_llm, LLMError
from app.core.task_classifier import classify_query
from app.postprocessing.files import save_markdown, save_docx


router = APIRouter(prefix="/api/v1", tags=["tasks"])


class TaskQueryRequest(BaseModel):
    task_id: str = Field(..., description="Идентификатор задачи, например demo_hello")
    task_type: Literal["demo", "corporate"] = Field(
        "demo",
        description="Тип задачи: demo или corporate (должен совпадать с TaskConfig)",
    )
    query: str = Field(..., description="Текст пользовательского запроса")
    debug: bool = Field(
        True,
        description="Если true, в ответ добавляется отладочная meta-информация",
    )


class TaskQueryResponse(BaseModel):
    ok: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class TaskInfo(BaseModel):
    task_id: str
    task_type: str
    name: str
    description: str


class GenericQueryRequest(BaseModel):
    query: str = Field(..., description="Текст пользовательского запроса")
    debug: bool = Field(False, description="Возвращать служебную информацию")


class GenericQueryResponse(BaseModel):
    ok: bool
    answer: Optional[str] = None
    routed_task_id: Optional[str] = None
    routed_task_type: Optional[str] = None
    error: Optional[str] = None
    classification_confidence: Optional[float] = None


async def _run_task(task_id: str, task_type: str, query: str, debug: bool = True) -> TaskQueryResponse:
    from app.core.task_config import TaskConfig

    request_id = str(uuid.uuid4())
    env = get_environment()

    prof = check_profanity(query)
    profanity_flag = prof.detected

    normalized_query = normalize_query(query)

    log_event(
        event_type="query",
        request_id=request_id,
        task_id=task_id,
        task_type=task_type,
        payload={
            "query": query,
            "normalized_query": normalized_query,
            "profanity_detected": profanity_flag,
            "profanity_matches": prof.matches,
        },
    )

    try:
        task_cfg: TaskConfig = task_registry.get_task_config(task_id)
    except TaskRegistryError as e:
        log_event(
            event_type="query_error",
            request_id=request_id,
            task_id=task_id,
            task_type=task_type,
            payload={"error": str(e)},
        )
        raise HTTPException(status_code=404, detail=str(e))

    if task_cfg.task_type != task_type:
        detail = (
            f"Request task_type={task_type} does not match "
            f"TaskConfig.task_type={task_cfg.task_type}"
        )
        log_event(
            event_type="query_error",
            request_id=request_id,
            task_id=task_id,
            task_type=task_type,
            payload={"error": detail},
        )
        raise HTTPException(status_code=400, detail=detail)

    decision = check_task_access(task_cfg)
    if not decision.allowed:
        log_event(
            event_type="access_denied",
            request_id=request_id,
            task_id=task_id,
            task_type=task_type,
            payload={"reason": decision.reason},
        )
        raise HTTPException(status_code=403, detail=decision.reason)

    meta: Dict[str, Any] = {}

    # ========== DEMO-задачи: общий RAG-путь ==========
    if task_cfg.task_type == "demo":
        t0 = time.time()
        chunks = retrieve_text_chunks(task_cfg, normalized_query, top_k=3)
        t1 = time.time()

        log_event(
            event_type="search",
            request_id=request_id,
            task_id=task_id,
            task_type=task_type,
            payload={
                "duration_ms": int((t1 - t0) * 1000),
                "chunks_found": len(chunks),
            },
        )

        # TEST MODE: только retrieval без LLM
        if env == "test":
            meta["retrieved_chunks"] = [
                {
                    "text": ch.text,
                    "source_id": ch.source_id,
                    "score": ch.score,
                }
                for ch in chunks
            ]

            log_event(
                event_type="final",
                request_id=request_id,
                task_id=task_id,
                task_type=task_type,
                payload={
                    "ok": True,
                    "answer_length": 0,
                    "mode": "test_no_llm",
                },
            )

            return TaskQueryResponse(
                ok=True,
                answer="TEST MODE: LLM не вызывался, возвращены только найденные чанки.",
                meta=meta,
            )

        if debug:
            meta["retrieved_chunks"] = [
                {
                    "text": ch.text,
                    "source_id": ch.source_id,
                    "score": ch.score,
                }
                for ch in chunks
            ]

        system_prompt = build_llm_prompt(task_cfg, normalized_query, chunks)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            # первая попытка LLM
            t_llm0 = time.time()
            resp = await call_llm(messages, model="llama3.2:1b")
            t_llm1 = time.time()
            answer = resp["choices"][0]["message"]["content"]

            log_event(
                event_type="llm",
                request_id=request_id,
                task_id=task_id,
                task_type=task_type,
                payload={
                    "duration_ms": int((t_llm1 - t_llm0) * 1000),
                    "ok": True,
                    "profanity_detected": profanity_flag,
                    "attempt": 1,
                },
            )

            # лёгкий research: вторая попытка с расширенным контекстом (если включен)
            if task_cfg.enable_research:
                t_r0 = time.time()
                research_chunks = retrieve_text_chunks_for_research(task_cfg, normalized_query)
                t_r1 = time.time()

                log_event(
                    event_type="research_search",
                    request_id=request_id,
                    task_id=task_id,
                    task_type=task_type,
                    payload={
                        "duration_ms": int((t_r1 - t_r0) * 1000),
                        "chunks_found": len(research_chunks),
                    },
                )

                if debug:
                    meta["research_chunks"] = [
                        {
                            "text": ch.text,
                            "source_id": ch.source_id,
                            "score": ch.score,
                        }
                        for ch in research_chunks
                    ]

                research_prompt = (
                    build_llm_prompt(task_cfg, normalized_query, research_chunks)
                    + "\n\nЭто вторая попытка ответа с расширенным контекстом. "
                      "Если в новом контексте есть уточняющая информация, учти её."
                )
                research_messages = [
                    {"role": "system", "content": research_prompt},
                    {"role": "user", "content": query},
                ]

                t_llm2_0 = time.time()
                resp2 = await call_llm(research_messages, model="llama3.2:1b")
                t_llm2_1 = time.time()
                answer2 = resp2["choices"][0]["message"]["content"]

                log_event(
                    event_type="llm",
                    request_id=request_id,
                    task_id=task_id,
                    task_type=task_type,
                    payload={
                        "duration_ms": int((t_llm2_1 - t_llm2_0) * 1000),
                        "ok": True,
                        "profanity_detected": profanity_flag,
                        "attempt": 2,
                    },
                )

                answer = answer2

            # постобработка в файл
            if task_cfg.postprocessing_type == "markdown-file":
                abs_path, fname = save_markdown(answer, task_cfg)
                meta["postprocessing"] = {
                    "type": "markdown-file",
                    "path": abs_path,
                    "filename": fname,
                }
            elif task_cfg.postprocessing_type == "docx-file":
                abs_path, fname = save_docx(answer, task_cfg)
                meta["postprocessing"] = {
                    "type": "docx-file",
                    "path": abs_path,
                    "filename": fname,
                }

            log_event(
                event_type="final",
                request_id=request_id,
                task_id=task_id,
                task_type=task_type,
                payload={
                    "ok": True,
                    "answer_length": len(answer) if isinstance(answer, str) else None,
                },
            )

            return TaskQueryResponse(ok=True, answer=answer, meta=meta or None)
        except LLMError as e:
            log_event(
                event_type="llm_error",
                request_id=request_id,
                task_id=task_id,
                task_type=task_type,
                payload={"error": str(e), "profanity_detected": profanity_flag},
            )
            log_event(
                event_type="final",
                request_id=request_id,
                task_id=task_id,
                task_type=task_type,
                payload={"ok": False, "error": str(e)},
            )
            return TaskQueryResponse(ok=False, error=str(e), meta=meta or None)

    # ===== Любые corporate-задачи (общая заглушка) =====
    if task_cfg.task_type == "corporate":
        # В test-окружении не вызываем LLM
        if env == "test":
            log_event(
                event_type="final",
                request_id=request_id,
                task_id=task_id,
                task_type=task_type,
                payload={"ok": True, "answer_length": 0, "mode": "test_no_llm"},
            )
            return TaskQueryResponse(
                ok=True,
                answer="TEST MODE: LLM не вызывался, corporate-задача ещё не реализована.",
                meta=meta or None,
            )

        system_prompt = (
            "Ты ассистент корпоративной системы заявок. "
            "Пока для этой corporate-задачи нет специализированного RAG-пайплайна, "
            "ответь аккуратно, что функциональность в разработке."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            t_llm0 = time.time()
            resp = await call_llm(messages, model="llama3.2:1b")
            t_llm1 = time.time()
            answer = resp["choices"][0]["message"]["content"]

            if task_cfg.postprocessing_type == "markdown-file":
                abs_path, fname = save_markdown(answer, task_cfg)
                meta["postprocessing"] = {
                    "type": "markdown-file",
                    "path": abs_path,
                    "filename": fname,
                }
            elif task_cfg.postprocessing_type == "docx-file":
                abs_path, fname = save_docx(answer, task_cfg)
                meta["postprocessing"] = {
                    "type": "docx-file",
                    "path": abs_path,
                    "filename": fname,
                }

            log_event(
                event_type="llm",
                request_id=request_id,
                task_id=task_id,
                task_type=task_type,
                payload={
                    "duration_ms": int((t_llm1 - t_llm0) * 1000),
                    "ok": True,
                },
            )
            log_event(
                event_type="final",
                request_id=request_id,
                task_id=task_id,
                task_type=task_type,
                payload={
                    "ok": True,
                    "answer_length": len(answer) if isinstance(answer, str) else None,
                },
            )
            return TaskQueryResponse(ok=True, answer=answer, meta=meta or None)
        except LLMError as e:
            log_event(
                event_type="llm_error",
                request_id=request_id,
                task_id=task_id,
                task_type=task_type,
                payload={"error": str(e)},
            )
            log_event(
                event_type="final",
                request_id=request_id,
                task_id=task_id,
                task_type=task_type,
                payload={"ok": False, "error": str(e)},
            )
            return TaskQueryResponse(ok=False, error=str(e), meta=meta or None)

    raise HTTPException(status_code=400, detail="Unsupported task_type")


@router.get("/tasks", response_model=List[TaskInfo])
async def list_tasks() -> List[TaskInfo]:
    env = get_environment()
    tasks = task_registry.list_registered_tasks()
    visible: List[TaskInfo] = []

    for t in tasks:
        if env == "prod" and t.task_type == "demo":
            continue
        visible.append(
            TaskInfo(
                task_id=t.task_id,
                task_type=t.task_type,
                name=t.name,
                description=t.description,
            )
        )

    return visible


@router.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str) -> TaskInfo:
    env = get_environment()
    try:
        cfg = task_registry.get_task_config(task_id)
    except TaskRegistryError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if env == "prod" and cfg.task_type == "demo":
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

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
        debug=request.debug,
    )


@router.post("/query", response_model=GenericQueryResponse)
async def generic_query(request: GenericQueryRequest) -> GenericQueryResponse:
    request_id = str(uuid.uuid4())
    env = get_environment()

    classification = await classify_query(
        query=request.query,
        request_id=request_id,
        debug=request.debug,
    )

    if not classification.ok or not classification.task_id or not classification.task_type:
        return GenericQueryResponse(
            ok=False,
            error=classification.error or "classification_failed",
            classification_confidence=classification.confidence,
        )

    if env == "prod" and classification.task_type == "demo":
        return GenericQueryResponse(
            ok=False,
            error="demo tasks are not available in prod environment",
            routed_task_id=None,
            routed_task_type=None,
            classification_confidence=classification.confidence,
        )

    try:
        task_registry.get_task_config(classification.task_id)
    except TaskRegistryError as e:
        return GenericQueryResponse(
            ok=False,
            error=str(e),
            routed_task_id=classification.task_id,
            routed_task_type=classification.task_type,
            classification_confidence=classification.confidence,
        )

    task_resp = await _run_task(
        task_id=classification.task_id,
        task_type=classification.task_type,
        query=request.query,
        debug=request.debug,
    )

    return GenericQueryResponse(
        ok=task_resp.ok,
        answer=task_resp.answer,
        routed_task_id=classification.task_id,
        routed_task_type=classification.task_type,
        error=task_resp.error,
        classification_confidence=classification.confidence,
    )
