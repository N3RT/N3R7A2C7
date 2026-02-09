from dataclasses import dataclass
from typing import List, Optional

from app.core.task_config import TaskConfig
from app.core.chroma_client import get_chroma_client


@dataclass
class RetrievedChunk:
    text: str
    source_id: str
    score: float


def _get_collection_name_for_task(task_cfg: TaskConfig) -> Optional[str]:
    if task_cfg.task_id == "demo_hello":
        return "demo_hello_texts"
    if task_cfg.task_id == "demo_rules":
        return "demo_rules_texts"
    return None


def ensure_collection_for_task(task_cfg: TaskConfig):
    client = get_chroma_client()
    coll_name = _get_collection_name_for_task(task_cfg)
    if coll_name is None:
        return None

    coll = client.get_or_create_collection(name=coll_name)

    if coll.count() == 0:
        if task_cfg.task_id == "demo_hello":
            texts = [
                (
                    "demo_hello_doc_1",
                    "Это демо-документ системы заявок. Система помогает сотрудникам задавать вопросы "
                    "по регламентам и процессам, а также получать подсказки по доступам и ИТ-сервисам. "
                    "В будущем сюда будут загружены настоящие корпоративные документы и инструкции.",
                )
            ]
        elif task_cfg.task_id == "demo_rules":
            texts = [
                (
                    "demo_rules_doc_1",
                    "В компании действуют базовые регламенты: сотрудник обязан согласовывать доступы "
                    "к системам через заявки, соблюдать правила информационной безопасности и "
                    "использовать только утверждённые каналы коммуникации.",
                )
            ]
        else:
            texts = []

        if texts:
            ids = []
            docs = []
            metadatas = []
            for source_id, text in texts:
                ids.append(source_id)
                docs.append(text)
                metadatas.append({"source_id": source_id, "kind": task_cfg.task_id})
            coll.add(ids=ids, documents=docs, metadatas=metadatas)

    return coll


def retrieve_text_chunks(task_cfg: TaskConfig, query: str, top_k: int = 3) -> List[RetrievedChunk]:
    coll = ensure_collection_for_task(task_cfg)
    results: List[RetrievedChunk] = []
    if coll is None:
        return results

    ts = task_cfg.text_search
    mode = "topk"
    max_chunks = top_k
    threshold = None

    if ts is not None:
        mode = ts.mode or "topk"
        max_chunks = ts.max_chunks or top_k
        threshold = ts.similarity_threshold

    search = coll.query(
        query_texts=[query],
        n_results=max_chunks,
    )
    docs = search.get("documents", [[]])[0]
    ids = search.get("ids", [[]])[0]
    dists = search.get("distances", [[]])[0]

    raw_chunks: List[RetrievedChunk] = []
    for doc, sid, dist in zip(docs, ids, dists):
        raw_chunks.append(
            RetrievedChunk(text=doc, source_id=sid, score=float(dist))
        )

    if mode == "allwiththreshold" and threshold is not None:
        filtered = [ch for ch in raw_chunks if ch.score >= threshold]
        return filtered

    return raw_chunks[:top_k]


def retrieve_text_chunks_for_research(task_cfg: TaskConfig, query: str) -> List[RetrievedChunk]:
    """
    Упрощённый расширенный поиск для research:
    увеличиваем n_results в 2 раза относительно базового top_k.
    """
    base_top_k = 3
    ts = task_cfg.text_search
    if ts is not None and ts.topk:
        base_top_k = ts.topk

    extended_top_k = max(base_top_k * 2, base_top_k + 1)
    return retrieve_text_chunks(task_cfg, query, top_k=extended_top_k)


def build_llm_prompt(task_cfg: TaskConfig, query: str, chunks: List[RetrievedChunk]) -> str:
    base_prompt = task_cfg.technical_prompt or ""
    context_lines: List[str] = []

    for i, ch in enumerate(chunks, start=1):
        context_lines.append(
            f"{i}. [source={ch.source_id}, score={ch.score:.4f}] {ch.text}"
        )

    context_block = "\n".join(context_lines) if context_lines else ""

    full_prompt = (
        f"{base_prompt}\n\n"
        f"Контекст из базы знаний:\n{context_block}\n\n"
        f"Вопрос пользователя:\n{query}\n\n"
        f"Отвечай, опираясь только на контекст. "
        f"Если ответ неочевиден, явно скажи, что информации недостаточно."
    )
    return full_prompt
