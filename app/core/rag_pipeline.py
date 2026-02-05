from dataclasses import dataclass
from typing import List

from app.core.task_config import TaskConfig
from app.core.chroma_client import get_chroma_client


@dataclass
class RetrievedChunk:
    text: str
    source_id: str
    score: float


def ensure_demo_hello_collection():
    """
    Для demo_hello делаем ленивую инициализацию коллекции с одним демо-документом.
    В реальном проекте сюда приедет полноценный ingestion.
    """
    client = get_chroma_client()
    coll = client.get_or_create_collection(name="demo_hello_texts")

    # Проверяем, не пустая ли коллекция.
    existing = coll.count()
    if existing > 0:
        return coll

    demo_doc = (
        "Это демо-документ системы заявок. "
        "Система помогает сотрудникам задавать вопросы по регламентам и процессам, "
        "а также получать подсказки по доступам и ИТ-сервисам. "
        "В будущем сюда будут загружены настоящие корпоративные документы и инструкции."
    )

    coll.add(
        ids=["demo_doc_1"],
        documents=[demo_doc],
        metadatas=[{"source_id": "demo_doc_1", "kind": "demo"}],
    )
    return coll


def retrieve_text_chunks(task_cfg: TaskConfig, query: str, top_k: int = 3) -> List[RetrievedChunk]:
    """
    Минимальный поиск по текстам для demo_hello.
    """
    if task_cfg.task_id != "demo_hello":
        # Пока поддерживаем только одну демо-задачу.
        return []

    coll = ensure_demo_hello_collection()
    res = coll.query(query_texts=[query], n_results=top_k)

    docs = res.get("documents", [[]])[0]
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]  # чем меньше, тем лучше

    chunks: List[RetrievedChunk] = []
    for doc, _id, dist in zip(docs, ids, dists):
        chunks.append(
            RetrievedChunk(
                text=doc,
                source_id=_id,
                score=float(dist),
            )
        )
    return chunks


def build_llm_prompt(task_cfg: TaskConfig, query: str, chunks: List[RetrievedChunk]) -> str:
    """
    Формирует системный промпт для LLM: технический промпт + найденный контекст.
    """
    base_prompt = task_cfg.technical_prompt or (
        "Ты ассистент RAG-системы. Отвечай по делу, используй только приведённый контекст."
    )

    context_lines: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        context_lines.append(f"[Фрагмент {i}, source={ch.source_id}, score={ch.score:.4f}]\n{ch.text}")

    context_block = "\n\n".join(context_lines) if context_lines else "Контекст по документам не найден."

    full_prompt = (
        f"{base_prompt}\n\n"
        f"Ниже приведён контекст из внутренних документов:\n"
        f"{context_block}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        f"Отвечай кратко, опираясь на контекст. "
        f"Если информации не хватает, честно скажи об этом."
    )
    return full_prompt
