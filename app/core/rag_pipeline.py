from dataclasses import dataclass
from typing import List

from app.core.task_config import TaskConfig
from app.core.chroma_client import get_chroma_client


@dataclass
class RetrievedChunk:
    text: str
    source_id: str
    score: float


def ensure_collection_for_task(task_cfg: TaskConfig):
    """
    Создаёт (при первом обращении) и заполняет демо-коллекции для задач.
    Для реальных задач вместо этого будет отдельный ingestion.
    """
    client = get_chroma_client()

    # Демонстрационная коллекция для приветственной задачи
    if task_cfg.task_id == "demo_hello":
        coll = client.get_or_create_collection(name="demo_hello_texts")
        if coll.count() > 0:
            return coll

        demodoc = (
            "Этот демо-сервис показывает, как работает RAG-система для внутренних задач компании. "
            "Он умеет отвечать на общие вопросы о сервисе заявок, доступах к ИТ-системам и базовых регламентах. "
            "В реальной эксплуатации сюда будут загружены настоящие корпоративные документы, инструкции и регламенты."
        )
        coll.add(
            ids=["demo_hello_doc1"],
            documents=[demodoc],
            metadatas=[{"source_id": "demo_hello_doc1", "kind": "demo"}],
        )
        return coll

    # Демонстрационная коллекция для регламентов и правил
    if task_cfg.task_id == "demo_rules":
        coll = client.get_or_create_collection(name="demo_rules_texts")
        if coll.count() > 0:
            return coll

        texts = [
            (
                "Регламент подачи заявок. "
                "Все запросы на доступ к ИТ-системам оформляются через портал заявок. "
                "Сотрудник указывает систему, тип доступа и обоснование. "
                "Заявка автоматически уходит на согласование руководителю и, при необходимости, в службу информационной безопасности."
            ),
            (
                "Регламент согласования доступов. "
                "Руководитель отвечает за подтверждение бизнес-необходимости доступа. "
                "Служба ИБ проверяет соответствие запроса политике безопасности. "
                "Без согласования руководителя и ИБ доступ предоставляться не может."
            ),
            (
                "Регламент работы с инцидентами. "
                "Инциденты по ИТ-системам фиксируются в системе заявок с категорией 'Инцидент'. "
                "Для критичных инцидентов действует сокращённое время реакции и уведомление дежурной смены по заранее настроенным каналам."
            ),
            (
                "Общие правила безопасности. "
                "Запрещена передача корпоративных паролей третьим лицам. "
                "Работа с конфиденциальными данными допускается только на корпоративных устройствах. "
                "Подозрительная активность должна быть немедленно сообщена в службу ИБ."
            ),
        ]
        ids = [f"demo_rules_doc{i}" for i in range(1, len(texts) + 1)]
        metadatas = [
            {"source_id": ids[i], "kind": "demo_rules"} for i in range(len(texts))
        ]

        coll.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
        return coll

    return None


def retrieve_text_chunks(task_cfg: TaskConfig, query: str, top_k: int = 3) -> List[RetrievedChunk]:
    """
    Простейший retrieval по текстам ChromaDB для демо-задач.
    """
    coll = ensure_collection_for_task(task_cfg)
    res: List[RetrievedChunk] = []

    if coll is None:
        return res

    search = coll.query(query_texts=[query], n_results=top_k)
    docs = search.get("documents", [[]])[0]
    ids = search.get("ids", [[]])[0]
    dists = search.get("distances", [[]])[0]

    chunks: List[RetrievedChunk] = []
    for doc, id_, dist in zip(docs, ids, dists):
        chunks.append(
            RetrievedChunk(
                text=doc,
                source_id=id_,
                score=float(dist),
            )
        )
    return chunks


def build_llm_prompt(task_cfg: TaskConfig, query: str, chunks: List[RetrievedChunk]) -> str:
    """
    Формирует системный промпт для LLM на основе TaskConfig и найденных фрагментов.
    """
    base_prompt = (
        task_cfg.technical_prompt
        or "Ты помощник по внутренним регламентам и сервисам компании. Отвечай кратко и по делу."
    )

    context_lines = []
    for i, ch in enumerate(chunks, start=1):
        context_lines.append(f"{i}. [source={ch.source_id}, score={ch.score:.4f}] {ch.text}")

    context_block = "\n".join(context_lines) if context_lines else "Контекстов по запросу не найдено."

    full_prompt = (
        f"{base_prompt}\n\n"
        f"Ниже приведён контекст, найденный по запросу пользователя:\n"
        f"{context_block}\n\n"
        f"Запрос пользователя:\n"
        f"{query}\n\n"
        f"Дай понятный и аккуратный ответ на русском языке. "
        f"Если контекста не хватает, явно скажи об этом и ответь в общих чертах."
    )
    return full_prompt
