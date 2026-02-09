from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from pathlib import Path
import yaml


class TaskConfigError(Exception):
    pass


@dataclass
class TextSearchConfig:
    enabled: bool = True
    embedding_model: str = "default-multilingual"
    mode: str = "topk"  # topk / allwiththreshold / hybrid
    topk: int = 5
    max_chunks: int = 20
    similarity_threshold: float = 0.3
    chunker: str = "semanticsplit"  # semanticsplit / fixedsize / none

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextSearchConfig":
        if not isinstance(data, dict):
            raise TaskConfigError("text_search must be a mapping")
        return cls(
            enabled=bool(data.get("enabled", True)),
            embedding_model=str(data.get("embedding_model", "default-multilingual")),
            mode=str(data.get("mode", "topk")),
            topk=int(data.get("topk", 5)),
            max_chunks=int(data.get("max_chunks", 20)),
            similarity_threshold=float(data.get("similarity_threshold", 0.3)),
            chunker=str(data.get("chunker", "semanticsplit")),
        )


@dataclass
class TableSearchConfig:
    enabled: bool = False
    embedding_model: str = "technical-ru"
    mode: str = "topk"  # topk / allwiththreshold / hybrid
    topk: int = 5
    max_rows: int = 50
    similarity_threshold: float = 0.4
    chunker: str = "rowtochunk"  # rowtochunk / fulltable

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableSearchConfig":
        if not isinstance(data, dict):
            raise TaskConfigError("table_search must be a mapping")
        return cls(
            enabled=bool(data.get("enabled", False)),
            embedding_model=str(data.get("embedding_model", "technical-ru")),
            mode=str(data.get("mode", "topk")),
            topk=int(data.get("topk", 5)),
            max_rows=int(data.get("max_rows", 50)),
            similarity_threshold=float(data.get("similarity_threshold", 0.4)),
            chunker=str(data.get("chunker", "rowtochunk")),
        )


@dataclass
class TaskConfig:
    task_id: str
    name: str
    description: str
    task_type: str  # demo / corporate
    technical_prompt: str

    # Источники и режимы поиска
    sources_mode: str = "text"  # text / tables / texttables
    enable_text_search: bool = True
    enable_table_search: bool = False
    enable_research: bool = False
    postprocessing_type: str = "none"  # none / markdown-file / docx-file / customscript

    text_search: Optional[TextSearchConfig] = None
    table_search: Optional[TableSearchConfig] = None

    # Контекст и метаданные
    context_format: str = "structured_json"  # structured_json / plain_text
    preserve_full_query: bool = True
    include_sources_meta: bool = True
    meta_mode: str = "separate"  # inline / separate
    meta_fields: Optional[List[str]] = None

    # Research (итеративный поиск)
    research_enabled: bool = False
    research_trigger_on_low_confidence: bool = True
    research_max_iterations: int = 2
    research_min_confidence: float = 0.6

    # История диалога
    history_max_messages: int = 10
    history_ttl_days: int = 90

    # Логирование
    logging_level: str = "info"
    logging_log_prompts: bool = True
    logging_log_search_results: bool = True
    logging_log_postprocessing: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        try:
            task_id = data["task_id"]
            name = data.get("name", task_id)
            description = data.get("description", "")
            task_type = data.get("task_type", "demo")
            technical_prompt = data.get("technical_prompt", "")
        except KeyError as e:
            raise TaskConfigError(f"Missing required field in TaskConfig: {e}") from e

        if task_type not in ("demo", "corporate"):
            raise TaskConfigError(f"Invalid task_type in TaskConfig: {task_type}")

        sources_mode = data.get("sources_mode", "text")
        if sources_mode not in ("text", "tables", "texttables"):
            raise TaskConfigError(f"Invalid sources_mode in TaskConfig: {sources_mode}")

        enable_text_search = bool(data.get("enable_text_search", True))
        enable_table_search = bool(data.get("enable_table_search", False))
        enable_research = bool(data.get("enable_research", False))

        postprocessing_type = data.get("postprocessing_type", "none")
        if postprocessing_type not in ("none", "markdown-file", "docx-file", "customscript"):
            raise TaskConfigError(
                f"Invalid postprocessing_type in TaskConfig: {postprocessing_type}"
            )

        # Вложенные секции поиска
        text_search_cfg: Optional[TextSearchConfig] = None
        table_search_cfg: Optional[TableSearchConfig] = None

        ts_data = data.get("text_search")
        if ts_data is not None:
            text_search_cfg = TextSearchConfig.from_dict(ts_data)

        tbl_data = data.get("table_search")
        if tbl_data is not None:
            table_search_cfg = TableSearchConfig.from_dict(tbl_data)

        # Контекст и метаданные
        context_cfg = data.get("context") or {}
        context_format = context_cfg.get("format", "structured_json")
        preserve_full_query = bool(context_cfg.get("preserve_full_query", True))
        include_sources_meta = bool(context_cfg.get("include_sources_meta", True))
        meta_mode = context_cfg.get("meta_mode", "separate")
        meta_fields = context_cfg.get("meta_fields")
        if meta_fields is not None and not isinstance(meta_fields, list):
            raise TaskConfigError("context.meta_fields must be a list if provided")

        # Research
        research_cfg = data.get("research") or {}
        research_enabled = bool(research_cfg.get("enabled", False))
        research_trigger_on_low_confidence = bool(
            research_cfg.get("trigger_on_low_confidence", True)
        )
        research_max_iterations = int(research_cfg.get("max_iterations", 2))
        research_min_confidence = float(research_cfg.get("min_confidence", 0.6))

        # История
        history_cfg = data.get("history") or {}
        history_max_messages = int(history_cfg.get("max_messages", 10))
        history_ttl_days = int(history_cfg.get("ttl_days", 90))

        # Логирование
        logging_cfg = data.get("logging") or {}
        logging_level = logging_cfg.get("level", "info")
        logging_log_prompts = bool(logging_cfg.get("log_prompts", True))
        logging_log_search_results = bool(logging_cfg.get("log_search_results", True))
        logging_log_postprocessing = bool(logging_cfg.get("log_postprocessing", True))

        return cls(
            task_id=task_id,
            name=name,
            description=description,
            task_type=task_type,
            technical_prompt=technical_prompt,
            sources_mode=sources_mode,
            enable_text_search=enable_text_search,
            enable_table_search=enable_table_search,
            enable_research=enable_research,
            postprocessing_type=postprocessing_type,
            text_search=text_search_cfg,
            table_search=table_search_cfg,
            context_format=context_format,
            preserve_full_query=preserve_full_query,
            include_sources_meta=include_sources_meta,
            meta_mode=meta_mode,
            meta_fields=meta_fields,
            research_enabled=research_enabled,
            research_trigger_on_low_confidence=research_trigger_on_low_confidence,
            research_max_iterations=research_max_iterations,
            research_min_confidence=research_min_confidence,
            history_max_messages=history_max_messages,
            history_ttl_days=history_ttl_days,
            logging_level=logging_level,
            logging_log_prompts=logging_log_prompts,
            logging_log_search_results=logging_log_search_results,
            logging_log_postprocessing=logging_log_postprocessing,
        )


def load_task_config(task_id: str) -> TaskConfig:
    """
    Загружает TaskConfig из config/tasks/<task_id>.yaml.
    """
    tasks_dir = Path(__file__).parent.parent.parent / "config" / "tasks"
    cfg_path = tasks_dir / f"{task_id}.yaml"
    if not cfg_path.is_file():
        raise TaskConfigError(f"Task config not found for task_id={task_id} ({cfg_path})")

    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise TaskConfigError(f"Failed to read TaskConfig YAML: {e}") from e

    if not isinstance(data, dict):
        raise TaskConfigError("TaskConfig YAML must be a mapping at top level")

    return TaskConfig.from_dict(data)
