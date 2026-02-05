from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


class TaskConfigError(Exception):
    pass


@dataclass
class TaskConfig:
    task_id: str
    name: str
    description: str
    task_type: str  # "demo" | "corporate"
    technical_prompt: str

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

        if task_type not in {"demo", "corporate"}:
            raise TaskConfigError(f"Invalid task_type in TaskConfig: {task_type}")

        return cls(
            task_id=task_id,
            name=name,
            description=description,
            task_type=task_type,
            technical_prompt=technical_prompt,
        )


def load_task_config(task_id: str) -> TaskConfig:
    """
    Загружает config/tasks/<task_id>.yaml и парсит в TaskConfig.
    Сейчас используем сильно упрощённое подмножество полей из ТЗ.
    """
    tasks_dir = Path(__file__).parent.parent.parent / "config" / "tasks"
    cfg_path = tasks_dir / f"{task_id}.yaml"

    if not cfg_path.is_file():
        raise TaskConfigError(f"Task config not found for task_id={task_id}: {cfg_path}")

    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise TaskConfigError(f"Failed to read TaskConfig YAML: {e}") from e

    if not isinstance(data, dict):
        raise TaskConfigError("TaskConfig YAML must be a mapping at top level")

    return TaskConfig.from_dict(data)
