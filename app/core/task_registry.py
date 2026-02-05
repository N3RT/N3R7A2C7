from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from app.core.task_config import TaskConfig, load_task_config, TaskConfigError


@dataclass
class RegisteredTask:
    task_id: str
    task_type: str  # "demo" | "corporate"
    name: str
    description: str


class TaskRegistryError(Exception):
    pass


class TaskRegistry:
    """
    Файловый реестр задач поверх TaskConfig (config/tasks/*.yaml).
    Кэширует загруженные конфиги и умеет сканировать директорию задач.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskConfig] = {}

    def get_task_config(self, task_id: str) -> TaskConfig:
        if task_id in self._tasks:
            return self._tasks[task_id]
        try:
            cfg = load_task_config(task_id)
        except TaskConfigError as e:
            raise TaskRegistryError(str(e)) from e
        self._tasks[task_id] = cfg
        return cfg

    def _scan_tasks_dir(self) -> List[str]:
        """
        Находит все config/tasks/*.yaml и возвращает список task_id.
        """
        tasks_dir = Path(__file__).parent.parent.parent / "config" / "tasks"
        if not tasks_dir.is_dir():
            return []

        task_ids: List[str] = []
        for path in tasks_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            tid = data.get("task_id")
            if isinstance(tid, str):
                task_ids.append(tid)
        return sorted(set(task_ids))

    def list_registered_tasks(self) -> List[RegisteredTask]:
        """
        Возвращает список задач по файловой системе (config/tasks/*.yaml),
        загружая TaskConfig при необходимости.
        """
        result: List[RegisteredTask] = []

        for task_id in self._scan_tasks_dir():
            try:
                cfg = self.get_task_config(task_id)
            except TaskRegistryError:
                continue
            result.append(
                RegisteredTask(
                    task_id=cfg.task_id,
                    task_type=cfg.task_type,
                    name=cfg.name,
                    description=cfg.description,
                )
            )
        return result


# Глобальный singleton, чтобы не плодить реестры
task_registry = TaskRegistry()
