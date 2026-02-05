from dataclasses import dataclass
from typing import Dict, List

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
    Простой файловый реестр задач поверх TaskConfig (config/tasks/*.yaml).
    На данном этапе кэшируем только те задачи, к которым реально обращались.
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

    def list_registered_tasks(self) -> List[RegisteredTask]:
        """
        На данном этапе просто возвращаем список уже загруженных задач.
        В будущем сюда добавится сканирование config/tasks/*.yaml.
        """
        result: List[RegisteredTask] = []
        for cfg in self._tasks.values():
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
