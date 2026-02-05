from dataclasses import dataclass

from app.core.config_loader import get_environment
from app.core.task_config import TaskConfig


@dataclass
class AccessDecision:
    allowed: bool
    reason: str


def check_task_access(task_cfg: TaskConfig) -> AccessDecision:
    """
    Простейшая модель доступа:
    - demo-задачи разрешены в любом окружении;
    - corporate-задачи тоже разрешены в любом окружении (ограничение по dev снято).
    В будущем сюда можно добавить проверку ролей пользователя и другие правила.
    """
    env = get_environment()

    if task_cfg.task_type == "demo":
        return AccessDecision(
            allowed=True,
            reason=f"demo task in environment={env} is allowed",
        )

    if task_cfg.task_type == "corporate":
        return AccessDecision(
            allowed=True,
            reason=f"corporate task in environment={env} is allowed",
        )

    return AccessDecision(
        allowed=False,
        reason=f"Unknown task_type={task_cfg.task_type}",
    )
