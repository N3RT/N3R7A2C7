from dataclasses import dataclass

from app.core.config_loader import get_environment
from app.core.task_config import TaskConfig


@dataclass
class AccessDecision:
    allowed: bool
    reason: str


def check_task_access(task_cfg: TaskConfig) -> AccessDecision:
    """
    Реализация правил из ТЗ:
    - demo: можно в prod/dev/test;
    - corporate: только в prod, в dev/test — блок.
    """
    env = get_environment()

    if task_cfg.task_type == "demo":
        return AccessDecision(
            allowed=True,
            reason=f"demo task in environment={env} is allowed",
        )

    if task_cfg.task_type == "corporate":
        if env == "prod":
            return AccessDecision(
                allowed=True,
                reason="corporate task in prod is allowed",
            )
        return AccessDecision(
            allowed=False,
            reason="corporate tasks are allowed only in prod",
        )

    return AccessDecision(
        allowed=False,
        reason=f"Unknown task_type={task_cfg.task_type}",
    )
