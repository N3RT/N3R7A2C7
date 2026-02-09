from dataclasses import dataclass

from app.core.config_loader import get_environment, load_system_config
from app.core.task_config import TaskConfig


@dataclass
class AccessDecision:
    allowed: bool
    reason: str


def _allow_corporate_in_dev() -> bool:
    """
    Читает флаг из config/system.yaml:
    accesscontrol.allow_corporate_in_dev: true/false
    """
    cfg = load_system_config()
    ac = cfg.get("accesscontrol", {}) or {}
    return bool(ac.get("allow_corporate_in_dev", False))


def check_task_access(task_cfg: TaskConfig) -> AccessDecision:
    """
    Централизованная политика доступа к задачам по окружению и типу.

    Правила:
    - demo:
        - dev:  разрешено
        - test: разрешено (визуальный отладчик без LLM)
        - prod: ЗАПРЕЩЕНО
    - corporate:
        - prod: всегда разрешено
        - dev:  разрешено, если accesscontrol.allow_corporate_in_dev = true
        - test: разрешено (для отладки без LLM)
    """
    env = get_environment()

    # DEMO: только dev и test, prod запрещён
    if task_cfg.task_type == "demo":
        if env in ("dev", "test"):
            return AccessDecision(
                allowed=True,
                reason=f"demo task in environment={env} is allowed",
            )
        return AccessDecision(
            allowed=False,
            reason=f"demo tasks are not allowed in environment={env} by policy",
        )

    # CORPORATE: завязано на окружение и флаг в конфиге
    if task_cfg.task_type == "corporate":
        if env == "prod":
            return AccessDecision(
                allowed=True,
                reason="corporate task is allowed in prod",
            )
        if env == "dev" and _allow_corporate_in_dev():
            return AccessDecision(
                allowed=True,
                reason="corporate task allowed in dev for development purposes",
            )
        if env == "test":
            # В test-режиме мы не зовём LLM, но хотим видеть контекст для corporate
            return AccessDecision(
                allowed=True,
                reason="corporate task allowed in test for debugging without LLM",
            )
        return AccessDecision(
            allowed=False,
            reason=f"corporate tasks are not allowed in environment={env} by policy",
        )

    return AccessDecision(
        allowed=False,
        reason=f"Unknown task_type={task_cfg.task_type}",
    )
