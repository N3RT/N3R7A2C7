from pathlib import Path
from functools import lru_cache
from typing import Any, Dict

import yaml


class SystemConfigError(Exception):
    pass


@lru_cache(maxsize=1)
def load_system_config() -> Dict[str, Any]:
    """
    Загружает config/system.yaml один раз за жизнь процесса.
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "system.yaml"
    if not config_path.is_file():
        raise SystemConfigError(f"Config file not found: {config_path}")

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemConfigError(f"Failed to read system config: {e}") from e

    if not isinstance(data, dict):
        raise SystemConfigError("System config must be a mapping at top level")

    return data


def get_environment() -> str:
    cfg = load_system_config()
    env = cfg.get("environment", "dev")
    if env not in {"dev", "test", "prod"}:
        raise SystemConfigError(f"Invalid environment in config: {env}")
    return env


def get_llm_mode() -> str:
    cfg = load_system_config()
    llm = cfg.get("llm", {}) or {}
    mode = llm.get("mode", "dev")
    return mode


def get_llm_connector_path() -> str:
    cfg = load_system_config()
    llm = cfg.get("llm", {}) or {}
    connector = llm.get("connector")
    if not connector:
        raise SystemConfigError("llm.connector is not set in system config")
    return connector
