import sys
from pathlib import Path
from typing import List

import click

# Добавляем корень проекта в sys.path, чтобы импортировать app.*
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.task_registry import task_registry  # noqa: E402
from app.core.task_config import TaskConfigError  # noqa: E402
from app.core.config_loader import get_environment  # noqa: E402


@click.group()
def cli() -> None:
    """
    Админ-CLI для RAG-системы (режим n3r7).
    """
    pass


@cli.command("env")
def show_env() -> None:
    """
    Показать текущий environment (dev/test/prod).
    """
    env = get_environment()
    click.echo(f"Environment: {env}")


@cli.command("task-info")
@click.argument("task_id", type=str)
def task_info(task_id: str) -> None:
    """
    Показать информацию о задаче по task_id.
    """
    try:
        cfg = task_registry.get_task_config(task_id)
    except TaskConfigError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Registry error: {e}")
        sys.exit(1)

    click.echo(f"task_id     : {cfg.task_id}")
    click.echo(f"name        : {cfg.name}")
    click.echo(f"description : {cfg.description}")
    click.echo(f"task_type   : {cfg.task_type}")


@cli.command("tasks-loaded")
def tasks_loaded() -> None:
    """
    Показать задачи, уже загруженные в TaskRegistry за время жизни процесса.
    """
    tasks = task_registry.list_registered_tasks()
    if not tasks:
        click.echo("No tasks loaded in registry yet.")
        return

    click.echo("Loaded tasks:")
    for t in tasks:
        click.echo(f"- {t.task_id} [{t.task_type}] — {t.name}")


if __name__ == "__main__":
    cli()
