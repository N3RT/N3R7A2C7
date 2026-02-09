import sys
from pathlib import Path
from typing import List

import click

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.task_registry import task_registry, TaskRegistryError  # noqa: E402
from app.core.task_config import TaskConfigError  # noqa: E402
from app.core.config_loader import get_environment  # noqa: E402


@click.group()
def cli() -> None:
    """
    Админ-CLI для RAG-сервиса n3r7.
    Позволяет смотреть окружение, список задач и подробный конфиг задачи.
    """
    pass


@cli.command("env")
def show_env() -> None:
    env = get_environment()
    click.echo(f"Environment: {env}")


@cli.command("tasks-loaded")
def tasks_loaded() -> None:
    tasks = task_registry.list_registered_tasks()
    if not tasks:
        click.echo("No tasks loaded in registry yet.")
        return

    click.echo("Loaded tasks:")
    for t in tasks:
        click.echo(f"- {t.task_id} [{t.task_type}] {t.name}")


@cli.command("tasks-list")
def tasks_list() -> None:
    """
    Красивый список задач с типом и режимом sources_mode.
    """
    tasks = task_registry.list_registered_tasks()
    if not tasks:
        click.echo("No tasks registered.")
        return

    from app.core.task_config import TaskConfig  # noqa: E402

    click.echo("Registered tasks:")
    click.echo("------------------------------------------------------------")
    click.echo(f"{'task_id':20} {'type':10} {'sources.mode':14} name")
    click.echo("------------------------------------------------------------")

    for t in tasks:
        try:
            cfg: TaskConfig = task_registry.get_task_config(t.task_id)
            sources_mode = cfg.sources_mode
        except TaskRegistryError:
            sources_mode = "?"
        click.echo(
            f"{t.task_id:20} {t.task_type:10} {sources_mode:14} {t.name}"
        )


@cli.command("task-info")
@click.argument("task_id", type=str)
def task_info(task_id: str) -> None:
    """
    Показать подробную информацию о задаче по её task_id,
    включая режимы поиска и чанкер.
    """
    from app.core.task_config import TaskConfig  # noqa: E402

    try:
        cfg: TaskConfig = task_registry.get_task_config(task_id)
    except TaskConfigError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)
    except TaskRegistryError as e:
        click.echo(f"Registry error: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}")
        sys.exit(1)

    click.echo(f"task_id       : {cfg.task_id}")
    click.echo(f"name          : {cfg.name}")
    click.echo(f"description   : {cfg.description}")
    click.echo(f"task_type     : {cfg.task_type}")
    click.echo(f"sources.mode  : {cfg.sources_mode}")
    click.echo(f"text_search   : {cfg.enable_text_search}")
    click.echo(f"table_search  : {cfg.enable_table_search}")
    click.echo(f"postprocessing: {cfg.postprocessing_type}")

    # Детали text_search
    if cfg.text_search is not None:
        ts = cfg.text_search
        click.echo("\ntext_search:")
        click.echo(f"  enabled             : {ts.enabled}")
        click.echo(f"  embedding_model     : {ts.embedding_model}")
        click.echo(f"  mode                : {ts.mode}")
        click.echo(f"  topk                : {ts.topk}")
        click.echo(f"  max_chunks          : {ts.max_chunks}")
        click.echo(f"  similarity_threshold: {ts.similarity_threshold}")
        click.echo(f"  chunker             : {ts.chunker}")

    # Детали table_search
    if cfg.table_search is not None:
        tbl = cfg.table_search
        click.echo("\ntable_search:")
        click.echo(f"  enabled             : {tbl.enabled}")
        click.echo(f"  embedding_model     : {tbl.embedding_model}")
        click.echo(f"  mode                : {tbl.mode}")
        click.echo(f"  topk                : {tbl.topk}")
        click.echo(f"  max_rows            : {tbl.max_rows}")
        click.echo(f"  similarity_threshold: {tbl.similarity_threshold}")
        click.echo(f"  chunker             : {tbl.chunker}")

    click.echo("\ntechnical_prompt:")
    click.echo(cfg.technical_prompt or "(empty)")


@cli.command("ingest-employee-data")
def ingest_employee_data() -> None:
    """
    Ingestion демо-данных сотрудников в Chroma (collection=employee_data_rows).
    Берёт строки из SQLite (ensure_employee_table + seed_demo_employees),
    конвертирует в чанки (row_to_chunk) и пишет в векторку.
    """
    from app.ingestion.employee_table import ensure_employee_table, seed_demo_employees
    from app.ingestion.employee_ingest import ingest_employee_table_to_chroma

    click.echo("Ensuring employee table and seeding demo employees...")
    ensure_employee_table()
    seed_demo_employees()

    click.echo("Ingesting employee table into Chroma collection 'employee_data_rows'...")
    ingest_employee_table_to_chroma(collection_name="employee_data_rows")
    click.echo("Done.")


if __name__ == "__main__":
    cli()
