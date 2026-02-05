import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402


client = TestClient(app)


def test_health_ok():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["environment"] in {"dev", "test", "prod"}
    assert "llm_mode" in data


def test_tasks_list_contains_demo_tasks():
    resp = client.get("/api/v1/tasks")
    assert resp.status_code == 200
    data = resp.json()
    task_ids = {t["task_id"] for t in data}
    assert "demo_hello" in task_ids
    assert "demo_rules" in task_ids


def test_get_task_demo_rules():
    resp = client.get("/api/v1/tasks/demo_rules")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "demo_rules"
    assert data["task_type"] == "demo"
    assert "регламенты" in data["name"]


def test_task_query_demo_hello():
    resp = client.post(
        "/api/v1/task/query",
        json={
            "task_id": "demo_hello",
            "task_type": "demo",
            "query": "Тестовый вопрос про систему заявок.",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert isinstance(data["answer"], str)
    assert data["answer"]


def test_task_query_corporate_forbidden_in_dev():
    resp = client.post(
        "/api/v1/task/query",
        json={
            "task_id": "demo_hello",  # task_type умышленно выставляем corporate
            "task_type": "corporate",
            "query": "Попробуй сделать корпоративный запрос в dev.",
        },
    )
    assert resp.status_code in {400, 403}
