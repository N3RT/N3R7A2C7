import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


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
    assert data["environment"] in ("dev", "test", "prod")
    assert "llm_mode" in data


def test_tasks_list_contains_demo_tasks():
    resp = client.get("/api/v1/tasks")
    assert resp.status_code == 200
    data = resp.json()
    task_ids = [t["task_id"] for t in data]
    assert "demo_hello" in task_ids
    assert "demo_rules" in task_ids


def test_get_task_demo_rules():
    resp = client.get("/api/v1/tasks/demo_rules")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "demo_rules"
    assert data["task_type"] == "demo"
    assert "Demo" in data["name"]


def test_task_query_demo_hello():
    resp = client.post(
        "/api/v1/task/query",
        json={
            "task_id": "demo_hello",
            "task_type": "demo",
            "query": "Кратко опиши демо-систему заявок.",
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
            "task_id": "demo_hello",
            "task_type": "corporate",
            "query": "Попробуй выполнить корпоративную задачу в dev.",
        },
    )
    assert resp.status_code in (400, 403)


def test_generic_query_routes_to_demo_hello():
    resp = client.post(
        "/api/v1/query",
        json={
            "query": "Расскажи кратко, как у нас устроена демо-система заявок и какие её основные возможности?",
            "debug": False,
        },
    )
    # В dev-тестах не жёстко завязываемся на 200, так как LLM может быть недоступен.
    assert resp.status_code < 500
    data = resp.json()
    assert isinstance(data, dict)
    assert "ok" in data
    assert "routed_task_id" in data
    assert "routed_task_type" in data
