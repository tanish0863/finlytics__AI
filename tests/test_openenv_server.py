"""Tests for the Finlytics OpenEnv server compatibility routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from openenv.server import app


client = TestClient(app)


def test_root_and_tasks_endpoints() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Finlytics OpenEnv"
    assert "tasks" in payload

    response = client.get("/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert isinstance(tasks, list)
    assert len(tasks) == 3


def test_sessionless_reset_step_and_grade() -> None:
    reset_response = client.post("/reset")
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert "observation" in reset_payload

    action = {
        "spend_pct": 0.25,
        "save_pct": 0.25,
        "invest_pct": 0.25,
        "debt_payment_pct": 0.25,
        "rationale": "Balanced allocation for compatibility testing.",
    }
    step_response = client.post("/step", json={"action": action})
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert "observation" in step_payload
    assert "reward" in step_payload
    assert step_payload["done"] in (True, False)

    grade_response = client.get("/grade")
    assert grade_response.status_code == 200
    grade_payload = grade_response.json()
    assert "score" in grade_payload


def test_openenv_alias_endpoints() -> None:
    reset_response = client.post("/openenv/reset")
    assert reset_response.status_code == 200
    assert "observation" in reset_response.json()

    action = {
        "spend_pct": 0.30,
        "save_pct": 0.30,
        "invest_pct": 0.20,
        "debt_payment_pct": 0.20,
        "rationale": "OpenEnv alias endpoint test.",
    }
    step_response = client.post("/openenv/step", json={"action": action})
    assert step_response.status_code == 200
    assert "reward" in step_response.json()

    state_response = client.get("/openenv/state")
    assert state_response.status_code == 200
    assert "task_id" in state_response.json()

    grade_response = client.get("/openenv/grade")
    assert grade_response.status_code == 200
    assert "score" in grade_response.json()
