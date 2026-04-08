"""FastAPI server for the Finlytics OpenEnv container deployment."""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .env import FinancialPlanningEnv
from .models import FinancialAction
from .scenarios import TASK_IDS, TASKS


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default="budget_easy")


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: FinancialAction


app = FastAPI(
    title="Finlytics OpenEnv",
    description="OpenEnv-compatible financial planning environment for real-world agent evaluation.",
    version="1.0.0",
)

_SESSIONS: dict[str, FinancialPlanningEnv] = {}


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "Finlytics OpenEnv",
        "tasks": TASK_IDS,
        "task_count": len(TASK_IDS),
        "environment": "personal-finance-planning",
        "docs": "/docs",
    }


@app.get("/tasks")
def list_tasks() -> list[dict[str, object]]:
    return [task.model_dump() for task in TASKS.values()]


@app.post("/sessions")
def create_session(request: CreateSessionRequest) -> dict[str, object]:
    try:
        env = FinancialPlanningEnv(task_id=request.task_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    session_id = str(uuid4())
    _SESSIONS[session_id] = env
    return {"session_id": session_id, "observation": env.reset().model_dump()}


@app.get("/sessions/{session_id}/state")
def get_state(session_id: str) -> dict[str, object]:
    env = _get_session(session_id)
    return env.state().model_dump()


@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str) -> dict[str, object]:
    env = _get_session(session_id)
    return {"observation": env.reset().model_dump()}


@app.post("/sessions/{session_id}/step")
def step_session(session_id: str, request: StepRequest) -> dict[str, object]:
    env = _get_session(session_id)
    try:
        observation, reward, done, info = env.step(request.action)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/sessions/{session_id}/grade")
def grade_session(session_id: str) -> dict[str, object]:
    env = _get_session(session_id)
    return env.grade().model_dump()


def _get_session(session_id: str) -> FinancialPlanningEnv:
    try:
        return _SESSIONS[session_id]
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Unknown session_id '{session_id}'.") from error
