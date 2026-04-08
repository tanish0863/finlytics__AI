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
_DEFAULT_ENV: FinancialPlanningEnv | None = None


def _get_default_env() -> FinancialPlanningEnv:
    global _DEFAULT_ENV
    if _DEFAULT_ENV is None:
        _DEFAULT_ENV = FinancialPlanningEnv()
    return _DEFAULT_ENV


def _get_session(session_id: str) -> FinancialPlanningEnv:
    try:
        return _SESSIONS[session_id]
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Unknown session_id '{session_id}'.") from error


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


@app.post("/reset")
@app.post("/reset/")
def reset_default() -> dict[str, object]:
    env = _get_default_env()
    return {"observation": env.reset().model_dump()}


@app.post("/step")
@app.post("/step/")
def step_default(request: StepRequest) -> dict[str, object]:
    env = _get_default_env()
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


@app.get("/state")
@app.get("/state/")
def get_state_default() -> dict[str, object]:
    env = _get_default_env()
    return env.state().model_dump()


@app.get("/grade")
@app.get("/grade/")
def grade_default() -> dict[str, object]:
    env = _get_default_env()
    return env.grade().model_dump()


@app.post("/openenv/reset")
@app.post("/openenv/reset/")
def openenv_reset() -> dict[str, object]:
    return reset_default()


@app.post("/openenv/step")
@app.post("/openenv/step/")
def openenv_step(request: StepRequest) -> dict[str, object]:
    return step_default(request)


@app.get("/openenv/state")
@app.get("/openenv/state/")
def openenv_state() -> dict[str, object]:
    return get_state_default()


@app.get("/openenv/grade")
@app.get("/openenv/grade/")
def openenv_grade() -> dict[str, object]:
    return grade_default()
