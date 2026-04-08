"""Tests for the Finlytics OpenEnv environment."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openenv.baseline import _heuristic_action
from openenv.env import FinancialPlanningEnv
from openenv.scenarios import TASK_IDS


def test_reset_returns_valid_observation() -> None:
    env = FinancialPlanningEnv(task_id="budget_easy")
    observation = env.reset()

    assert observation.task_id == "budget_easy"
    assert observation.step_index == 0
    assert observation.episode_length == 2
    assert observation.disposable_surplus > 0


def test_step_returns_reward_and_state() -> None:
    env = FinancialPlanningEnv(task_id="debt_medium")
    observation = env.reset()
    action = _heuristic_action(observation)
    next_observation, reward, done, info = env.step(action)

    assert 0.0 <= reward.score <= 1.0
    assert next_observation.task_id == "debt_medium"
    assert isinstance(done, bool)
    assert "task_progress" in info
    assert env.state().task_id == "debt_medium"


def test_full_episode_grades_between_zero_and_one() -> None:
    for task_id in TASK_IDS:
        env = FinancialPlanningEnv(task_id=task_id)
        observation = env.reset()
        done = False

        while not done:
            observation, reward, done, info = env.step(_heuristic_action(observation))

        grade = env.grade()
        assert 0.0 <= grade.score <= 1.0
        assert grade.task_id == task_id
