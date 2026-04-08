"""Local validation helper for the Finlytics OpenEnv environment."""

from __future__ import annotations

from pathlib import Path

import yaml

from .baseline import _heuristic_action
from .env import FinancialPlanningEnv
from .scenarios import TASK_IDS, TASKS


def validate_environment() -> dict[str, object]:
    """Smoke-test the spec, the API contract, and the reward bounds."""

    spec_path = Path(__file__).resolve().parents[1] / "openenv.yaml"
    with spec_path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    task_results = []
    for task_id in TASK_IDS:
        env = FinancialPlanningEnv(task_id=task_id)
        observation = env.reset()
        done = False
        last_reward = None

        while not done:
            action = _heuristic_action(observation)
            observation, reward, done, info = env.step(action)
            assert 0.0 <= reward.score <= 1.0
            last_reward = reward

        grade = env.grade()
        task_results.append(
            {
                "task_id": task_id,
                "score": grade.score,
                "passed": grade.passed,
                "final_reward": last_reward.score if last_reward else None,
            }
        )

    return {
        "spec_name": spec.get("name"),
        "spec_version": spec.get("version"),
        "task_count": len(TASKS),
        "results": task_results,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(validate_environment(), indent=2))