"""Baseline inference script for the Finlytics OpenEnv submission."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable

from .env import FinancialPlanningEnv
from .models import FinancialAction, FinancialObservation
from .scenarios import TASK_IDS, get_task


@dataclass(frozen=True)
class BaselineResult:
    task_id: str
    task_title: str
    score: float
    steps: int


def _heuristic_action(observation: FinancialObservation) -> FinancialAction:
    """Deterministic fallback policy used for offline smoke tests."""

    if observation.disposable_surplus <= 0:
        return FinancialAction(
            spend_pct=0.05,
            save_pct=0.15,
            invest_pct=0.00,
            debt_payment_pct=0.80,
            rationale="Negative cash flow: prioritize debt containment and preserve liquidity.",
        )

    if observation.debt_balance > observation.target_debt_balance:
        debt_share = min(0.45, observation.recommended_debt_floor + 0.15)
        invest_share = min(0.20, observation.recommended_invest_floor + 0.05)
        save_share = min(0.25, observation.recommended_save_floor)
        spend_share = max(0.0, 1.0 - debt_share - invest_share - save_share)
    elif observation.emergency_fund < observation.target_emergency_fund:
        save_share = min(0.40, observation.recommended_save_floor + 0.10)
        debt_share = observation.recommended_debt_floor
        invest_share = observation.recommended_invest_floor
        spend_share = max(0.0, 1.0 - save_share - debt_share - invest_share)
    else:
        spend_share = observation.recommended_spend_cap
        save_share = max(observation.recommended_save_floor - 0.05, 0.10)
        invest_share = min(0.30, observation.recommended_invest_floor + 0.10)
        debt_share = max(0.0, 1.0 - spend_share - save_share - invest_share)

    total = spend_share + save_share + invest_share + debt_share
    if abs(total - 1.0) > 1e-9:
        spend_share /= total
        save_share /= total
        invest_share /= total
        debt_share /= total

    return FinancialAction(
        spend_pct=round(spend_share, 4),
        save_pct=round(save_share, 4),
        invest_pct=round(invest_share, 4),
        debt_payment_pct=round(debt_share, 4),
        rationale="Deterministic fallback policy for reproducible offline scoring.",
    )


def _openai_action(
    observation: FinancialObservation,
    model_name: str,
    client: object,
) -> FinancialAction:
    """Ask an OpenAI model to produce a structured allocation decision."""

    from openai import OpenAI  # Imported lazily so offline validation still works.

    assert isinstance(client, OpenAI)
    prompt = f"""
You are managing a real monthly financial plan.

Return ONLY valid JSON with these keys:
- spend_pct
- save_pct
- invest_pct
- debt_payment_pct
- rationale

The percentages must be decimals that sum to 1.0.
Prefer lower spend, higher debt payment, and savings when debt or emergency-fund targets are not met.

Observation:
{json.dumps(observation.model_dump(), indent=2)}
""".strip()

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": "You allocate personal-finance actions as strict JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    data = _extract_json_object(content)
    return FinancialAction.model_validate(data)


def _extract_json_object(content: str) -> dict[str, object]:
    """Extract the first JSON object from an OpenAI response."""

    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])
    return json.loads(stripped)


def _resolve_action_provider(provider: str) -> Callable[[FinancialObservation], FinancialAction]:
    if provider == "heuristic":
        return _heuristic_action

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _heuristic_action

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = __import__("openai").OpenAI(api_key=api_key)

    def _provider(observation: FinancialObservation) -> FinancialAction:
        return _openai_action(observation, model_name=model_name, client=client)

    return _provider


def run_baseline(provider: str = "openai") -> list[BaselineResult]:
    """Run the baseline policy across all three tasks and return reproducible scores."""

    action_provider = _resolve_action_provider(provider)
    results: list[BaselineResult] = []

    for task_id in TASK_IDS:
        task = get_task(task_id)
        env = FinancialPlanningEnv(task_id=task_id)
        observation = env.reset()
        done = False
        steps = 0

        while not done:
            action = action_provider(observation)
            observation, _, done, _ = env.step(action)
            steps += 1

        grade = env.grade()
        results.append(
            BaselineResult(
                task_id=task.task_id,
                task_title=task.title,
                score=round(grade.score, 4),
                steps=steps,
            )
        )

    return results


def _print_results(results: Iterable[BaselineResult]) -> None:
    scores = list(results)
    total = 0.0
    for result in scores:
        total += result.score
        print(f"{result.task_id}: {result.task_title} -> score={result.score:.4f} steps={result.steps}")
    if scores:
        print(f"mean_score={total / len(scores):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Finlytics OpenEnv baseline.")
    parser.add_argument(
        "--provider",
        choices=["openai", "heuristic"],
        default=os.getenv("OPENENV_BASELINE_PROVIDER", "openai"),
        help="Action provider to use. Defaults to OpenAI when an API key is available.",
    )
    args = parser.parse_args()
    _print_results(run_baseline(provider=args.provider))


if __name__ == "__main__":
    main()