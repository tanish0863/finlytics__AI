"""Deterministic graders for the Finlytics OpenEnv tasks."""

from __future__ import annotations

from math import fsum
from typing import Iterable

from .models import FinancialAction, FinancialState, TaskGrade
from .scenarios import TaskSpec


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _average(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return fsum(values) / len(values)


def _common_metrics(task: TaskSpec, state: FinancialState) -> dict[str, float]:
    total_income = max(task.total_income, 1.0)
    emergency_progress = _clamp(state.emergency_fund / max(task.target_emergency_fund, 1.0))
    debt_reduction = _clamp((task.starting_debt - state.debt_balance) / max(task.starting_debt, 1.0))
    investment_progress = _clamp(state.investment_balance / max(task.target_investment_balance, 1.0))
    solvency = _clamp(1.0 - (state.cumulative_shortfall / total_income))
    budget_alignment = _clamp(1.0 - (state.cumulative_alignment_penalty / max(state.step_index, 1)))
    reward_efficiency = _clamp(state.cumulative_reward / max(state.step_index, 1))
    shock_resilience = _clamp(1.0 - (state.cumulative_risk_penalty / max(state.step_index, 1)))
    consistency = _clamp(1.0 - (state.action_change_penalty / max(state.step_index, 1)))

    return {
        "solvency": solvency,
        "emergency_progress": emergency_progress,
        "debt_reduction": debt_reduction,
        "investment_progress": investment_progress,
        "budget_alignment": budget_alignment,
        "reward_efficiency": reward_efficiency,
        "shock_resilience": shock_resilience,
        "consistency": consistency,
    }


def grade_task(task: TaskSpec, state: FinancialState, history: list[FinancialAction]) -> TaskGrade:
    """Return a deterministic score between 0.0 and 1.0 for the completed episode."""

    metrics = _common_metrics(task, state)
    score = 0.0
    for metric_name, weight in task.grader_weights.items():
        score += metrics.get(metric_name, 0.0) * weight

    score = _clamp(score)
    passed = score >= task.grader_threshold and state.cumulative_shortfall == 0.0

    if task.difficulty == "easy":
        notes = "Budget discipline and emergency-fund growth were the primary success criteria."
    elif task.difficulty == "medium":
        notes = "Debt reduction, emergency savings, and disciplined cash allocation were scored together."
    else:
        notes = "The hard task rewards stability through shocks, consistency, and controlled downside." 

    return TaskGrade(
        task_id=task.task_id,
        task_title=task.title,
        difficulty=task.difficulty,
        score=round(score, 4),
        passed=passed,
        breakdown={key: round(value, 4) for key, value in metrics.items()},
        notes=notes,
    )


def action_alignment(task: TaskSpec, action: FinancialAction) -> float:
    """Measure how well the agent action aligns with the task's recommended mix."""

    target_spend = task.recommended_spend_cap
    target_save = task.recommended_save_floor
    target_invest = task.recommended_invest_floor
    target_debt = task.recommended_debt_floor

    distance = (
        abs(action.spend_pct - target_spend)
        + abs(action.save_pct - target_save)
        + abs(action.invest_pct - target_invest)
        + abs(action.debt_payment_pct - target_debt)
    )
    return _clamp(1.0 - distance / 2.0)


def average_action_change(history: list[FinancialAction]) -> float:
    """Compute a normalized measure of how much the action mix changes over time."""

    if len(history) < 2:
        return 0.0

    deltas = []
    previous = history[0]
    for current in history[1:]:
        deltas.append(
            abs(current.spend_pct - previous.spend_pct)
            + abs(current.save_pct - previous.save_pct)
            + abs(current.invest_pct - previous.invest_pct)
            + abs(current.debt_payment_pct - previous.debt_payment_pct)
        )
        previous = current
    return _average(deltas) / 2.0
