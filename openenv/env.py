"""OpenEnv-compatible financial planning environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .grading import action_alignment, average_action_change, grade_task
from .models import FinancialAction, FinancialObservation, FinancialReward, FinancialState
from .scenarios import TaskSpec, get_task


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class _StepRecord:
    action: FinancialAction
    reward: FinancialReward


class FinancialPlanningEnv:
    """A real-world finance environment for agent training and evaluation.

    Environment: household/business cash flow, savings, debt, and investment balances.
    Agent: the policy that allocates disposable cash across spend/save/invest/debt.
    Actions: spending, saving, investing, and debt repayment percentages.
    Objective: maximize financial stability while preserving positive trajectory.
    """

    def __init__(self, task_id: str = "budget_easy") -> None:
        self.task: TaskSpec = get_task(task_id)
        self._state: FinancialState | None = None
        self._done = False
        self._history: list[_StepRecord] = []

    def reset(self) -> FinancialObservation:
        """Reset the episode and return the initial observation."""

        first_month = self.task.monthly_schedule[0]
        self._state = FinancialState(
            task_id=self.task.task_id,
            task_title=self.task.title,
            difficulty=self.task.difficulty,
            step_index=0,
            episode_length=self.task.episode_length,
            income=first_month.income,
            fixed_expenses=first_month.fixed_expenses,
            variable_expenses=first_month.variable_expenses,
            disposable_surplus=first_month.income - first_month.fixed_expenses - first_month.variable_expenses,
            savings_balance=0.0,
            emergency_fund=0.0,
            investment_balance=0.0,
            debt_balance=self.task.starting_debt,
            starting_debt=self.task.starting_debt,
            target_emergency_fund=self.task.target_emergency_fund,
            target_investment_balance=self.task.target_investment_balance,
            target_debt_balance=self.task.target_debt_balance,
            cumulative_reward=0.0,
            cumulative_shortfall=0.0,
            cumulative_overspend=0.0,
            cumulative_risk_penalty=0.0,
            cumulative_alignment_penalty=0.0,
            action_change_penalty=0.0,
            last_market_shock=first_month.market_shock,
            objective=self.task.objective,
            done=False,
            last_action=None,
        )
        self._done = False
        self._history = []
        return self._make_observation()

    def state(self) -> FinancialState:
        """Return the current full internal state."""

        self._ensure_state()
        return self._state.model_copy(deep=True)

    def step(self, action: FinancialAction | dict[str, Any]) -> tuple[FinancialObservation, FinancialReward, bool, dict[str, Any]]:
        """Advance the episode by one month using the supplied action."""

        self._ensure_state()
        if self._done:
            raise RuntimeError("Episode has already finished. Call reset() before stepping again.")

        validated_action = action if isinstance(action, FinancialAction) else FinancialAction.model_validate(action)
        current_month = self.task.monthly_schedule[self._state.step_index]
        disposable_surplus = current_month.income - current_month.fixed_expenses - current_month.variable_expenses
        risk_ratio = (current_month.fixed_expenses + current_month.variable_expenses) / max(current_month.income, 1.0)

        # Update the current visible financial snapshot before we compute reward.
        self._state.income = current_month.income
        self._state.fixed_expenses = current_month.fixed_expenses
        self._state.variable_expenses = current_month.variable_expenses
        self._state.disposable_surplus = disposable_surplus
        self._state.last_market_shock = current_month.market_shock

        alignment_score = action_alignment(self.task, validated_action)
        previous_action = self._state.last_action
        if previous_action is None:
            action_change_penalty = 0.0
        else:
            action_change_penalty = (
                abs(validated_action.spend_pct - previous_action.spend_pct)
                + abs(validated_action.save_pct - previous_action.save_pct)
                + abs(validated_action.invest_pct - previous_action.invest_pct)
                + abs(validated_action.debt_payment_pct - previous_action.debt_payment_pct)
            ) / 2.0

        if disposable_surplus < 0:
            shortfall = abs(disposable_surplus)
            self._state.cumulative_shortfall += shortfall
            self._state.debt_balance += shortfall
            risk_penalty = _clamp(shortfall / max(current_month.income, 1.0))
            overspending_penalty = _clamp(risk_ratio - 0.70)
            progress_bonus = 0.0
            stability_bonus = _clamp(1.0 - risk_ratio)
            reward_score = _clamp(0.15 * stability_bonus - 0.55 * risk_penalty - 0.30 * overspending_penalty)
            explanation = "Cash flow was negative this month, so the environment penalized the shortfall."
            spend_amount = save_amount = invest_amount = debt_payment = 0.0
        else:
            spend_amount = disposable_surplus * validated_action.spend_pct
            save_amount = disposable_surplus * validated_action.save_pct
            invest_amount = disposable_surplus * validated_action.invest_pct
            debt_payment = min(disposable_surplus * validated_action.debt_payment_pct, self._state.debt_balance)

            emergency_gap = max(self.task.target_emergency_fund - self._state.emergency_fund, 0.0)
            emergency_addition = min(save_amount * 0.70, emergency_gap)
            savings_addition = save_amount - emergency_addition

            self._state.emergency_fund += emergency_addition
            self._state.savings_balance += savings_addition
            self._state.investment_balance += invest_amount
            self._state.debt_balance = max(0.0, self._state.debt_balance - debt_payment)

            overspending_penalty = _clamp(max(validated_action.spend_pct - self.task.recommended_spend_cap, 0.0))
            risk_penalty = _clamp(max(risk_ratio - 0.70, 0.0))
            progress_bonus = _clamp(
                0.4 * _clamp(self._state.emergency_fund / max(self.task.target_emergency_fund, 1.0))
                + 0.3 * _clamp((self.task.starting_debt - self._state.debt_balance) / max(self.task.starting_debt, 1.0))
                + 0.3 * _clamp(self._state.investment_balance / max(self.task.target_investment_balance, 1.0))
            )
            stability_bonus = _clamp(1.0 - risk_ratio)
            reward_score = _clamp(
                0.30 * alignment_score
                + 0.30 * progress_bonus
                + 0.20 * stability_bonus
                + 0.20 * _clamp(1.0 - risk_penalty - overspending_penalty)
                - 0.10 * action_change_penalty
            )
            explanation = "Positive cash flow allowed the agent to allocate surplus toward savings, investing, and debt repayment."

        self._state.cumulative_alignment_penalty += 1.0 - alignment_score
        self._state.cumulative_risk_penalty += risk_penalty
        self._state.cumulative_overspend += overspending_penalty
        self._state.action_change_penalty += action_change_penalty
        self._state.cumulative_reward += reward_score
        self._state.last_action = validated_action
        self._state.step_index += 1
        self._done = self._state.step_index >= self.task.episode_length
        self._state.done = self._done

        reward = FinancialReward(
            score=round(reward_score, 4),
            profit_gain=round(max(disposable_surplus, 0.0), 2),
            risk_penalty=round(risk_penalty, 4),
            overspending_penalty=round(overspending_penalty, 4),
            progress_bonus=round(progress_bonus, 4),
            stability_bonus=round(stability_bonus, 4),
            explanation=explanation,
        )

        self._history.append(_StepRecord(action=validated_action, reward=reward))

        if not self._done:
            next_month = self.task.monthly_schedule[self._state.step_index]
            self._state.income = next_month.income
            self._state.fixed_expenses = next_month.fixed_expenses
            self._state.variable_expenses = next_month.variable_expenses
            self._state.disposable_surplus = (
                next_month.income - next_month.fixed_expenses - next_month.variable_expenses
            )
            self._state.last_market_shock = next_month.market_shock

        observation = self._make_observation()
        info = {
            "task_id": self.task.task_id,
            "task_title": self.task.title,
            "reward": reward.model_dump(),
            "task_progress": round(self._compute_task_progress(), 4),
            "grader_score": round(self.grade().score, 4) if self._done else None,
            "done": self._done,
        }
        return observation, reward, self._done, info

    def grade(self):
        """Grade the completed episode with the deterministic task grader."""

        self._ensure_state()
        return grade_task(self.task, self._state, [record.action for record in self._history])

    def _compute_task_progress(self) -> float:
        self._ensure_state()
        emergency_progress = _clamp(self._state.emergency_fund / max(self.task.target_emergency_fund, 1.0))
        debt_progress = _clamp((self.task.starting_debt - self._state.debt_balance) / max(self.task.starting_debt, 1.0))
        investment_progress = _clamp(self._state.investment_balance / max(self.task.target_investment_balance, 1.0))
        solvency = _clamp(1.0 - (self._state.cumulative_shortfall / max(self.task.total_income, 1.0)))
        return _clamp(0.35 * emergency_progress + 0.25 * debt_progress + 0.20 * investment_progress + 0.20 * solvency)

    def _make_observation(self) -> FinancialObservation:
        self._ensure_state()
        if self._done:
            current_index = min(self._state.step_index - 1, self.task.episode_length - 1)
        else:
            current_index = self._state.step_index
        current_month = self.task.monthly_schedule[current_index]
        disposable_surplus = current_month.income - current_month.fixed_expenses - current_month.variable_expenses
        risk_ratio = (current_month.fixed_expenses + current_month.variable_expenses) / max(current_month.income, 1.0)
        return FinancialObservation(
            task_id=self.task.task_id,
            task_title=self.task.title,
            difficulty=self.task.difficulty,
            step_index=self._state.step_index,
            episode_length=self.task.episode_length,
            income=current_month.income,
            fixed_expenses=current_month.fixed_expenses,
            variable_expenses=current_month.variable_expenses,
            disposable_surplus=disposable_surplus,
            savings_balance=self._state.savings_balance,
            emergency_fund=self._state.emergency_fund,
            investment_balance=self._state.investment_balance,
            debt_balance=self._state.debt_balance,
            risk_ratio=round(risk_ratio, 4),
            target_emergency_fund=self.task.target_emergency_fund,
            target_investment_balance=self.task.target_investment_balance,
            target_debt_balance=self.task.target_debt_balance,
            market_shock=current_month.market_shock,
            objective=self.task.objective,
            recommended_spend_cap=self.task.recommended_spend_cap,
            recommended_save_floor=self.task.recommended_save_floor,
            recommended_invest_floor=self.task.recommended_invest_floor,
            recommended_debt_floor=self.task.recommended_debt_floor,
        )

    def _ensure_state(self) -> None:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
