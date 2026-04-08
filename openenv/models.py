"""Typed OpenEnv models for the Finlytics financial planning environment."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskDifficulty = Literal["easy", "medium", "hard"]


class OpenEnvModel(BaseModel):
    """Base model with strict validation for the OpenEnv package."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MonthlySnapshot(OpenEnvModel):
    """A single month of financial inputs used by a task scenario."""

    month_label: str
    income: float = Field(ge=0)
    fixed_expenses: float = Field(ge=0)
    variable_expenses: float = Field(ge=0)
    market_shock: str = "none"


class FinancialAction(OpenEnvModel):
    """Agent action that allocates disposable cash across realistic priorities."""

    spend_pct: float = Field(ge=0, le=1)
    save_pct: float = Field(ge=0, le=1)
    invest_pct: float = Field(ge=0, le=1)
    debt_payment_pct: float = Field(ge=0, le=1)
    rationale: str = ""

    @model_validator(mode="after")
    def validate_allocation(self) -> "FinancialAction":
        total_allocation = self.spend_pct + self.save_pct + self.invest_pct + self.debt_payment_pct
        if abs(total_allocation - 1.0) > 0.03:
            raise ValueError("Action allocations must sum to 1.0 within a 3% tolerance.")
        return self


class FinancialObservation(OpenEnvModel):
    """Observation visible to the agent at each step."""

    task_id: str
    task_title: str
    difficulty: TaskDifficulty
    step_index: int = Field(ge=0)
    episode_length: int = Field(gt=0)
    income: float = Field(ge=0)
    fixed_expenses: float = Field(ge=0)
    variable_expenses: float = Field(ge=0)
    disposable_surplus: float
    savings_balance: float = Field(ge=0)
    emergency_fund: float = Field(ge=0)
    investment_balance: float = Field(ge=0)
    debt_balance: float = Field(ge=0)
    risk_ratio: float = Field(ge=0)
    target_emergency_fund: float = Field(ge=0)
    target_investment_balance: float = Field(ge=0)
    target_debt_balance: float = Field(ge=0)
    market_shock: str
    objective: str
    recommended_spend_cap: float = Field(ge=0, le=1)
    recommended_save_floor: float = Field(ge=0, le=1)
    recommended_invest_floor: float = Field(ge=0, le=1)
    recommended_debt_floor: float = Field(ge=0, le=1)


class FinancialReward(OpenEnvModel):
    """Dense reward signal returned by each environment step."""

    score: float = Field(ge=0, le=1)
    profit_gain: float = Field(ge=0)
    risk_penalty: float = Field(ge=0)
    overspending_penalty: float = Field(ge=0)
    progress_bonus: float = Field(ge=0)
    stability_bonus: float = Field(ge=0)
    explanation: str


class FinancialState(OpenEnvModel):
    """Full internal state snapshot exposed through state()."""

    task_id: str
    task_title: str
    difficulty: TaskDifficulty
    step_index: int = Field(ge=0)
    episode_length: int = Field(gt=0)
    income: float = Field(ge=0)
    fixed_expenses: float = Field(ge=0)
    variable_expenses: float = Field(ge=0)
    disposable_surplus: float
    savings_balance: float = Field(ge=0)
    emergency_fund: float = Field(ge=0)
    investment_balance: float = Field(ge=0)
    debt_balance: float = Field(ge=0)
    starting_debt: float = Field(ge=0)
    target_emergency_fund: float = Field(ge=0)
    target_investment_balance: float = Field(ge=0)
    target_debt_balance: float = Field(ge=0)
    cumulative_reward: float
    cumulative_shortfall: float = Field(ge=0)
    cumulative_overspend: float = Field(ge=0)
    cumulative_risk_penalty: float = Field(ge=0)
    cumulative_alignment_penalty: float = Field(ge=0)
    action_change_penalty: float = Field(ge=0)
    last_market_shock: str
    objective: str
    done: bool
    last_action: FinancialAction | None = None


class TaskGrade(OpenEnvModel):
    """Deterministic grade returned by the task-specific grader."""

    task_id: str
    task_title: str
    difficulty: TaskDifficulty
    score: float = Field(ge=0, le=1)
    passed: bool
    breakdown: dict[str, float]
    notes: str
