"""Task scenarios for the Finlytics OpenEnv submission."""

from __future__ import annotations

from typing import Dict, List

from .models import MonthlySnapshot, TaskDifficulty, TaskGrade
from pydantic import BaseModel, ConfigDict, Field


class TaskSpec(BaseModel):
    """Scenario specification for one concrete environment task."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str
    title: str
    difficulty: TaskDifficulty
    objective: str
    episode_length: int = Field(gt=0)
    starting_debt: float = Field(ge=0)
    target_emergency_fund: float = Field(ge=0)
    target_investment_balance: float = Field(ge=0)
    target_debt_balance: float = Field(ge=0)
    recommended_spend_cap: float = Field(ge=0, le=1)
    recommended_save_floor: float = Field(ge=0, le=1)
    recommended_invest_floor: float = Field(ge=0, le=1)
    recommended_debt_floor: float = Field(ge=0, le=1)
    monthly_schedule: List[MonthlySnapshot]
    grader_threshold: float = Field(ge=0, le=1)
    grader_weights: dict[str, float]

    @property
    def total_income(self) -> float:
        return sum(month.income for month in self.monthly_schedule)

    @property
    def total_fixed_expenses(self) -> float:
        return sum(month.fixed_expenses for month in self.monthly_schedule)

    @property
    def total_variable_expenses(self) -> float:
        return sum(month.variable_expenses for month in self.monthly_schedule)


TASKS: Dict[str, TaskSpec] = {
    "budget_easy": TaskSpec(
        task_id="budget_easy",
        title="Monthly Budget Stabilization",
        difficulty="easy",
        objective=(
            "Keep the household budget positive, build an emergency cushion, and avoid overspending "
            "on discretionary cash flow."
        ),
        episode_length=2,
        starting_debt=0.0,
        target_emergency_fund=24000.0,
        target_investment_balance=8000.0,
        target_debt_balance=0.0,
        recommended_spend_cap=0.35,
        recommended_save_floor=0.30,
        recommended_invest_floor=0.10,
        recommended_debt_floor=0.05,
        monthly_schedule=[
            MonthlySnapshot(
                month_label="Month 1",
                income=120000.0,
                fixed_expenses=72000.0,
                variable_expenses=12000.0,
                market_shock="stable salary",
            ),
            MonthlySnapshot(
                month_label="Month 2",
                income=120000.0,
                fixed_expenses=72000.0,
                variable_expenses=13000.0,
                market_shock="bonus opportunity",
            ),
        ],
        grader_threshold=0.75,
        grader_weights={
            "solvency": 0.35,
            "emergency_progress": 0.30,
            "budget_alignment": 0.20,
            "reward_efficiency": 0.15,
        },
    ),
    "debt_medium": TaskSpec(
        task_id="debt_medium",
        title="Debt Paydown With Safety Buffer",
        difficulty="medium",
        objective=(
            "Reduce high-interest debt while still funding an emergency fund and keeping the budget "
            "resilient to one month of pressure."
        ),
        episode_length=3,
        starting_debt=180000.0,
        target_emergency_fund=60000.0,
        target_investment_balance=15000.0,
        target_debt_balance=90000.0,
        recommended_spend_cap=0.30,
        recommended_save_floor=0.20,
        recommended_invest_floor=0.10,
        recommended_debt_floor=0.25,
        monthly_schedule=[
            MonthlySnapshot(
                month_label="Month 1",
                income=110000.0,
                fixed_expenses=70000.0,
                variable_expenses=10000.0,
                market_shock="credit card interest",
            ),
            MonthlySnapshot(
                month_label="Month 2",
                income=102000.0,
                fixed_expenses=70000.0,
                variable_expenses=11000.0,
                market_shock="medical bill",
            ),
            MonthlySnapshot(
                month_label="Month 3",
                income=108000.0,
                fixed_expenses=70000.0,
                variable_expenses=9000.0,
                market_shock="steady recovery",
            ),
        ],
        grader_threshold=0.68,
        grader_weights={
            "solvency": 0.25,
            "debt_reduction": 0.30,
            "emergency_progress": 0.20,
            "budget_alignment": 0.15,
            "reward_efficiency": 0.10,
        },
    ),
    "volatile_hard": TaskSpec(
        task_id="volatile_hard",
        title="Volatile Cash-Flow Planning",
        difficulty="hard",
        objective=(
            "Survive income volatility, absorb shocks, and still preserve enough liquidity to avoid "
            "a cash crisis while paying down debt."
        ),
        episode_length=4,
        starting_debt=240000.0,
        target_emergency_fund=90000.0,
        target_investment_balance=22000.0,
        target_debt_balance=140000.0,
        recommended_spend_cap=0.25,
        recommended_save_floor=0.15,
        recommended_invest_floor=0.10,
        recommended_debt_floor=0.35,
        monthly_schedule=[
            MonthlySnapshot(
                month_label="Month 1",
                income=95000.0,
                fixed_expenses=69000.0,
                variable_expenses=12000.0,
                market_shock="tax catch-up",
            ),
            MonthlySnapshot(
                month_label="Month 2",
                income=76000.0,
                fixed_expenses=69000.0,
                variable_expenses=15000.0,
                market_shock="income dip",
            ),
            MonthlySnapshot(
                month_label="Month 3",
                income=118000.0,
                fixed_expenses=70000.0,
                variable_expenses=13000.0,
                market_shock="equipment repair",
            ),
            MonthlySnapshot(
                month_label="Month 4",
                income=86000.0,
                fixed_expenses=70000.0,
                variable_expenses=14000.0,
                market_shock="medical shock",
            ),
        ],
        grader_threshold=0.62,
        grader_weights={
            "solvency": 0.20,
            "shock_resilience": 0.25,
            "debt_reduction": 0.20,
            "consistency": 0.20,
            "budget_alignment": 0.15,
        },
    ),
}

TASK_IDS: List[str] = ["budget_easy", "debt_medium", "volatile_hard"]


def get_task(task_id: str) -> TaskSpec:
    """Return the configured task or raise a clear error."""

    try:
        return TASKS[task_id]
    except KeyError as error:
        raise ValueError(f"Unknown task_id '{task_id}'. Available tasks: {', '.join(TASK_IDS)}") from error
