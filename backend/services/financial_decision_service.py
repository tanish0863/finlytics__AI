"""
Financial decision service for the Finlytics AI environment.

This module turns income and expense inputs into a simplified reinforcement
learning style decision output. The environment is the user's financial data,
the agent is the recommendation logic, and the reward function simulates how
well the agent balances profit with risk.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


def format_currency(value: float) -> str:
    """Format numeric values as INR currency for the UI and summaries."""
    return f"₹{value:,.2f}"


@dataclass(frozen=True)
class FinancialDecisionResult:
    """Structured response returned by the decision service."""

    income: float
    expenses: float
    savings: float
    suggested_investment: float
    risk_level: str
    reward: float
    reward_breakdown: Dict[str, float]
    recommended_action: str
    advice: str


def _determine_risk_level(income: float, expenses: float, savings: float) -> str:
    """Map expense pressure to a simple risk band."""
    if income <= 0:
        return "High"

    expense_ratio = expenses / income
    if expense_ratio > 0.70 or savings < 0:
        return "High"
    if expense_ratio > 0.50:
        return "Medium"
    return "Low"


def _build_advice(risk_level: str, savings: float, suggested_investment: float) -> str:
    """Generate human-readable guidance for the agent output."""
    if risk_level == "High":
        return (
            "Focus on reducing discretionary spending, build a cash buffer, "
            "and avoid aggressive investing until monthly outflows fall below 70% of income."
        )
    if risk_level == "Medium":
        return (
            "Keep spending under control, save consistently, and prefer conservative "
            "investment options while maintaining an emergency buffer."
        )
    return (
        f"Healthy surplus detected. Save {format_currency(savings)} and consider "
        f"investing {format_currency(suggested_investment)} in low-risk instruments."
    )


def analyze_financial_environment(income: float, expenses: float) -> FinancialDecisionResult:
    """
    Convert income and expenses into a simplified RL-style financial decision.

    Reward calculation:
    Reward = Profit_Gain - Risk_Penalty - Overspending_Penalty
    - Profit_Gain rewards positive savings.
    - Risk_Penalty increases when the expense ratio becomes unhealthy.
    - Overspending_Penalty applies when expenses cross the 70% income threshold.
    """
    if income < 0:
        raise ValueError("Income cannot be negative.")
    if expenses < 0:
        raise ValueError("Expenses cannot be negative.")

    savings = income - expenses
    positive_savings = max(savings, 0.0)

    # The agent recommends investing 30% of the available savings.
    suggested_investment = round(positive_savings * 0.30, 2)

    risk_level = _determine_risk_level(income, expenses, savings)

    # Profit gain is the amount of money preserved after expenses.
    profit_gain = positive_savings

    # Risk penalty is a fixed signal that grows as the environment becomes less stable.
    if risk_level == "High":
        risk_penalty = 25.0
    elif risk_level == "Medium":
        risk_penalty = 10.0
    else:
        risk_penalty = 0.0

    # Overspending penalty activates only when expenses exceed 70% of income.
    overspending_threshold = income * 0.70
    overspending_penalty = max(expenses - overspending_threshold, 0.0)

    reward = profit_gain - risk_penalty - overspending_penalty

    if savings <= 0:
        recommended_action = "Reduce spending and rebuild savings before investing."
    elif risk_level == "Low":
        recommended_action = "Save surplus cash and invest a disciplined 30% of savings."
    elif risk_level == "Medium":
        recommended_action = "Balance saving with cautious investment and control costs."
    else:
        recommended_action = "Prioritize expense reduction and cash preservation over new investments."

    advice = _build_advice(risk_level, savings, suggested_investment)

    return FinancialDecisionResult(
        income=round(income, 2),
        expenses=round(expenses, 2),
        savings=round(savings, 2),
        suggested_investment=round(suggested_investment, 2),
        risk_level=risk_level,
        reward=round(reward, 2),
        reward_breakdown={
            "profit_gain": round(profit_gain, 2),
            "risk_penalty": round(risk_penalty, 2),
            "overspending_penalty": round(overspending_penalty, 2),
        },
        recommended_action=recommended_action,
        advice=advice,
    )


def financial_decision_to_dict(result: FinancialDecisionResult) -> Dict[str, object]:
    """Convert the result dataclass into a JSON-friendly dictionary."""
    return asdict(result)