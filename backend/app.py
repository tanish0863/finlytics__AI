"""
Gradio entrypoint for the Finlytics AI financial decision environment.

The UI simulates a reinforcement-learning style loop where the environment is
the user's financial data, the agent is the decision layer, and the actions are
spend, save, and invest recommendations.
"""
from __future__ import annotations

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "Gradio is required to run this project. Install backend/requirements.txt first."
    ) from exc

try:
    from .services.financial_decision_service import (
        analyze_financial_environment,
        format_currency,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from services.financial_decision_service import (
        analyze_financial_environment,
        format_currency,
    )


def generate_financial_decision(income: float | None, expenses: float | None):
    """Convert raw user inputs into a clean decision summary for the UI."""
    normalized_income = float(income or 0.0)
    normalized_expenses = float(expenses or 0.0)

    try:
        result = analyze_financial_environment(normalized_income, normalized_expenses)
    except ValueError as error:
        return f"Invalid input: {error}", {"error": str(error)}

    summary = (
        f"Savings: {format_currency(result.savings)}. "
        f"Suggested Investment: {format_currency(result.suggested_investment)}. "
        f"Risk Level: {result.risk_level}."
    )

    structured_output = {
        "environment": {
            "income": format_currency(result.income),
            "expenses": format_currency(result.expenses),
            "savings": format_currency(result.savings),
        },
        "agent": {
            "recommended_action": result.recommended_action,
            "investment_suggestion": format_currency(result.suggested_investment),
            "risk_level": result.risk_level,
        },
        "reward_mechanism": {
            "profit_gain": format_currency(result.reward_breakdown["profit_gain"]),
            "risk_penalty": round(result.reward_breakdown["risk_penalty"], 2),
            "overspending_penalty": format_currency(
                result.reward_breakdown["overspending_penalty"]
            ),
            "reward": round(result.reward, 2),
        },
        "advice": result.advice,
    }

    return summary, structured_output


demo = gr.Interface(
    fn=generate_financial_decision,
    inputs=[
        gr.Number(label="Income (₹)", value=50000),
        gr.Number(label="Expenses (₹)", value=35000),
    ],
    outputs=[
        gr.Textbox(label="Decision Summary"),
        gr.JSON(label="Structured Output"),
    ],
    title="Finlytics AI - Reinforcement Learning Financial Environment",
    description=(
        "This demo frames personal finance as an AI environment. "
        "The environment is the user's income and expenses, the agent is the "
        "decision system, and the available actions are to spend, save, or "
        "invest. The output simulates a reward function so judges can see how "
        "the agent balances profit and financial stability."
    ),
    examples=[
        [50000, 35000],
        [100000, 65000],
        [80000, 30000],
    ],
)


app = demo


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
