# Finlytics AI Backend

This backend now powers the Gradio-based financial decision environment used by the hackathon demo.

## What It Does

- Accepts income and expense inputs
- Computes savings and investment suggestions
- Simulates a reinforcement-learning style reward signal
- Returns a structured decision response for the UI

## Run

```bash
cd backend
pip install -r requirements.txt
python app.py
```

## Main Files

- `app.py`: Gradio entrypoint and UI wiring
- `services/financial_decision_service.py`: Environment, agent, action, and reward logic
- `services/scoring_engine.py`: Existing scoring engine kept for the broader project

## Decision Logic

The service uses the following simplified policy:

- Savings = Income - Expenses
- Suggested investment = 30% of positive savings
- Risk level becomes High when expenses exceed 70% of income
- Reward = Profit_Gain - Risk_Penalty - Overspending_Penalty

## Output Style

The UI returns readable summary text such as:

`Savings: ₹20,000.00. Suggested Investment: ₹6,000.00. Risk Level: Low.`
