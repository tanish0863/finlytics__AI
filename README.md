# 🏦 Finlytics AI — OpenEnv RL Environment
### MSME Loan Credit Decision Environment for RL Agent Training

> **Meta PyTorch OpenEnv Hackathon × Scaler School of Technology, Bengaluru**
> Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — Meta's open-source RL environment framework

[![openenv-core](https://img.shields.io/badge/openenv--core-compatible-blue)](https://pypi.org/project/openenv-core/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-compatible-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 What Is This Environment?

**Finlytics AI** is an OpenEnv-compliant reinforcement learning environment where an AI agent learns to make **MSME loan credit decisions** — deciding whether to approve, reject, or restructure loan applications, and recommending appropriate amounts and tenures.

The agent interacts with a simulated credit evaluation environment containing realistic Indian MSME financial profiles. It receives structured observations (GST data, bank statement metrics, ITR signals), takes credit decision actions, and earns rewards based on decision quality measured against ground-truth risk labels produced by a calibrated XGBoost model.

This environment is fully compatible with any RL framework that supports the OpenEnv spec: **TRL, torchforge, SkyRL, Unsloth, ART, and Oumi**.

---

## 🌍 Why This Environment?

India has **63 million MSMEs** but **80% are credit-invisible** — rejected not because they are risky, but because evaluation is manual, slow, and inconsistent. Training an RL agent in this environment teaches it to:

- Make faster, consistent credit decisions across diverse MSME profiles
- Learn the relationship between financial signals and default probability
- Generalise across sectors: retail, manufacturing, services, agriculture
- Optimise for both portfolio quality and approval rate simultaneously

This is a **real-world, high-stakes sequential decision problem** — exactly the kind where RL agents can demonstrably outperform rule-based systems.

---

## ⚙️ OpenEnv Spec Compliance

| Spec Requirement | Status | File |
|---|---|---|
| `Environment` base class | ✅ | `server/environment.py` |
| `EnvClient` base class | ✅ | `client.py` |
| `reset()` → `Observation` | ✅ | Returns new MSME applicant profile |
| `step(action)` → `StepResult` | ✅ | Returns scored decision + reward |
| `state()` → `State` | ✅ | Episode ID, step count, metadata |
| Pydantic `Action` model | ✅ | `models.py` → `CreditDecisionAction` |
| Pydantic `Observation` model | ✅ | `models.py` → `ApplicantObservation` |
| Docker containerised server | ✅ | `server/Dockerfile` |
| FastAPI server | ✅ | `server/app.py` |
| `openenv.yaml` manifest | ✅ | `openenv.yaml` |
| `pyproject.toml` | ✅ | `pyproject.toml` |
| Programmatic grader | ✅ | `tests/test_environment.py` |
| Defined reward logic | ✅ | `server/environment.py` |
| `inference.py` RL agent loop | ✅ | `inference.py` |
| Hugging Face deployable | ✅ | `openenv push` compatible |

---

## 🏗️ Project Structure

```
finlytics__AI/
│
├── __init__.py                  # Exports: CreditDecisionAction, ApplicantObservation, FinlyticsEnv
├── models.py                    # Pydantic Action + Observation types
├── client.py                    # FinlyticsEnv(EnvClient) — OpenEnv client
├── openenv.yaml                 # Environment manifest for HF Hub
├── pyproject.toml               # Package deps + metadata
├── inference.py                 # Sample RL agent loop (required by spec)
│
├── server/
│   ├── app.py                   # FastAPI server entry point
│   ├── environment.py           # FinlyticsEnvironment(Environment) — core RL logic
│   ├── Dockerfile               # Container image
│   └── requirements.txt         # Docker dependencies
│
├── tests/
│   ├── __init__.py
│   └── test_environment.py      # Programmatic grader (hackathon evaluation)
│
├── examples/
│   └── quickstart.py            # Offline usage demo for judges
│
│   ── Existing Finlytics Production Stack ──
│
├── backend/                     # FastAPI scoring microservice + document extraction
├── frontend/                    # Next.js credit manager + borrower dashboard
├── ml/                          # XGBoost PD model, SHAP explainability, training scripts
├── mock_pdfs/                   # Sample GST, bank statement, ITR documents
├── docs/                        # Architecture docs
├── strict_mock_eval.py          # End-to-end scoring evaluation
└── validate_scoring_engine.py   # Scoring engine validation
```

---

## 🔄 Environment Design

### Action Space — `CreditDecisionAction`

```python
class CreditDecisionAction(Action):
    decision: Literal["approve", "reject", "restructure"]
    recommended_amount: float        # INR; 0 if decision is reject
    recommended_tenure_months: int   # 0 if decision is reject
    risk_band: Literal["low", "medium", "high"]
    reasoning: str                   # Agent's justification (used in LLM scoring)
```

### Observation Space — `ApplicantObservation`

```python
class ApplicantObservation(Observation):
    # Business identity
    gstin: str
    business_age_months: int
    sector: str                          # retail | manufacturing | services | agri

    # Revenue signals
    monthly_revenue: float               # INR
    revenue_trend_3m: float              # % change over 3 months
    gst_filing_consistency: float        # 0.0 – 1.0

    # Debt and obligations
    total_outstanding_debt: float
    monthly_emi_commitments: float
    debt_service_coverage_ratio: float   # DSCR

    # Cash flow (from bank statement)
    avg_monthly_balance: float
    balance_volatility: float            # std dev / mean

    # Tax compliance
    itr_filed: bool
    itr_years_filed: int

    # Episode tracking
    episode_id: str
    step: int
    max_steps: int
```

### Reward Function

Reward is computed by comparing the agent's decision against a **ground-truth risk label** generated by the XGBoost PD model:

```
reward = decision_accuracy          # correct approve/reject/restructure
       + calibration_bonus          # +0.3 for correct risk band
       + amount_appropriateness     # +0.2 if amount within valid range
       + tenure_appropriateness     # +0.1 if tenure within valid range
       - false_approval_penalty     # -1.5 for approving a high-risk applicant
       - false_rejection_penalty    # -0.5 for rejecting a low-risk applicant

reward ∈ [-2.0, 2.0]
```

The asymmetric penalty structure reflects real-world credit costs: a missed default is far more damaging than a missed approval.

---

## 🚀 Quick Start

### 1. Install

```bash
pip install openenv-core
pip install -e .
```

### 2. Run Server Locally (no Docker)

```bash
cd server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Run Server via Docker

```bash
docker build -t finlytics-env ./server
docker run -p 8000:8000 finlytics-env
```

### 4. Use the Environment (Async)

```python
import asyncio
from finlytics_env import CreditDecisionAction, FinlyticsEnv

async def main():
    async with FinlyticsEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        obs = result.observation
        print(f"Applicant GSTIN: {obs.gstin} | Revenue: ₹{obs.monthly_revenue:,.0f}/mo")
        print(f"DSCR: {obs.debt_service_coverage_ratio:.2f} | GST Consistency: {obs.gst_filing_consistency:.2f}")

        action = CreditDecisionAction(
            decision="approve",
            recommended_amount=obs.monthly_revenue * 3,
            recommended_tenure_months=24,
            risk_band="low",
            reasoning="Strong GST compliance, healthy DSCR, stable revenue trend."
        )
        result = await env.step(action)
        print(f"Reward: {result.reward:.3f} | Done: {result.done}")

asyncio.run(main())
```

### 5. Synchronous Usage

```python
from finlytics_env import CreditDecisionAction, FinlyticsEnv

with FinlyticsEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    action = CreditDecisionAction(
        decision="restructure",
        recommended_amount=250000,
        recommended_tenure_months=12,
        risk_band="medium",
        reasoning="Moderate risk — smaller amount recommended."
    )
    result = env.step(action)
    print(f"Reward: {result.reward}")
```

---

## 🧪 Programmatic Grader

The hackathon evaluates submissions using the programmatic grader in `tests/test_environment.py`.

```bash
# Start the server
docker build -t finlytics-env ./server && docker run -d -p 8000:8000 finlytics-env

# Run grader
pytest tests/test_environment.py -v
```

**What the grader checks:**

- `reset()` returns a valid `ApplicantObservation` with all required fields populated
- `step()` returns a `StepResult` with a numeric reward, `done` flag, and next observation
- `state()` returns a `State` with correct `episode_id` and `step_count`
- Reward is bounded within `[-2.0, 2.0]`
- Episode terminates correctly after `max_steps`
- Server handles concurrent requests without error
- Invalid actions are rejected with appropriate error responses

---

## 🤖 RL Agent Loop (`inference.py`)

```bash
python inference.py
```

Runs a heuristic RL agent for 5 episodes, printing step-by-step rewards and final episode scores. Demonstrates the complete RL training loop against this environment — ready to plug into TRL, torchforge, or any other OpenEnv-compatible framework.

---

## 📊 ML Oracle (Reward Ground Truth)

The environment's reward computation is backed by a production-grade ML pipeline:

| Component | Detail |
|---|---|
| Algorithm | XGBoost Gradient Boosting |
| Training data | 2,000+ synthetic MSME records, calibrated to Indian market |
| AUC-ROC | 0.82 |
| Features | 20+ indicators: revenue, DSCR, GST compliance, ITR, cash flow volatility |
| Explainability | SHAP values — top 5 factors per applicant |
| Scoring latency | < 100ms |

The scoring engine at `backend/services/scoring_engine.py` serves as the ground-truth oracle. Agents that learn to align their decisions with the PD model's risk assessments will achieve the highest cumulative rewards.

---

## 🌐 Deploy to Hugging Face Spaces

```bash
huggingface-cli login
openenv push --repo-id your-username/finlytics-env
```

Once deployed, any RL framework can connect directly:

```python
async with FinlyticsEnv(base_url="https://your-username-finlytics-env.hf.space") as env:
    result = await env.reset()
```

---

## 🔗 RL Framework Compatibility

| Framework | Compatible |
|---|---|
| [TRL (Hugging Face)](https://huggingface.co/docs/trl/openenv) | ✅ GRPO training |
| [torchforge](https://github.com/meta-pytorch/torchforge) | ✅ PyTorch-native RL |
| [SkyRL (UC-Berkeley)](https://skyrl.readthedocs.io) | ✅ |
| [Unsloth](https://github.com/unslothai/unsloth) | ✅ Efficient fine-tuning |
| [ART (OpenPipe)](https://art.openpipe.ai/integrations/openenv-integration) | ✅ |
| [Oumi](https://github.com/oumi-ai/oumi) | ✅ |

---

## 📋 `openenv.yaml` Manifest

```yaml
name: finlytics-env
version: 0.1.0
description: >
  MSME loan credit decision RL environment. An agent evaluates
  Indian MSME applicant profiles and learns to make accurate
  credit decisions optimised for both approval rate and portfolio quality.
author: your-username
license: MIT
tags:
  - finance
  - credit-risk
  - msme
  - india
  - decision-making
action_space: CreditDecisionAction
observation_space: ApplicantObservation
max_steps: 10
reward_range: [-2.0, 2.0]
```

---

## 👥 Team

| Role | Responsibility |
|---|---|
| **RL / OpenEnv** | Environment design, reward function, grader, inference loop |
| **Backend** | FastAPI scoring API, document extraction, application management |
| **ML** | XGBoost PD model, SHAP pipeline, synthetic data generation |
| **Frontend** | Next.js dashboard for borrowers and credit managers |

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 🙏 Built With

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) · [PyTorch](https://pytorch.org) · [Hugging Face](https://huggingface.co) · [XGBoost](https://xgboost.readthedocs.io) · [FastAPI](https://fastapi.tiangolo.com) · [Next.js](https://nextjs.org)

---

> Built at **Meta PyTorch OpenEnv Hackathon × Scaler School of Technology, Bengaluru** 🇮🇳
> Real infrastructure. Real impact. Contributing to the OpenEnv open-source ecosystem.
