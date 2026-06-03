## Week 4 — Credit Risk Probability Model (Alternative Data)

This repository implements the Week 4 challenge: build the foundations of a credit-risk scoring product for a **buy-now-pay-later (BNPL)** use case using transaction-level alternative data.

### Repository structure

| Path | Role |
|------|------|
| `data/raw/data.csv` | Xente transactions (local; gitignored) |
| `notebooks/eda.ipynb` | **Task 2** — full EDA notebook |
| `reports/interim_report.md` | Interim report (Tasks 1 & 2) |
| `analysis_outputs/task2/` | Saved EDA figures |
| `scripts/run_task2_eda.py` | Regenerate plots from CLI |
| `scripts/build_eda_notebook.py` | Regenerate `eda.ipynb` |

### Quick start

```bash
cd credit-risk-model
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_task2_eda.py
python scripts/build_eda_notebook.py
python scripts/generate_interim_report_pdf.py
jupyter notebook notebooks/eda.ipynb
```

> Generated artifacts:
> - `analysis_outputs/task2/` — EDA figures
> - `notebooks/eda.ipynb` — regenerated Task 2 notebook
> - `reports/interim_report_revised.pdf` — final interim report PDF

---

## Credit Scoring Business Understanding (Task 1)

### 1) Basel II and why interpretability + documentation matter

Basel II emphasizes that banks must be able to **measure, explain, and govern** their credit risk models. In practice, this pushes us toward:

- **Interpretability**: stakeholders (risk, compliance, auditors) must understand *why* the model assigns high risk to a customer; opaque models make it harder to justify decisions and detect bias or instability.
- **Traceability and auditability**: every modeling choice (data snapshot, transformations, target definition, features, hyperparameters, evaluation) must be reproducible so outcomes can be reviewed.
- **Model governance**: monitoring drift, backtesting, and clear limitations are required because the model affects credit decisions, loss provisioning, and capital calculations.

### 2) Why a proxy “default” variable is necessary + business risks

The provided transaction dataset contains **no direct label for default** (no loan repayment history). To train a supervised model, we must define a **proxy target** (e.g., an RFM-based “high risk” segment) that approximates default likelihood.

This introduces business risks:

- **Label risk (proxy ≠ truth)**: the proxy may not match real default behavior, leading to wrong approvals/declines.
- **Feedback loops**: if the proxy is based on spending patterns, the model may penalize customers who transact less for reasons unrelated to creditworthiness (e.g., new users, seasonal behavior).
- **Fairness and inclusion**: alternative-data proxies can embed socioeconomic or access biases; decisions must be tested for disparate impact.
- **Operational risk**: if the proxy changes over time (platform growth, pricing/channel changes), model performance can degrade quickly without monitoring.

### 3) Trade-offs: interpretable model (LogReg + WoE) vs high-performance (Gradient Boosting)

- **Logistic Regression + WoE (interpretable)**
  - Pros: transparent coefficients; easy to document; stable; aligns with classic scorecard practice; simpler to validate.
  - Cons: may underfit complex nonlinear relationships; feature engineering effort can be higher; performance may be lower.

- **Gradient Boosting (high performance)**
  - Pros: captures nonlinearities/interactions; often better predictive power on tabular data.
  - Cons: harder to explain; needs additional explainability tooling (e.g., SHAP); more sensitive to leakage and drift; governance burden is higher.

In a regulated context, the usual approach is to **start with an interpretable baseline** and only adopt higher-capacity models if the performance gains justify the added governance and explainability overhead.

