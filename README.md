# FraudShield: Leakage-Safe Transaction Fraud Risk Engine

[![Test Suite](https://img.shields.io/badge/pytest-45%20passed-brightgreen.svg)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-blue.svg)](src/fraudshield/api/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

FraudShield is an institutional-grade machine learning system designed for financial transaction fraud risk scoring and operational review capacity prioritization. It is built strictly under zero-lookahead temporal evaluation constraints on the IEEE-CIS Fraud Detection dataset (590,540 transactions).

---

## 🎯 Executive Summary & Research Findings

### 1. Research Question 1 (RQ1): Backward-Looking Entity History Features
- **Question**: Under strict time-ordered evaluation ($T_{prior} < T_{current}$), do leakage-safe backward-looking entity history features (past 1h/24h/7d counts, timedelta, amount-to-mean ratios) improve PR-AUC over transaction-only features?
- **Pre-Specified Evaluation**: Evaluated under decision rules pre-specified before the M5 bootstrap and test evaluation in [`docs/ANALYSIS_PLAN.md`](docs/ANALYSIS_PLAN.md).
- **Empirical Result**: **`NO RELIABLE GAIN`**.
  - On Validation ($N=88,581$): Paired bootstrap 95% CI for $\Delta \text{PR-AUC}$ was `[-0.0005, +0.0059]` (includes 0).
  - On Held-Out Test ($N=88,581$): Baseline **Set B achieved PR-AUC = 0.4296** vs **Set C1 PR-AUC = 0.4002** ($\Delta = -0.0294$, 95% CI: `[-0.0345, -0.0243]`).
  - **Val-to-Test Swing Interpretation**: The performance drop from validation (+0.0027) to test (-0.0294) is treated as a hypothesis (potential entity count distribution drift over time), not a definitive finding.
  - **Probability Calibration on Test**: Calibration worsened test Brier score from 0.0290 (uncalibrated) to 0.0319 (isotonic calibrator fit on validation only) under out-of-time distribution drift; the effect is small (+0.00286).
  - **Served Model**: The API serves the Primary Model (Set C1) with raw scores and calibrated probabilities.

### 2. Research Question 2 (RQ2): Temporal vs. Random Split Degradation
- **Question**: How much does standard random cross-validation overstate performance versus realistic chronological splitting?
- **Empirical Result**: **+12.4 percentage points (0.646 vs 0.522 PR-AUC, ~24% relative)** overstatement.
- **Conclusion**: Random train/val splitting creates massive optimistic leakage due to future-to-past entity lookahead. This shows (single dataset) that temporal evaluation is essential for production fraud risk estimation.

### 2. Research Question 2 (RQ2): Temporal vs. Random Split Degradation
- **Question**: How much does standard random cross-validation overstate performance versus realistic chronological splitting?
- **Empirical Result**: **+12.4 percentage points (0.646 vs 0.522 PR-AUC, ~24% relative)** overstatement.
- **Conclusion**: Random train/val splitting creates massive optimistic leakage due to future-to-past entity lookahead. This shows (single dataset) that temporal evaluation is essential for production fraud risk estimation.

---

## 🏛️ System Architecture

```
FraudShield/
├── data/
│   ├── raw/                 # Sealed raw transaction & identity tables
│   ├── interim/             # Deterministically joined parquet
│   └── processed/           # Temporal splits (Train 70%, Val 15%, Test 15%)
├── models/
│   ├── xgb_set_c1.json      # Final trained XGBoost model binary
│   ├── preprocessor_c1.pkl  # Train-fit feature encoder & imputer
│   └── calibrator.pkl       # Isotonic probability calibrator
├── reports/
│   ├── fraudshield.db       # SQLite database with 88,581 scored test records & SHAP
│   ├── test_access_log.txt  # Tamper-evident authorized test access log
│   ├── m5_paired_bootstrap.json
│   ├── m5_rq2_split_comparison.json
│   ├── test_evaluation_summary.csv
│   └── test_operating_points.csv
├── src/fraudshield/
│   ├── api/                 # FastAPI REST API & interactive dashboard
│   ├── features/            # Leakage-safe preprocessors & pipeline
│   ├── splits/              # Strict chronological splitter & security guard
│   ├── evaluation/          # PR-AUC, ROC-AUC, review budget metrics engine
│   └── models/              # Model training, convergence & baselines
└── tests/                   # 45 passing unit, integration, and guard tests
```

---

## ⚡ Quickstart: Running the API & Dashboard

### 1. Start the FastAPI Service
```powershell
uvicorn fraudshield.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Access the Application
- **Interactive Review Dashboard**: Navigate to [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/)
- **Swagger Interactive API Docs**: Navigate to [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)
- **Health Check**: `GET http://127.0.0.1:8000/health`

---

## 📊 Operational Review Budget Operating Points

Operating points are fixed on Validation ($\theta$) and evaluated on the held-out Test split ($N=88,581$, 3,083 actual frauds):

| Operational Tier | Validation Cutoff ($\theta$) | Test Flagged Count & Share | Frauds Caught | Fraud Capture (Recall) | Alert Precision (Threshold-Based) | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top 0.5% Priority** | $\ge 0.8684$ | 515 (~0.58%) | 486 | **15.76%** | **94.37%** | 0.03% |
| **Top 1.0% High Risk** | $\ge 0.6139$ | 1,742 (~1.97%) | 835 | **27.08%** | **47.93%** | 1.06% |
| **Top 2.0% Secondary** | $\ge 0.2665$ | 3,503 (~3.95%) | 1,278 | **41.45%** | **36.48%** | 2.60% |
| **Top 5.0% Elevated** | $\ge 0.1154$ | 6,670 (~7.53%) | 1,791 | **58.09%** | **26.85%** | 5.71% |

> **Rank-Based vs. Threshold-Based Precision Confirmation**:
> - **Rank-Based Precision** (reviewing strictly the top $N \times b$ transactions by descending risk score):
>   - Baseline Set B at top 1.0% capacity (top 886 transactions): **86.00% precision** (860 true frauds).
>   - Primary Set C1 at top 1.0% capacity (top 886 transactions): **72.69% precision** (644 true frauds).
> - **Threshold-Based Precision** (flagging all transactions with score $\ge \theta$ determined on validation):
>   - Because score distributions shifted out-of-time on the test split, the validation cutoff $\theta \ge 0.6139$ flagged **1,742 transactions (~1.97% volume)** instead of 886, capturing 835 frauds with **47.93% precision**.

---

## 🔍 REST API Endpoints

- `GET /health`: Health and readiness state.
- `GET /api/metadata`: Trained model specifications, commit hash, and test scores.
- `GET /api/operating-points`: Cutoffs, review volumes, precision, and recall by tier.
- `GET /api/features/importance`: Top global SHAP feature attributions and feature families.
- `GET /api/transactions`: Paginated test transactions with risk scores and tier filters.
- `GET /api/transactions/{id}`: Single transaction details with local SHAP positive/negative drivers.
- `POST /api/score`: Real-time transaction scoring and tier assignment.
- `GET /api/experiments/rq2`: RQ2 random vs temporal split overstatement data.
- `GET /api/experiments/bootstrap`: 500-sample paired bootstrap confidence intervals.

---

## 🧪 Reproducibility & Testing

Run the full automated test suite (45 tests covering data integrity, preprocessors, leakage verification, metrics rounding, guard isolation, and API routes):

```powershell
python -m pytest -v
```

All 45 tests pass in under 75 seconds without touching sealed test data.
