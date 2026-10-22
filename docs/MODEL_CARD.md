# FraudShield Model Card

> [!NOTE]
> **Demo only: no auth, returns labels, not production-ready.**

Following the Model Cards for Model Reporting framework (Mitchell et al., 2019).

---

## 1. Model Details
- **Model Name**: FraudShield XGBoost Risk Prioritization Model (`xgb_set_c1_m6`).
- **Model Version**: 1.0.0
- **Model Architecture**: Extreme Gradient Boosted Decision Trees (`xgboost.Booster`, `tree_method="hist"`).
- **Hyperparameters**: `max_depth=6`, `learning_rate=0.05`, `n_estimators=300`, `early_stopping_rounds=30`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=1.0`.
- **Calibration**: Non-parametric Isotonic Regression fit on validation predicted probabilities.
- **Developer**: FraudShield Redesign Team.
- **License**: MIT License.

---

## 2. Intended Use
- **Primary Intended Use**: Financial transaction manual review capacity prioritization under severe class imbalance ($\sim 3.5\%$ base fraud prevalence).
- **Primary Users**: Fraud operations analysts, compliance investigators, risk strategy teams.
- **Out-of-Scope Uses**: Automated transaction cancellation without human review, credit scoring, legal culpability determinations.

---

## 3. Training & Evaluation Data
- **Dataset**: IEEE-CIS Fraud Detection Benchmark (590,540 real-world e-commerce transactions).
- **Temporal Splitting**: Chronological 70/15/15 partition:
  - **Training Split**: First 413,378 transactions ($T \le 11,544,008$).
  - **Validation Split**: Intermediate 88,581 transactions ($11,544,008 < T \le 13,802,372$).
  - **Test Split**: Final held-out 88,581 transactions ($T > 13,802,372$).
- **Features**: Transaction core attributes (`TransactionAmt`, `ProductCD`, `card1-6`), network/address info (`addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`), device identity attributes (`id_01` to `id_38`), engineered time features, and backward-looking entity history (`card1_prior_count`, 1h/24h/7d counts, amount ratios).

---

## 4. Evaluation Metrics & Performance
- **Primary Performance Metric**: PR-AUC (Average Precision).
- **Secondary Metrics**: ROC-AUC, Brier Calibration Score, Recall & Precision at 0.5%, 1.0%, 2.0%, 5.0% manual review capacity budgets.
- **Evaluation Plan**: Pre-specified before the M5 bootstrap and test evaluation in [`docs/ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md).

### Out-of-Time Test Performance ($N=88,581$):
- **Test PR-AUC**: `0.400179` (Set C1) vs `0.429595` (Baseline Set B).
- **Test ROC-AUC**: `0.874754` (Set C1) vs `0.870473` (Set B).
- **Test Brier Score**: `0.029021` (uncalibrated) vs `0.031882` (calibrated with Isotonic regression fit on validation); calibration worsened test Brier under out-of-time drift (effect is small: +0.00286).
- **Served Model**: The production API serves Model Set C1 with both raw scores and calibrated probabilities.
- **Top 0.5% Capacity Tier**: Validation Cutoff $\ge 0.8684$ $\rightarrow$ flags 515 transactions (~0.58% of test), capturing 486 frauds with 94.37% Precision, 15.76% Fraud Capture Rate (FPR = 0.03%).
- **Top 1.0% Capacity Tier**: Validation Cutoff $\ge 0.6139$ $\rightarrow$ flags 1,742 transactions (~1.97% of test), capturing 835 frauds with 47.93% Precision, 27.08% Fraud Capture Rate (FPR = 1.06%).
- **Rank-Based vs. Threshold-Based Precision**: Under exact rank-based manual review (top 886 alerts), Baseline Set B achieved 86.00% precision and Set C1 achieved 72.69% precision. When evaluated at fixed validation thresholds, distribution drift expanded alert volume to 1,742 (~1.97%), yielding 47.93% precision.
- **Val-to-Test Swing**: The swing in relative performance from validation (+0.0027) to test (-0.0294) is treated as a hypothesis (entity drift), not a definitive finding.
- **RQ2 Generalization Gap**: Random 70/15 splitting overstates PR-AUC by **+12.4 percentage points (0.646 vs 0.522 PR-AUC, ~24% relative)**, which shows (single dataset) the danger of non-temporal evaluation.

---

## 5. Ethical Considerations & Fairness
- **Proxy Key Anonymity**: Primary entity proxy is `card1` (card issuing bank identification). Demographic attributes (race, gender, age) are absent from the dataset.
- **Vesta Anonymised Features (`V*`)**: Highly weighted in tree splits, but explicitly labeled in analyst UIs as structural behavioral patterns rather than causal confirmations.
- **Operational Bias**: Review queues prioritized by calibrated risk scores ensure human analysts inspect high-risk transactions before financial settlement.

---

## 6. Caveats & Recommendations
- **Entity Count Drift**: All-time transaction counts grow monotonically over time; rolling window features (1h, 24h, 7d) are recommended for continuous deployment.
- **Concept Drift**: Fraud tactics evolve rapidly; model weights should be refreshed periodically using expanding-window retraining.
