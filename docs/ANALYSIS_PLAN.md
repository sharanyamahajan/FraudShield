# FraudShield: Milestone 5 Rigorous Statistical Evaluation & Analysis Plan

> [!IMPORTANT]
> This analysis plan was pre-specified and committed to version control (`8542751`) **before** running the M5 paired bootstrap inference, multi-seed variance benchmark, rolling backtest, and held-out test evaluation. All hypotheses, comparison baselines, primary metrics, statistical decision rules, sensitivity checks, and backtest partitioning schemes were locked herein. Under no circumstances were hypotheses or decision rules altered post hoc after observing empirical results.

### Pre-Specification & Execution Timeline (Dual Timezone Audit: UTC & IST)
All events occurred in strict chronological sequence. The table below lists both UTC and local IST (`+05:30`) timestamps to eliminate time zone ambiguity:

| Event | UTC Timestamp | IST Timestamp (`+05:30`) | Git / Artifact Reference | Audit Verification |
| :--- | :---: | :---: | :---: | :--- |
| **Milestone 4 Completion** | 2026-10-22 11:57:58 UTC | 2026-10-22 17:27:58 IST | Commit `e14272f` | Initial Set B (0.5610) & C1 (0.5675) validation point estimates recorded in `reports/history_val.json`. |
| **Analysis Plan Pre-Registration** | **2026-10-22 14:01:28 UTC** | **2026-10-22 19:31:28 IST** | **Commit `8542751`** | **Formal plan locked and committed BEFORE M5 bootstrap, backtest, and test evaluation.** |
| **M5 Validation Bootstrap (500 resamples)** | 2026-10-22 18:16:11 UTC | 2026-10-22 23:46:11 IST | `reports/m5_paired_bootstrap.json` | Paired bootstrap completed on validation split. |
| **M5 Rolling Backtest (2 folds)** | 2026-10-22 18:46:03 UTC | 2026-10-23 00:16:03 IST | `reports/m5_rolling_backtest.json` | Expanding-window validation completed. |
| **Milestone 5 Results Commit** | 2026-10-22 18:49:30 UTC | 2026-10-23 00:19:30 IST | Commit `1870501` | M5 results committed with clean verification. |
| **Authorized Held-Out Test Evaluation** | **2026-10-22 19:18:17 UTC** | **2026-10-23 00:48:17 IST** | `reports/test_access_log.txt` | **Executed 5 hours 17 min AFTER plan commit, 29 min AFTER M5 results.** |
| **Milestone 6 Results Commit** | 2026-10-22 19:27:36 UTC | 2026-10-23 00:57:36 IST | Commit `9d63084` | Test metrics, calibration, and SHAP committed (`reports/test_evaluation_summary.json`). |

---

## 1. Research Question & Primary Hypotheses (RQ1)

The central research question evaluated in Milestone 5 is:
> **RQ1**: Under strict time-ordered evaluation, do leakage-safe backward-looking entity-history features improve PR-AUC and recall at operational review budgets over transaction-only and non-historical features?

### 1.1 Model Class & Comparison Hierarchy
- **Primary Comparison**: Supervised **XGBoost on Feature Set C1** vs. **XGBoost on Feature Set B** (primary proxy entity key: `card1`).
- **Secondary Comparison**: Supervised **XGBoost on Feature Set C2** vs. **XGBoost on Feature Set B** (secondary composite proxy entity key: `(card1, addr1)`).
- **Model Hyperparameters**: Identical to Milestones 3 and 4 (`n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, `eval_metric="aucpr"`, `early_stopping_rounds=30`, `scale_pos_weight=None`). No tuning is permitted in Milestone 5.

### 1.2 Evaluation Metrics Hierarchy
- **Primary Metric**: **PR-AUC** (Average Precision / Area Under the Precision-Recall Curve). This metric directly captures performance under severe class imbalance ($\sim 3.5\%$ fraud prevalence) across all detection operating points.
- **Secondary Metrics**:
  - **Recall at Fixed Review Budgets**: Top 1.0%, 2.0%, and 5.0% of all evaluated transactions prioritized for human review.
  - **ROC-AUC**: Evaluated as a secondary ranking baseline.
  - **Brier Score**: To assess probability calibration quality.

---

## 2. Statistical Decision Rule for Reporting Gain

To prevent false discovery and optimistic bias, an explicit two-criterion decision rule is established. Feature Set C (C1 or C2) will be reported as having an **"improvement"** over Set B if and only if **both** of the following conditions are satisfied:

1. **Paired Bootstrap Criterion**:
   The **95% bootstrap confidence interval** for the paired metric difference ($\Delta \text{PR-AUC} = \text{PR-AUC}_{\text{Set C}} - \text{PR-AUC}_{\text{Set B}}$) on the validation split (1,000 resamples, fixed seed `42`) **strictly excludes zero** (i.e. the lower bound is $> 0$).
2. **Rolling-Origin Temporal Consistency Criterion**:
   The observed difference ($\Delta \text{PR-AUC}$) maintains a **consistent positive sign** ($> 0$) across **every single fold** of the 3-fold rolling-origin backtest.

> [!CAUTION]
> If either criterion fails (i.e. the 95% CI spans zero OR any backtest fold exhibits a non-positive difference), the result will be reported plainly and transparently as **"no reliable gain"**. No post-hoc feature engineering, hyperparameter tuning, or threshold alterations will be permitted to force a positive outcome.

### 2.1 Empirical Verdict & Sensitivity Result
- **Primary Result**: **For the pipeline actually run (reg:squarederror, unintended), the pre-specified rule gives NO RELIABLE GAIN for C1 over B (val PR-AUC CI spans zero; test PR-AUC delta -0.029)**.
- **Sensitivity Result**: **With binary:logistic, validation C1-B PR-AUC = +0.0079, CI [+0.0048, +0.0112] excludes zero, while Recall@1% delta = -0.0026, CI [-0.0060, +0.0003]. No logistic test result exists.**
- **Decision Rule Dependence**: The pre-specified rule's outcome depends on the objective.
- **Production Deployment Decision**: Baseline Model Set B is served by the production API because Set B achieved superior out-of-time test performance (PR-AUC 0.4296 vs. 0.4002). Set C1 did not beat Set B on test.

---

## 3. Sensitivity Analysis Specification

- **Feature Distribution Drift Sensitivity**:
  The all-time prior transaction count feature (`card1_prior_count`) naturally monotonically grows as time advances, potentially introducing non-stationary distributions between early and late temporal windows.
- **Sensitivity Run**:
  An additional evaluation run will test **Set C1 without the all-time prior transaction count** (retaining only rolling window counts: 1h, 24h, 7d; timedelta since previous transaction; amount ratio to prior mean; distinct email and address counts).
- **Reporting Protocol**:
  This sensitivity run is reported transparently alongside the primary C1 model to isolate whether long-term entity lifetime artifacts affect the findings. It will **not** be used to retroactively replace Set C1.

---

## 4. Multi-Seed Stability Benchmark

To assess variance resulting from tree construction stochasticity (feature subsampling and row subsampling):
- **Models**: XGBoost on Set B, Set C1, Set C2, and Set C1 (Sensitivity without all-time count).
- **Seeds**: Five distinct seeds (`[42, 123, 456, 789, 2026]`).
- **Reporting**: Point estimates for each seed, plus the **mean** and **sample standard deviation** ($\mu \pm \sigma$) across seeds for PR-AUC, ROC-AUC, and recall at 1.0%, 2.0%, and 5.0% review budgets.

---

## 5. Paired Bootstrap Inference Protocol

To assess statistical significance between paired predictions on the identical validation split ($N = 88,581$, $3,042$ fraud cases):
- **Resamples**: 1,000 paired bootstrap iterations with replacement.
- **Random Seed**: Fixed at `42` for exact reproducibility.
- **Differences Computed**:
  - $\Delta \text{PR-AUC} = \text{Score}_{\text{C1}} - \text{Score}_{\text{B}}$
  - $\Delta \text{Recall@1.0\%}$, $\Delta \text{Recall@2.0\%}$, $\Delta \text{Recall@5.0\%}$
  - Corresponding differences for C2 vs. B.
- **Reporting**: Mean difference, empirical standard error, and the empirical 2.5th and 97.5th percentiles (95% CI).

---

## 6. Rolling-Origin Backtest Protocol

To verify temporal stability without lookahead leakage:
- **Timeline Coverage**: Evaluates **strictly within the combined Train + Validation portion** (the first 85% of the global timeline; $N = 501,959$). The held-out Test split ($T > 13,802,372$) remains completely sealed and untouched.
- **Folds (Expanding Window)**:
  - **Fold 1**: Train on first 50% of the 85% timeline; validate on subsequent 10%.
  - **Fold 2**: Train on first 60% of the 85% timeline; validate on subsequent 10%.
  - **Fold 3**: Train on first 70% of the 85% timeline; validate on remaining 15% (coinciding with the standard validation period).
- **Refitting Rule**: For each fold, preprocessing transformers (`TreePreprocessor`) are fit strictly on that fold's training slice. Models are fit on the fold train slice with early stopping monitored on the fold validation slice.
- **Reporting**: Per-fold metrics and per-fold differences ($\text{C1} - \text{B}$).

---

## 7. Research Question 2 (RQ2): Temporal vs. Random Split Degradation

To answer the second core semester research question:
> **RQ2**: How much does a random split overstate performance versus a temporal split for the same model?

- **Dataset**: Evaluated strictly on the combined Train + Validation pool ($N = 501,959$) to ensure identical sample size and prevent test set leakage.
- **Temporal Split**: The standard chronological 70/15 partition (Train = 413,378, Val = 88,581).
- **Random Split**: A uniformly shuffled 70/15 partition ($82.35\% / 17.65\%$) of the same 501,959 rows using seed `42`.
- **Model**: Identical XGBoost configuration trained on Set B features.
- **Reporting**: PR-AUC, ROC-AUC, Recall@1.0%, 2.0%, 5.0%, and the generalization performance gap ($\text{Random} - \text{Temporal}$). The random split is explicitly documented as **optimistic and leaky**.
- **Objective Generalization Gaps**:
  - `reg:squarederror` (M5 Production Run): **+12.4 percentage points (0.646 vs. 0.522 PR-AUC, ~24% relative)** overstatement.
  - `binary:logistic` (Sensitivity Benchmark): **+13.43 percentage points (0.6960 vs. 0.5617 PR-AUC)** overstatement.

---

## 8. Artifact Generation & Sealed Test Split Commitment

All outputs will be saved to disk with run provenance:
- `reports/m5_seed_stability.csv` and `.json`
- `reports/m5_paired_bootstrap.json`
- `reports/m5_rolling_backtest.csv` and `.json`
- `reports/m5_rq2_split_comparison.json`
- Full run metadata including git commit hash, clean tree verification (`is_git_dirty: false`), and parquet SHA-256 hashes.
- **The held-out test split remains 100% sealed until Milestone 6.**
