# FraudShield: System Limitations & Operational Constraints

This document details critical real-world limitations, failure modes, and deployment boundaries for the FraudShield risk engine.

---

## 1. Temporal Generalization & Feature Distribution Drift

### Monotonic Growth of All-Time History
- The feature `card1_prior_count` naturally increases as an entity ages in the dataset. Early training windows exhibit lower mean entity counts than late validation and test windows.
- In Milestone 6, held-out test evaluation demonstrated that Set B (non-historical features) outperformed Set C1 (`card1` history) by **+2.94% PR-AUC**, indicating that unconditioned long-term entity counts suffer from non-stationary drift over multi-month horizons.
- **Production Recommendation**: Restrict entity historical features to bounded rolling windows (e.g. 1 hour, 24 hours, 7 days) and relative ratios (`amount_ratio_to_prior_mean`), excluding unbounded lifetime counters.

---

## 2. Cold-Start Entities & Missing Key Context
- Transactions with unobserved `card1` values lack prior transaction history, resulting in `NaN` for historical velocity, timedelta, and amount ratios.
- The pipeline utilizes cold-start indicators (`card1_prior_count = 0` / indicator flags) and tree-native NaN branch routing. However, cold-start predictions exhibit wider prediction confidence intervals.
- In production, new payment cards should be augmented with cross-entity signals (e.g. device fingerprinting, IP subnet velocity).

---

## 3. Explainability Boundaries of Anonymised Features (`V*`)
- Features `V1` through `V339` are proprietary Vesta-engineered features whose exact mathematical definitions are not publicly disclosed.
- While TreeExplainer SHAP values accurately describe tree path splits (e.g., `V244` as the top global importance driver), **analysts cannot interpret `V*` features as causal real-world attributes**.
- In the FraudShield UI and API, anonymised features are explicitly flagged with `is_anonymised = True` to prevent analysts from fabricating causal justifications during customer disputes.

---

## 4. Class Imbalance & Operating Capacity Ceilings
- With $\sim 3.5\%$ fraud prevalence, traditional threshold-based decision making (e.g. fixed $\tau = 0.5$) generates erratic review volumes that overwhelm manual review capacity.
- The engine enforces capacity-constrained operational budgeting. Operating at top 0.5% review capacity achieves **94.4% precision**, but only captures **15.8% of total frauds**. Capturing 58.1% of frauds requires reviewing 5.0% of all transactions with precision dropping to **26.9%**.
- Organizations must calibrate operating points against available human analyst headcount and false-alert operational costs.

---

## 5. Security & Test Guard Isolation
- The test split isolation guard (`src/fraudshield/splits/guard.py`) uses Python call stack inspection (`inspect.stack()`) to prevent inadvertent data leakage during model development.
- This mechanism is an internal accidental leakage barrier rather than a cryptographic security boundary. It requires standard code review and deployment automation to enforce tamper prevention.

---

## 6. Discrepancy Analysis, Audit Trail & Backtest Structure

### 6.1 Objective Function & Verdict Discrepancy Analysis
- **Primary Result**: **For the pipeline actually run (reg:squarederror, unintended), the pre-specified rule gives NO RELIABLE GAIN for C1 over B (val PR-AUC CI spans zero; test PR-AUC delta -0.029)**.
- **Sensitivity Result**: **With binary:logistic, validation C1-B PR-AUC = +0.0079, CI [+0.0048, +0.0112] excludes zero, while Recall@1% delta = -0.0026, CI [-0.0060, +0.0003]. No logistic test result exists.**
- **Decision Rule Dependence on Objective**: The pre-specified decision rule's outcome depends on the objective function. Under squared-error, validation PR-AUC CI spans zero; under logistic, validation PR-AUC CI strictly excludes zero. Recall@1% is reported as secondary context and is not substituted for the pre-specified PR-AUC criterion.
- **Technical Cause (Unintended Deviation)**: In Milestones 3 and 4, models were trained using `xgb.XGBClassifier`, which defaults to `objective="binary:logistic"` optimizing log-loss cross-entropy. In Milestone 5, to overcome peak RAM bloat and achieve training acceleration with contiguous `float32` DMatrix arrays, the script switched to the low-level `xgb.train` API. Omitting the `objective` parameter in `xgb.train` caused XGBoost to default to `reg:squarederror` (optimizing mean squared error on $\{0, 1\}$ targets). **This was an unintended deviation, not an intentional speed optimization.**
- **Validation Score Impact**: The M3/M4 logistic models scored higher on validation (`0.5610` for Set B and `0.5675` for Set C1) than the squared-error models (`0.5222` for Set B and `0.5250` for Set C1).
- **Validation-Only Logistic Recomputation**: A follow-up validation-only benchmark (`reports/validation_logistic_benchmark.json`) confirmed this discrepancy. Re-running Set B and Set C1 with `objective="binary:logistic"` under identical hyperparameters restored validation PR-AUC to `0.561701` for Set B and `0.569598` for Set C1. The rolling backtest and multi-seed stability runs were not repeated under logistic.
- **Side-by-Side Validation 95% Confidence Intervals (500 Resamples, C1 - B)**:
  - `reg:squarederror` (M5 Production Run): $\Delta \text{PR-AUC} = +0.002740$, 95% CI: **`[-0.000483, +0.005893]`** (spans zero).
  - `binary:logistic` (Sensitivity Benchmark): $\Delta \text{PR-AUC} = +0.007898$, 95% CI: **`[+0.004832, +0.011221]`** (strictly positive).
  - *Secondary Recall@1%*: $\Delta \text{Recall@1.0\%} = -0.002630$, 95% CI: `[-0.006008, +0.000340]`.
  - *RQ2 Gap with Logistic vs Squared-Error*: Random split overstates PR-AUC by **+12.4 pp** under squared-error (`0.646` vs. `0.522`), and by **+13.43 pp** under logistic (`0.6960` vs. `0.5617`).
- **Deployed Model & Production Serving**:
  - The downstream production API, SHAP explainability, and calibration serve **Baseline Model Set B** (`models/xgb_set_b.json`), because Set B achieved superior out-of-time test performance (PR-AUC 0.4296 vs. 0.4002). Model Set C1 did not beat Model Set B on test.
  - **Post-Hoc Model B Test Calibration**: On the held-out test split, served Model B achieved an uncalibrated Brier score of `0.026810` and an isotonic-calibrated Brier score of `0.029408` (calibrator fit on validation only). This metric is explicitly labeled **"post-hoc, test split already seen"**, as Model B was selected for deployment following comparative test evaluation. Consistent with Set C1 (`0.0290` $\rightarrow$ `0.0319`), calibration slightly widened test Brier error (+0.0026) under temporal drift.
  - In both `scripts/evaluate_test.py` and `src/fraudshield/api/main.py`, raw scores are explicitly clipped to $[0, 1]$ (`np.clip(score, 1e-6, 1.0 - 1e-6)`) before calibration and evaluation.

### 6.2 Execution Script Commit Audit & Test Split Sealing
- **Analysis Plan Locked Pre-Run**: `docs/ANALYSIS_PLAN.md` was registered and committed at commit `8542751` at `14:01:28 UTC` (`19:31:28 IST`), well **before** executing M5 experiments.
- **Uncommitted Pipeline Script**: `scripts/run_milestone5.py` was adapted for memory constraints and speed optimization during execution and ran from an uncommitted working tree; therefore, `is_git_dirty=true` is recorded in all M5 report metadata files. The script was formally committed at `1870501` along with M5 results.
- **Test Access Log Cleanup**: `reports/test_access_log.txt` was edited in commit `719d51e` to remove transient pytest test fixture entries; the original log lines with temporary test invocations are preserved in Git history (`0317c1f` and earlier). The real log now contains exclusively the two authorized production test accesses.
- **Single Test Split Read**: The held-out test split was read strictly once (at `19:18:17 UTC` / `00:48:17 IST`), after `docs/ANALYSIS_PLAN.md` was committed and after Milestone 5 was finalized.

### 6.3 Backtest Redundancy & Independence of Folds
- **Fold 2 Duplication**: In the 2-fold rolling-origin backtest, Fold 2 (70% train / 15% val) trained on rows 0 to 413,378 and validated on rows 413,378 to 501,959. This partition window is mathematically identical to the primary validation split.
- **Independent Folds**: Because Fold 1 and Fold 2 share training data (expanding window) and Fold 2 duplicates the main validation split, the "same sign in every fold" decision rule actually evaluated only **ONE** temporally distinct out-of-sample backtest fold (Fold 1) beyond the main validation split.
- **No Logistic Backtest**: The rolling backtest was evaluated under `reg:squarederror` and was not repeated under `binary:logistic`.
