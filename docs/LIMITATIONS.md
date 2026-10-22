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

### 6.1 Set B PR-AUC Discrepancy: 0.5610 (M3/M4) vs. 0.5222 (M5)
- **Technical Cause (Unintended Deviation)**: In Milestones 3 and 4, models were trained using `xgb.XGBClassifier`, which defaults to `objective="binary:logistic"` optimizing log-loss cross-entropy. In Milestone 5, to overcome peak RAM bloat and achieve 5x training acceleration with contiguous `float32` DMatrix arrays, the script switched to the low-level `xgb.train` API. Omitting the `objective` parameter in `xgb.train` caused XGBoost to default to `reg:squarederror` (optimizing mean squared error on $\{0, 1\}$ targets). **This was an unintended deviation, not an intentional speed optimization.**
- **Validation Score Impact**: The M3/M4 logistic models scored higher on validation (`0.5610` for Set B and `0.5675` for Set C1) than the squared-error models (`0.5222` for Set B and `0.5250` for Set C1).
- **Validation-Only Logistic Recomputation**: A follow-up validation-only benchmark (`reports/validation_logistic_benchmark.json`) confirmed this discrepancy. Re-running Set B and Set C1 with `objective="binary:logistic"` under identical hyperparameters restored validation PR-AUC to `0.561701` for Set B and `0.569598` for Set C1.
- **Side-by-Side Validation 95% Confidence Intervals (500 Resamples, C1 - B)**:
  - `reg:squarederror` (M5 Production Run): $\Delta \text{PR-AUC} = +0.002740$, 95% CI: **`[-0.000483, +0.005893]`** (spans zero).
  - `binary:logistic` (Audit Run): $\Delta \text{PR-AUC} = +0.007898$, 95% CI: **`[+0.004832, +0.011221]`** (strictly positive).
  - *Operational Recall*: Even under `binary:logistic`, at the top 1.0% review budget, $\Delta \text{Recall@1.0\%} = -0.002630$ (95% CI `[-0.006008, +0.000340]`), confirming that entity history did not improve fraud capture at operational review capacity.
  - *RQ2 Gap with Logistic*: Random split Set B achieved `0.695991` vs. Temporal Set B `0.561701`, yielding a **+13.43 percentage point overstatement** (+0.134290).
- **Deployed Models & Clipping**:
  - The downstream test evaluation, API, SHAP, and calibration utilize `models/xgb_set_c1.json` (Set C1) and `models/xgb_set_b.json` (Set B baseline), which were trained under `reg:squarederror`.
  - In both `scripts/evaluate_test.py` and `src/fraudshield/api/main.py`, raw scores are explicitly clipped to $[0, 1]$ (`np.clip(score, 1e-6, 1.0 - 1e-6)`) before calibration and Brier evaluation.

### 6.2 Execution Script Commit Audit
- `docs/ANALYSIS_PLAN.md` was registered and committed at commit `8542751` **before** executing M5 experiments.
- The pipeline script `scripts/run_milestone5.py` was adapted for memory constraints and speed optimization during execution and was formally committed at commit `1870501` **after** the run concluded.

### 6.3 Backtest Redundancy & Independence of Folds
- **Fold 2 Duplication**: In the 2-fold rolling-origin backtest, Fold 2 (70% train / 15% val) trained on rows 0 to 413,378 and validated on rows 413,378 to 501,959. This partition window is mathematically identical to the primary validation split.
- **Independent Folds**: Because Fold 1 and Fold 2 share training data (expanding window) and Fold 2 duplicates the main validation split, the "same sign in every fold" decision rule actually evaluated only **ONE** temporally distinct out-of-sample backtest fold (Fold 1) beyond the main validation split.
