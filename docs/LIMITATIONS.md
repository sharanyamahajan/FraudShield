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
- **Technical Cause**: In Milestones 3 and 4, models were trained using the scikit-learn API `xgb.XGBClassifier` (which defaults to `objective="binary:logistic"` optimizing log-loss cross-entropy) and ingested raw pandas DataFrames with object columns. In Milestone 5, to overcome RAM exhaustion and achieve 5x training acceleration, training transitioned to the lower-level `xgb.train` API with pre-encoded contiguous `float32` DMatrix inputs. In `xgb.train`, omitting the `objective` parameter defaults to `reg:squarederror` (optimizing mean squared error on $\{0, 1\}$ targets rather than log-loss).
- **Internal Validity**: Because Set B (0.5222), Set C1 (0.5250), and Set C2 (0.5147) were all trained under the identical `xgb.train` objective and matrix formatting in Milestone 5, the comparative evaluation remains controlled and valid.

### 6.2 Execution Script Commit Audit
- `docs/ANALYSIS_PLAN.md` was registered and committed at commit `8542751` **before** executing M5 experiments.
- The pipeline script `scripts/run_milestone5.py` was adapted for memory constraints and speed optimization during execution and was formally committed at commit `1870501` **after** the run concluded.

### 6.3 Backtest Redundancy & Independence of Folds
- **Fold 2 Duplication**: In the 2-fold rolling-origin backtest, Fold 2 (70% train / 15% val) trained on rows 0 to 413,378 and validated on rows 413,378 to 501,959. This partition window is mathematically identical to the primary validation split.
- **Independent Folds**: Because Fold 1 and Fold 2 share training data (expanding window) and Fold 2 duplicates the main validation split, the "same sign in every fold" decision rule actually evaluated only **ONE** temporally distinct out-of-sample backtest fold (Fold 1) beyond the main validation split.
