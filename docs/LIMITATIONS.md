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
