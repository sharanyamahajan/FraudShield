# FraudShield (Redesigned): Project Plan

## 1. Project Objective & Scope
Build a reproducible, leakage-safe transaction fraud **risk-scoring system** for prioritizing manual review using the real-world, non-synthetic IEEE-CIS Fraud Detection (Vesta) dataset. 

The system estimates relative transaction risk to help fraud operations teams prioritize queue investigations. It **never** claims to prevent fraud or confirm criminal activity; all user-facing documentation and UI components strictly employ terminology such as *"estimated risk score"* and *"prioritised for review"*.

### Research Questions
- **RQ1**: Under strict time-ordered evaluation, do leakage-safe backward-looking entity-history features improve PR-AUC and recall at fixed review budgets (top 0.5%, 1%, 2%, 5%) over transaction-only features?
- **RQ2**: How much does a random split overstate performance versus a temporal split for the same model, and how much does performance drop from the validation period to the test period?
- **RQ3 (Optional Milestone 10 - Gated on explicit user confirmation)**: On the Elliptic Bitcoin graph dataset, does graph structure add value over node features under a temporal split?

---

## 2. Non-Negotiable Rules
1. **Zero Fabrication**: No synthetic data, fabricated metrics, or invented citations. Every metric shown in reports, README, API, or dashboard is loaded directly from saved results artifacts produced by verified code in this repo.
2. **No Leakage & No Synthetic Sampling**: Strict time-ordered splits. No SMOTE, ADASYN, or synthetic oversampling anywhere in the pipeline.
3. **No GNNs or Fraud-Ring Claims in Core Pipeline**: No graph neural networks in the primary IEEE-CIS pipeline; no unverified claims of "fraud rings" or "mule networks" without ground-truth topological identity.
4. **Verified Schema**: Before any column is used in features or models, verify its existence and data type in the actual files. If any expected column is missing, stop and report rather than improvising.
5. **Git Hygiene**: Raw data files (`data/raw/`, `*.csv`, `*.tar.gz`, `*.zip`, parquet interim files, model binaries, `.env`, `kaggle.json`) are strictly git-ignored and never committed.
6. **Cross-Platform Operability**: Every step runs seamlessly across Windows, macOS, and Linux via a standard cross-platform entry point (`python -m fraudshield <command>`), with a `Makefile` provided as a convenience wrapper.

---

## 3. Assumptions & Caveats
- **Proxy Entity Key**: The dataset does not provide verified account or user IDs. Features using `card1` (or `card1 + addr1`) are strictly defined and labeled as **proxy entity keys**.
- **Time Representation**: `TransactionDT` is a relative time offset in seconds from an unrevealed reference date, not a real timestamp. Any "hour of day" or "day of week" feature is an approximation and documented as such.
- **Anonymised Features**: V, C, D, and M columns are anonymised; their real-world meaning is unknown. They are grouped into feature families without inventing ungrounded domain narratives.
- **Label Semantics**: The exact label definition is unverified; `isFraud` is treated as the provided ground truth. Any claim about what it represents (e.g. chargebacks) is an assumption and must not appear in results as fact.
- **Strict Label-Delay Safety**: In real fraud operations, fraud labels arrive days or weeks late. For Set C, historical aggregation uses only past transaction metadata (frequencies, amount statistics, device transitions) and **never** past fraud labels, avoiding label-delay leakage.
- **Cost Formulation**: Financial impact is modeled as an illustrative proxy: missed fraud loss ≈ `TransactionAmt` (an explicitly assumed simplification), false-alert review cost ≈ configurable constant. Labeled clearly as an assumption-based illustration.

---

## 4. Split Design
1. **Primary Split (Time-Ordered Only)**:
   - Sort by `TransactionDT`.
   - Train = first 70%, Validation = next 15%, Test = final 15% (configurable via `configs/config.yaml`).
   - Assert strictly non-overlapping, ordered time ranges, and verify fraud rate and row count per split.
   - Held-out test set is evaluated **once** for final reporting. All tuning, early stopping, calibration, and threshold selection use validation only.
2. **Rolling-Origin (Expanding Window) Backtest**:
   - 3–4 folds, each training on the past and testing on the subsequent block, to measure variance across temporal periods.
3. **Random Split (RQ2 Reference Only)**:
   - A random split is executed solely as an optimistic, leaky reference baseline to answer RQ2, clearly marked as such.

---

## 5. Feature Sets (Ablation Ladder)
- **Set A (Transaction-only)**: Raw transaction attributes (`TransactionAmt`, `ProductCD`, `card1-6`, `addr1-2`, `dist1-2`, `P_emaildomain`, `R_emaildomain`, log amount, amount cents, approximate hour-of-day proxy from `TransactionDT`). Excludes all V, C, D, M, and identity columns.
- **Set B (Transaction + Anonymised + Identity)**: Set A + provided anonymised C*, D*, M*, V* columns and identity table columns (`DeviceType`, `DeviceInfo`, `id_*`).
- **Set C (Transaction + Anonymised + Identity + Entity-History)**: Set B + leakage-safe backward-looking entity-history features derived from proxy entity keys (`card1`, optionally `card1 + addr1`).

All encoders, imputers, and scalers are fitted **strictly on the Train split** and applied to Validation and Test splits. Unseen categories are mapped to a reserved missing/unknown category.

---

## 6. Experiments (E1 to E4)
- **E1 (Ablation Ladder & Baselines on Temporal Split - RQ1)**: Set A vs Set B vs Set C evaluated against Logistic Regression and Isolation Forest under the temporal split.
- **E2 (Temporal Split vs Random Split & Temporal Drop - RQ2)**: Best model trained and evaluated under a random split versus the primary temporal split, plus measurement of the performance drop from the validation period to the test period.
- **E3 (Rolling-Origin Backtest)**: 3–4 fold expanding-window evaluation across time blocks to capture temporal stability and variance.
- **E4 (Feature-Family Importance)**: Aggregated SHAP importance by feature family (raw transaction, anonymised V/C/D/M, identity, entity-history).

Saved artifacts in `reports/`: `metrics.json`, `model_comparison.csv`, `threshold_sweep.csv`, `rolling_backtest.csv`, `feature_importance.csv`.

---

## 7. Testing Plan
- **Leakage Tests**:
  - For a random sample of rows, entity-history features computed on the full dataset equal those computed on data truncated at that row's `TransactionDT`.
  - Shuffling or altering rows with later timestamps does not change earlier rows' features.
  - The current row is never included in its own history ($T_{prior} < T_{current}$).
- **Split Tests**:
  - Strict time ordering ($T_{train} < T_{val} < T_{test}$), non-overlapping ranges, disjoint `TransactionID` sets across splits.
- **Preprocessing Tests**:
  - Encoders, imputers, and scalers fitted on train only (asserting no val/test statistics are accessed); unseen categories handled gracefully.
- **Metric Tests**:
  - Metric calculations (PR-AUC, ROC-AUC, Recall@budget, FPR, Brier score) verified against small hand-built numerical fixtures with known outputs.
- **API Tests**:
  - FastAPI `TestClient` suite validating health, metric artifact retrieval, schema validation, transaction browsing, scoring endpoint, and clear 4xx/503 error handling.
- **Determinism Tests**:
  - Fixed seeds ensure reproducible metric generation within a documented numerical tolerance.

---

## 8. Risks & Mitigation Strategies
| Risk | Severity | Mitigation Strategy |
| :--- | :--- | :--- |
| **Data Leakage in Rolling Entity Aggregations** | Critical | Strict strict-inequality filtering ($T_{prior} < T_{current}$); automated 3-part leakage unit test suite required to pass before any model consumes Set C. |
| **Proxy Entity Key Effectiveness** | High | `card1` (or `card1+addr1`) may be too coarse or too sparse to be a useful proxy entity key, so Set C may show no gain over Set B. This is a valid and reportable RQ1 outcome, not a failure. Do not tune features until Set C wins; report the result as found. |
| **Optimistic Threshold Selection** | Medium | Calibration and operating-threshold selection both use the validation split, so the chosen operating point is slightly optimistic. Mitigation: split validation into two time-ordered halves (calibration vs. threshold selection) if the fraud count allows; otherwise document this as a limitation. |
| **Overly Optimistic Performance Metrics** | High | Primary metric is PR-AUC (Average Precision) and recall at operational review budgets (top 0.5%, 1%, 2%, 5%), not ROC-AUC alone; temporal split strictly enforced. |
| **Memory Exhaustion on IEEE-CIS (590k+ rows, 400+ cols)** | Medium | Downcast numeric datatypes upon loading; convert raw CSVs into memory-efficient Parquet with snappy compression for interim stages; release intermediate memory during preprocessing. |
| **Model Calibration Drift Across Time** | Medium | Calibrate final models on the validation split using Platt scaling / Isotonic regression; evaluate Brier score and calibration curves on the held-out test split. |
| **Dependency / Environment Variations** | Low | Pin dependencies in `requirements.txt`; provide standard cross-platform CLI (`python -m fraudshield`) and `Makefile` convenience wrapper; adhere to fixed random seeds. |

---

## 9. Milestone Checklist

- [ ] **M0: Plan and Repository Skeleton**
  - Project directory structure setup (`configs/`, `src/fraudshield/`, `scripts/`, `tests/`, `docs/`, `reports/`, `app/`).
  - Configuration (`configs/config.yaml`), `requirements.txt` (pinned), `.gitignore`, cross-platform command runner (`python -m fraudshield <command>`), `Makefile`.
  - Empty-but-passing pytest suite. Initial git commit.
  - *Verification*: `pytest` and `python -m fraudshield test` pass with 0 errors.

- [ ] **M1: Download Script, Load/Join, Data Report & Proxy Key Decision**
  - Kaggle acquisition script (`scripts/download_data.sh` and `scripts/download_data.py`).
  - Left join `train_transaction.csv` and `train_identity.csv` on `TransactionID`.
  - Data-load verification on actual files: row count, fraud rate, column list and dtypes, missingness, identity coverage, `TransactionDT` min/max/monotonicity/ties handling (asserting whether `TransactionDT` is sorted, reporting duplicate values, and defining how ties are handled so $T_{prior} < T_{current}$ remains valid), duplicate IDs, cardinality of key columns.
  - Write `docs/DATA_REPORT.md` and `notebooks/01_eda.ipynb`. Decide and justify proxy entity key.
  - *Verification*: Data report populated from verified counts on actual raw files.

- [ ] **M2: Time-Ordered Splits & Feature Pipelines (Sets A & B)**
  - Implement time-ordered split logic: 70% Train, 15% Validation, 15% Test sorted strictly by `TransactionDT`.
  - Split boundary non-overlap tests and ID disjointness verification.
  - Preprocessing pipelines for Set A and Set B fitted strictly on Train split.
  - *Verification*: Unit tests confirm zero data leakage across split boundaries and proper handling of unseen categories.

- [ ] **M3: Validation Baselines & Metric Suite**
  - Metric engine (PR-AUC, ROC-AUC, Recall@budget, Precision@budget, FPR, threshold sweep) + unit tests on hand-built arrays.
  - Baselines evaluated on validation split: Logistic Regression (scaled, median imputed), Isolation Forest (unsupervised on train features), XGBoost on Set A and Set B (`scale_pos_weight` vs none, early stopping).
  - *Verification*: Baseline benchmark table generated on the validation split.

- [ ] **M4: Leakage-Safe Entity-History Features (Set C) & Leakage Tests**
  - Engineer backward-looking entity history features (past 1h, 24h, 7d transaction counts; seconds since prior transaction; current amount vs prior mean/median; distinct addresses/emails/devices used; novel entity attribute flags).
  - Strict-inequality rule ($T_{prior} < T_{current}$); no past fraud labels used.
  - Passing the full 3-part leakage test suite before any model consumes Set C.
  - *Verification*: Full leakage test suite passes cleanly.

- [ ] **M5: Final Model Optimization, Test Evaluation & Experiments**
  - XGBoost optimization on Sets A, B, and C with validation-based early stopping.
  - Probability calibration (Platt / Isotonic) on validation set.
  - Single evaluation on the held-out Test set with 1000-replicate bootstrap 95% confidence intervals.
  - Execute and record experiments E1, E2, E3, and E4 into `reports/` artifacts.
  - *Verification*: Test evaluated once; bootstrap CIs computed; artifacts serialized.

- [ ] **M6: Model Explainability & Scored SQLite Storage**
  - Compute TreeExplainer SHAP values for final model (global feature-family importance + local top-5 contributions).
  - SQLite database storing scored test records, metadata, entity context, and top local SHAP attributions.
  - Explicit UI guidance: anonymised features indicate model behavior, not causal fraud confirmation.
  - *Verification*: SQLite database populated; queries return scored records and SHAP drivers.

- [ ] **M7: FastAPI Backend & API Test Suite**
  - FastAPI application serving endpoints from saved artifacts and SQLite.
  - Pydantic models for validation; standard 4xx/503 error handling; single model load at startup.
  - FastAPI `TestClient` suite covering all endpoints and edge cases.
  - *Verification*: API tests pass; endpoint responses match saved artifacts.

- [ ] **M8: Interactive Risk Review Dashboard**
  - Plain HTML/CSS/Vanilla JS frontend + Chart.js (no Node.js build step).
  - Pages: Overview, Model Comparison, Operating Point Explorer, Transaction Risk Explorer, Evaluation Integrity, Methods & Limitations.
  - All charts populated dynamically from FastAPI endpoints; strictly review-prioritization terminology.
  - *Verification*: Dashboard served by FastAPI at `/`; charts render real artifact data.

- [ ] **M9: Documentation & Reproducibility Verification**
  - Complete documentation: `README.md`, `docs/METHODOLOGY.md`, `docs/DATA_REPORT.md`, `docs/MODEL_CARD.md`, `docs/LIMITATIONS.md`.
  - Validate end-to-end execution of CLI / Makefile targets (`data`, `features`, `train`, `evaluate`, `test`, `serve`).
  - Final audit against all non-negotiable rules.
  - *Verification*: Clean environment reproduction check passes.

- [ ] **M10 (OPTIONAL - Gated on explicit user confirmation): Elliptic AML Graph Module**
  - Standalone package `src/fraudshield_elliptic/`.
  - Ingest Elliptic Bitcoin graph dataset; verify counts (203,769 nodes, 234,355 edges, 49 time steps; 4,545 illicit, 42,019 licit).
  - Temporal split by time steps (1–34 train/val, 35–49 test).
  - Tabular model vs Graph Neural Network (GCN/GraphSAGE); honest reporting of illicit-class PR-AUC and F1.
  - *Verification*: Only executed after M0–M9 are complete and explicit user approval is granted.
