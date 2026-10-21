# FraudShield Methodology

## 1. Split Design & Temporal Integrity

In transaction fraud detection, real-world systems operate strictly forward in time. Standard $k$-fold cross-validation or random splits introduce massive lookahead leakage by allowing models to condition on future transaction velocity, evolving fraud campaigns, and future macro trends to predict past transactions.

### 1.1 Primary Time-Ordered Split (70 / 15 / 15)
- All transactions are sorted chronologically by `(TransactionDT, TransactionID)`.
- **Train Split**: First ~70% of chronological data.
- **Validation Split**: Next ~15% of chronological data. Used for model selection, early stopping, hyperparameter tuning, threshold determination, and probability calibration.
- **Test Split**: Final ~15% of chronological data. Held out and evaluated **strictly once** to provide unbiased performance estimates.

### 1.2 Tied Timestamp Boundary Rule
In the IEEE-CIS dataset, 33,932 transactions (5.75%) share a timestamp with at least one other transaction (`TransactionDT` ties), with up to 8 transactions in a single second.
To prevent split leakage:
- **Rule**: A split boundary **never** divides transactions sharing the same timestamp.
- **Implementation**: If a candidate 70% or 85% split index lands inside a group of identical timestamps, the boundary automatically advances to the next distinct `TransactionDT`.
- **Result**: The time ranges of Train, Validation, and Test are strictly non-overlapping:
  $$\max(\text{TransactionDT}_{\text{train}}) < \min(\text{TransactionDT}_{\text{val}}) \le \max(\text{TransactionDT}_{\text{val}}) < \min(\text{TransactionDT}_{\text{test}})$$

### 1.3 Rolling-Origin (Expanding Window) Backtest
To assess performance variance across different time horizons, a 4-fold expanding window backtest is implemented:
- Fold 1: Train on Block 1 $\rightarrow$ Test on Block 2
- Fold 2: Train on Blocks 1–2 $\rightarrow$ Test on Block 3
- Fold 3: Train on Blocks 1–3 $\rightarrow$ Test on Block 4

### 1.4 Random Split Reference (RQ2 Only)
A standard random 70/15/15 split is run solely as a reference benchmark to quantify the empirical optimism of random splitting compared to the true temporal split (answering Research Question 2). It is explicitly labeled as a leaky, optimistic reference.

---

## 2. Feature Sets (Ablation Ladder)

### 2.1 Set A: Transaction-Only Baseline
Set A relies solely on raw payment attributes directly present on the transaction slip, excluding all masked columns and identity tables:
- **Amounts**: `TransactionAmt`, `log_amount` ($\log(1 + \text{amt})$), and `amount_cents` ($\text{amt} \pmod 1$).
- **Instrument Metadata**: `card1`, `card2`, `card3`, `card4`, `card5`, `card6`.
- **Geographic & Network Attributes**: `addr1`, `addr2`, `dist1`, `dist2`, `P_emaildomain`, `R_emaildomain`.
- **Diurnal Cycle Proxy**: `hour_of_day_proxy` = `(TransactionDT // 3600) % 24`.
  > [!WARNING]
  > `TransactionDT` is an offset in seconds from an unrevealed reference epoch, not a true calendar timestamp. Any diurnal feature is an approximation and must be treated as such.
- **Strict Exclusion**: All V, C, D, M, and identity columns are strictly excluded and verified via automated unit tests.

### 2.2 Set B: Raw + Anonymised + Identity
Set B adds all proprietary engineered attributes supplied in the dataset:
- **C1–C14**: Anonymised counting attributes (frequencies, velocity counts).
- **D1–D15**: Anonymised timedelta attributes (days since prior events).
- **M1–M9**: Anonymised match flags (names on card vs address, etc.).
- **V1–V339**: Vesta-engineered rich transaction attributes.
- **Identity Features**: `DeviceType`, `DeviceInfo`, and `id_01` through `id_38`.

> [!NOTE]
> Set B already contains Vesta's proprietary C (count) and D (timedelta) columns. This means Set B already incorporates masked, vendor-engineered historical signals before our explicit, leakage-safe backward-looking entity history features (Set C) are introduced in Milestone 4.

### 2.3 Set C: Backward-Looking Entity History (Milestone 4)
Set C layers engineered behavioral velocity on top of Set B using proxy entity keys (`card1` primary, `card1+addr1` secondary).
- Evaluated strictly with $T_{\text{prior}} < T_{\text{current}}$.
- No past fraud labels are used in history (avoiding label-delay leakage).

---

## 3. Preprocessing & Leakage Controls

### 3.1 Strict Train-Only Fitting
- All imputers (e.g. median calculation for numeric features), standard scalers, and categorical encoders are fitted **strictly on the Train split**.
- Validation and Test splits are transformed using the frozen parameters learned on Train.
- Automated tests verify that transformer parameters remain 100% identical whether or not validation/test rows are observed.

### 3.2 Model Evaluation Caveats & Hypotheses
- **XGBoost Early Stopping Optimism**: XGBoost uses early stopping on the validation split (`eval_metric="aucpr"`), which means validation scores are slightly optimistic due to stopping rule selection. Unbiased generalization will be confirmed once on the held-out test split in Milestone 5.
- **Isolation Forest Performance**: Unsupervised Isolation Forest yields substantially lower PR-AUC than supervised baselines. Our working hypothesis is that in tabular transaction fraud, fraudulent records often closely mimic the multivariate distribution of legitimate transactions rather than appearing as isolated geometric outliers. This is a hypothesis to interpret the observed validation metrics, not a confirmed causal claim.

### 3.3 Unseen Category & Missingness Handling
- **Categoricals**: Known categories in Train are mapped to discrete integers. Unseen categories appearing in Validation or Test are mapped to a reserved missing token (`-1`). Missing categorical values are mapped to `-2`.
- **Missing Indicators**: In linear models, missing value indicators (`_isna`) are generated alongside median-imputed values. In tree models, numeric NaNs are preserved for native directional split routing in XGBoost.

---

## 4. Exclusion Rationale: Why No GNNs or SMOTE

1. **Why No GNNs in Core Pipeline**:
   - The IEEE-CIS dataset does not contain an account-to-account transaction graph (there is no recipient/payee account ID or merchant routing node).
   - Building a bipartite graph based solely on coarse proxy keys (such as `card1` or `DeviceInfo`) creates dense, artificial pseudo-edges that obscure rather than clarify fraud patterns.
   - Graph neural networks introduce substantial computational overhead and potential transductive temporal leakage if not partitioned with extreme care. (Graph structure is evaluated strictly as an optional study on the Elliptic Bitcoin dataset in Milestone 10).
2. **Why No SMOTE or Synthetic Resampling**:
   - Synthetic oversampling methods (SMOTE, ADASYN) interpolate synthetic transactions in high-dimensional feature space.
   - In financial fraud, fraudulent transactions are not smoothly distributed clusters; synthetic points frequently lie in legitimate transaction space, creating false decision boundaries.
   - Furthermore, synthetic data destroys the true operational base rate (3.5%), distorting probability calibration and review budget prioritization. Class imbalance is instead handled properly through cost-sensitive learning (`scale_pos_weight`), rank-based metrics (PR-AUC), and operational review budgets.
