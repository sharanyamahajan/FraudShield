# IEEE-CIS Fraud Detection: Verified Data Report

> [!IMPORTANT]
> Every number in this report was computed directly on the actual files (`train_transaction.csv` and `train_identity.csv`) loaded in Milestone 1. No values are fabricated or assumed.

## 1. Executive Summary & File Integrity
- **Primary Table**: `train_transaction.csv` (590,540 rows)
- **Joined Secondary Table**: `train_identity.csv` (144,233 rows)
- **Joined Table Rows**: 590,540
- **Joined Table Columns**: 434
- **Duplicate Transaction IDs**: 0 (asserted 0 duplicates)
- **Identity Match Coverage**: 140,810 transactions (23.84%) matched an identity record via left join.
- **Target (`isFraud`) Class Balance**:
  - Non-Fraud (0): 569,877 (96.50%)
  - Fraud (1): 20,663 (3.4990%)
  - Total Fraud Rate: **3.4990%**

## 2. Temporal Properties & Tie Handling (`TransactionDT`)
- **Minimum `TransactionDT`**: 86,400 seconds
- **Maximum `TransactionDT`**: 15,811,131 seconds
- **Total Time Span**: 15,724,731 seconds (~182.0 days / ~6.0 months)
- **Strict Monotonicity**: `False`
- **Weak Monotonicity (Non-Decreasing)**: `True`
- **Unique Timestamps**: 573,349
- **Distinct Timestamps Occurring > 1 Times**: **16,741**
- **Rows Involved in a Tie**: **33,932** (5.75% of total dataset)
- **Surplus Timestamp Rows** (Rows minus unique timestamps): **17,191** (2.91%)
- **Maximum Concurrent Transactions in a Single Second**: **8**

### Tie-Breaking and Strict-Inequality Rule
> [!NOTE]
> Because 33,932 transactions occur within tied timestamps, any backward-looking rolling feature MUST strictly enforce:
> $$\text{TransactionDT}_{\text{prior}} < \text{TransactionDT}_{\text{current}}$$
> A transaction occurring at time $t$ will **never** include other transactions occurring at time $t$ in its history. This ensures that concurrent transactions cannot leak into each other regardless of execution order.
> Furthermore, in time-ordered train/val/test splits, split boundaries never split inside a group of tied timestamps; the boundary is moved to the next distinct `TransactionDT`.
> For deterministic processing, rows are sorted by `(TransactionDT, TransactionID)`.

## 3. Cardinality of Key Columns
| Column | Unique Values | Missing Count | Missing % |
| :--- | :--- | :--- | :--- |
| `card1` | 13,553 | 0 | 0.0% |
| `card2` | 500 | 8,933 | 1.51% |
| `card3` | 114 | 1,565 | 0.27% |
| `card4` | 4 | 1,577 | 0.27% |
| `card5` | 119 | 4,259 | 0.72% |
| `card6` | 4 | 1,571 | 0.27% |
| `addr1` | 332 | 65,706 | 11.13% |
| `addr2` | 74 | 65,706 | 11.13% |
| `P_emaildomain` | 59 | 94,456 | 15.99% |
| `R_emaildomain` | 60 | 453,249 | 76.75% |
| `DeviceType` | 2 | 449,730 | 76.16% |
| `DeviceInfo` | 1,786 | 471,874 | 79.91% |

## 4. Missingness Profile
- **Columns with 0% Missingness**: 20 of 434 columns
- **Columns with >50% Missingness**: 214 columns

### Top 15 Missing Columns
| Column | Missing Count | Missing % |
| :--- | :--- | :--- |
| `id_24` | 585,793 | 99.20% |
| `id_25` | 585,408 | 99.13% |
| `id_07` | 585,385 | 99.13% |
| `id_08` | 585,385 | 99.13% |
| `id_21` | 585,381 | 99.13% |
| `id_26` | 585,377 | 99.13% |
| `id_27` | 585,371 | 99.12% |
| `id_23` | 585,371 | 99.12% |
| `id_22` | 585,371 | 99.12% |
| `dist2` | 552,913 | 93.63% |
| `D7` | 551,623 | 93.41% |
| `id_18` | 545,427 | 92.36% |
| `D13` | 528,588 | 89.51% |
| `D14` | 528,353 | 89.47% |
| `D12` | 525,823 | 89.04% |

## 5. Proxy Entity Key Decision & Justification
Because IEEE-CIS does not provide genuine account IDs, customer IDs, or authenticated identity numbers, any entity-level aggregation relies on an unverified proxy key.

### Definition and Nature of `card1`
`card1` is an anonymised column whose meaning is undocumented. Its largest group has 14,932 transactions, so it is a coarse proxy that may aggregate many cardholders; entity-history features built on it describe group behaviour, not necessarily one person's.

### Entity Key Policy (Fixed)
- **Primary Proxy Key**: `card1`
- **Secondary Proxy Key**: `(card1, addr1)` composite
- No other keys will be added or tested without approval. Both will be reported in Milestone 4.

### Candidate Comparison and Transaction Coverage
| Metric | `card1` (Primary) | `(card1, addr1)` Composite (Secondary) |
| :--- | :--- | :--- |
| Unique Entity Keys | 13,553 | 39,974 |
| Single-Transaction Entities | 3,444 (25.41%) | 15,752 (39.41%) |
| Multi-Transaction Entities | 10,109 (74.59%) | 24,222 (60.59%) |
| **Share of All Transactions in Entities with $\ge 10$ Txns** | **562,488 (95.25%)** | **510,882 (86.51%)** |

### Entity Transaction Count Percentiles
| Percentile | `card1` Transactions per Entity | `(card1, addr1)` Transactions per Entity |
| :--- | :--- | :--- |
| **Median (50th)** | 4.0 | 2.0 |
| **75th** | 14.0 | 7.0 |
| **90th** | 46.0 | 22.0 |
| **95th** | 107.0 | 45.0 |
| **99th** | 786.4 | 201.3 |
| **Max** | 14,932 | 9,928 |

### Decision Rationale
1. **High Volume Coverage**: While 25.41% of unique `card1` entities appear only once, **95.25% of all transactions** belong to entities with at least 10 historical events. This confirms that backward-looking rolling features will have prior history for the vast majority of scored transactions.
2. **Coarse Proxy Awareness**: Because `card1` represents group behavior rather than a single individual, features will capture shared card-bin / portfolio velocity. If Set C exhibits no gain over Set B in RQ1, this will be reported honestly as an experimental finding.

## 6. Documented Caveats
1. **`TransactionDT` Offset**: `TransactionDT` is a relative offset in seconds from an unrevealed reference epoch, not a true timestamp. Any 'hour-of-day' feature is an approximation and documented as such.
2. **Anonymised Features**: Columns C1-C14, D1-D15, M1-M9, and V1-V339 are masked and anonymised by Vesta Corporation. Their underlying business definitions are unknown.
3. **Unverified Label**: The exact criteria used to flag `isFraud` (e.g. chargeback window, customer dispute, issuer return) are not documented in the dataset files. `isFraud` is treated strictly as provided ground truth. Any claim about what it represents (e.g. chargebacks) is an assumption and must not appear in results as fact.
4. **No Account-to-Account Graph**: This dataset contains transaction records and device metadata, but no recipient/merchant account graph. No fraud-ring or mule-network assumptions are made.
5. **Dataset Licensing**: The dataset was published for an open Kaggle research competition under competition rules. Its commercial licence terms are unverified. Link: [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection).