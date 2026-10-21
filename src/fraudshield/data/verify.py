"""Verification and data profiling script for Milestone 1.

Inspects actual raw and merged IEEE-CIS data files, computes verified metrics,
evaluates proxy entity keys, and generates docs/DATA_REPORT.md and reports/data_summary.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import yaml
from fraudshield.data.loader import verify_time_properties

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudshield.data.verify")


def analyze_dataset(
    df: pd.DataFrame,
    identity_df_len: int,
    id_col: str = "TransactionID",
    time_col: str = "TransactionDT",
    target_col: str = "isFraud",
) -> Dict[str, Any]:
    """Calculate all required Milestone 1 facts directly on the actual loaded dataset."""
    total_rows = len(df)
    total_cols = df.shape[1]

    # Target stats
    fraud_counts = df[target_col].value_counts().to_dict()
    fraud_rate = float(df[target_col].mean())

    # Identity coverage
    # DeviceType or id_01 indicates identity record presence
    id_present_count = df["DeviceType"].notna().sum()
    id_coverage_rate = float(id_present_count / total_rows)

    # Time verification
    time_stats = verify_time_properties(df, time_col=time_col, id_col=id_col)

    # Duplicate IDs
    duplicate_ids = int(df[id_col].duplicated().sum())

    # Cardinality of key columns
    key_cols = [
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2",
        "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
    ]
    cardinalities = {}
    missingness = {}
    for col in key_cols:
        if col in df.columns:
            cardinalities[col] = int(df[col].nunique(dropna=True))
            missingness[col] = {
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            }
        else:
            cardinalities[col] = None
            missingness[col] = None

    # Full missingness summary
    missing_series = df.isna().sum()
    missing_pct_series = (missing_series / total_rows) * 100
    top_missing = (
        pd.DataFrame({"missing_count": missing_series, "missing_pct": missing_pct_series})
        .sort_values(by="missing_pct", ascending=False)
        .head(25)
        .to_dict(orient="index")
    )
    cols_zero_missing = int((missing_series == 0).sum())
    cols_over_50pct_missing = int((missing_pct_series > 50.0).sum())

    # Proxy Entity Key Analysis: card1 vs (card1, addr1)
    logger.info("Analyzing proxy entity key candidates...")
    c1_series = df["card1"]
    c1_nunique = int(c1_series.nunique())
    c1_counts = c1_series.value_counts()
    c1_single_txn = int((c1_counts == 1).sum())
    c1_multi_txn = int((c1_counts > 1).sum())
    c1_max_txn = int(c1_counts.max())
    c1_median_txn = float(c1_counts.median())
    c1_p95_txn = float(c1_counts.quantile(0.95))

    # Composite proxy key: card1 + addr1 (where addr1 is string, fillna 'missing')
    comp_series = df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(int).astype(str)
    comp_nunique = int(comp_series.nunique())
    comp_counts = comp_series.value_counts()
    comp_single_txn = int((comp_counts == 1).sum())
    comp_multi_txn = int((comp_counts > 1).sum())
    comp_max_txn = int(comp_counts.max())
    comp_median_txn = float(comp_counts.median())
    comp_p95_txn = float(comp_counts.quantile(0.95))

    entity_analysis = {
        "card1": {
            "unique_entities": c1_nunique,
            "single_transaction_entities": c1_single_txn,
            "single_transaction_pct": round(c1_single_txn / c1_nunique * 100, 2),
            "multi_transaction_entities": c1_multi_txn,
            "multi_transaction_pct": round(c1_multi_txn / c1_nunique * 100, 2),
            "max_txns_per_entity": c1_max_txn,
            "median_txns_per_entity": c1_median_txn,
            "p95_txns_per_entity": c1_p95_txn,
        },
        "card1_addr1_composite": {
            "unique_entities": comp_nunique,
            "single_transaction_entities": comp_single_txn,
            "single_transaction_pct": round(comp_single_txn / comp_nunique * 100, 2),
            "multi_transaction_entities": comp_multi_txn,
            "multi_transaction_pct": round(comp_multi_txn / comp_nunique * 100, 2),
            "max_txns_per_entity": comp_max_txn,
            "median_txns_per_entity": comp_median_txn,
            "p95_txns_per_entity": comp_p95_txn,
        },
        "decision": (
            "Primary proxy entity key is chosen as 'card1', with '(card1, addr1)' available as a composite proxy. "
            "Because no verified account or client ID is provided in IEEE-CIS, card1 acts strictly as an unverified "
            "payment instrument proxy. The cardinality shows meaningful recurrence (multi-transaction entities exist), "
            "enabling backward-looking velocity and aggregate features under strict T_prior < T_current filtering."
        ),
    }

    report_data = {
        "dataset_name": "IEEE-CIS Fraud Detection (Vesta)",
        "source": "Kaggle Competition (train_transaction.csv + train_identity.csv)",
        "total_rows": total_rows,
        "total_columns": total_cols,
        "identity_rows_raw": identity_df_len,
        "identity_matches_in_transaction": int(id_present_count),
        "identity_coverage_pct": round(id_coverage_rate * 100, 2),
        "duplicate_transaction_ids": duplicate_ids,
        "target_col": target_col,
        "fraud_count": int(fraud_counts.get(1, 0)),
        "non_fraud_count": int(fraud_counts.get(0, 0)),
        "fraud_rate_pct": round(fraud_rate * 100, 4),
        "time_stats": time_stats,
        "key_cardinalities": cardinalities,
        "key_missingness": missingness,
        "column_missingness_summary": {
            "cols_zero_missing": cols_zero_missing,
            "cols_over_50pct_missing": cols_over_50pct_missing,
            "top_25_missing": top_missing,
        },
        "entity_analysis": entity_analysis,
    }
    return report_data


def generate_markdown_report(data: Dict[str, Any], output_path: Path) -> None:
    """Generate docs/DATA_REPORT.md containing verified facts."""
    ts = data["time_stats"]
    ea = data["entity_analysis"]
    ms = data["column_missingness_summary"]

    lines = [
        "# IEEE-CIS Fraud Detection: Verified Data Report",
        "",
        "> [!IMPORTANT]",
        "> Every number in this report was computed directly on the actual files (`train_transaction.csv` and `train_identity.csv`) loaded in Milestone 1. No values are fabricated or assumed.",
        "",
        "## 1. Executive Summary & File Integrity",
        f"- **Primary Table**: `train_transaction.csv` ({data['total_rows']:,} rows)",
        f"- **Joined Secondary Table**: `train_identity.csv` ({data['identity_rows_raw']:,} rows)",
        f"- **Joined Table Rows**: {data['total_rows']:,}",
        f"- **Joined Table Columns**: {data['total_columns']}",
        f"- **Duplicate Transaction IDs**: {data['duplicate_transaction_ids']} (asserted 0 duplicates)",
        f"- **Identity Match Coverage**: {data['identity_matches_in_transaction']:,} transactions ({data['identity_coverage_pct']}%) matched an identity record via left join.",
        f"- **Target (`isFraud`) Class Balance**:",
        f"  - Non-Fraud (0): {data['non_fraud_count']:,} ({100.0 - data['fraud_rate_pct']:.2f}%)",
        f"  - Fraud (1): {data['fraud_count']:,} ({data['fraud_rate_pct']:.4f}%)",
        f"  - Total Fraud Rate: **{data['fraud_rate_pct']:.4f}%**",
        "",
        "## 2. Temporal Properties & Tie Handling (`TransactionDT`)",
        f"- **Minimum `TransactionDT`**: {ts['min_transaction_dt']:,} seconds",
        f"- **Maximum `TransactionDT`**: {ts['max_transaction_dt']:,} seconds",
        f"- **Total Time Span**: {ts['span_seconds']:,} seconds (~{ts['span_days']:.1f} days / ~{ts['span_days']/30.4:.1f} months)",
        f"- **Strict Monotonicity**: `{ts['is_strictly_monotonic']}`",
        f"- **Weak Monotonicity (Non-Decreasing)**: `{ts['is_weakly_monotonic']}`",
        f"- **Unique Timestamps**: {ts['unique_timestamps']:,}",
        f"- **Timestamp Ties (Duplicate Seconds)**: {ts['timestamp_tie_rows']:,} rows ({ts['tie_rate']*100:.2f}%) share a `TransactionDT` with at least one other transaction.",
        f"- **Maximum Concurrent Transactions in a Single Second**: {ts['max_ties_in_single_second']:,}",
        "",
        "### Tie-Breaking and Strict-Inequality Rule",
        "> [!NOTE]",
        "> Because multiple transactions occur within the exact same second (`TransactionDT` ties), any backward-looking rolling feature MUST strictly enforce:",
        "> $$\\text{TransactionDT}_{\\text{prior}} < \\text{TransactionDT}_{\\text{current}}$$",
        "> A transaction occurring at time $t$ will **never** include other transactions occurring at time $t$ in its history. This ensures that concurrent transactions cannot leak into each other regardless of execution order.",
        "> For deterministic processing, rows are sorted by `(TransactionDT, TransactionID)`.",
        "",
        "## 3. Cardinality of Key Columns",
        "| Column | Unique Values | Missing Count | Missing % |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for col, card in data["key_cardinalities"].items():
        miss = data["key_missingness"][col]
        m_cnt = f"{miss['missing_count']:,}" if miss else "N/A"
        m_pct = f"{miss['missing_pct']}%" if miss else "N/A"
        c_str = f"{card:,}" if card is not None else "N/A"
        lines.append(f"| `{col}` | {c_str} | {m_cnt} | {m_pct} |")

    lines.extend([
        "",
        "## 4. Missingness Profile",
        f"- **Columns with 0% Missingness**: {ms['cols_zero_missing']} of {data['total_columns']} columns",
        f"- **Columns with >50% Missingness**: {ms['cols_over_50pct_missing']} columns",
        "",
        "### Top 15 Missing Columns",
        "| Column | Missing Count | Missing % |",
        "| :--- | :--- | :--- |",
    ])

    for col, info in list(ms["top_25_missing"].items())[:15]:
        lines.append(f"| `{col}` | {info['missing_count']:,} | {info['missing_pct']:.2f}% |")

    lines.extend([
        "",
        "## 5. Proxy Entity Key Decision & Justification",
        "Because IEEE-CIS does not provide genuine account IDs, customer IDs, or authenticated identity numbers, "
        "any entity-level aggregation relies on an unverified proxy key.",
        "",
        "### Candidate Comparison",
        "| Metric | `card1` | `(card1, addr1)` Composite |",
        "| :--- | :--- | :--- |",
        f"| Unique Entity Keys | {ea['card1']['unique_entities']:,} | {ea['card1_addr1_composite']['unique_entities']:,} |",
        f"| Single-Transaction Entities | {ea['card1']['single_transaction_entities']:,} ({ea['card1']['single_transaction_pct']}%) | {ea['card1_addr1_composite']['single_transaction_entities']:,} ({ea['card1_addr1_composite']['single_transaction_pct']}%) |",
        f"| Multi-Transaction Entities | {ea['card1']['multi_transaction_entities']:,} ({ea['card1']['multi_transaction_pct']}%) | {ea['card1_addr1_composite']['multi_transaction_entities']:,} ({ea['card1_addr1_composite']['multi_transaction_pct']}%) |",
        f"| Max Transactions per Entity | {ea['card1']['max_txns_per_entity']:,} | {ea['card1_addr1_composite']['max_txns_per_entity']:,} |",
        f"| Median Transactions per Entity | {ea['card1']['median_txns_per_entity']} | {ea['card1_addr1_composite']['median_txns_per_entity']} |",
        f"| 95th Percentile Txns per Entity | {ea['card1']['p95_txns_per_entity']} | {ea['card1_addr1_composite']['p95_txns_per_entity']} |",
        "",
        "### Decision Rationale",
        f"- **Primary Proxy Key**: `card1`",
        f"- **Secondary Composite Proxy Key**: `(card1, addr1)`",
        "- **Justification**: Both keys provide substantial multi-transaction histories for backward-looking aggregations (e.g., velocity over 1h/24h/7d, amount compared to historical entity average). However, `card1` represents the payment card issuer/identifier directly, whereas adding `addr1` splits users who transact from multiple billing regions or during travel. Both options will be supported in configuration (`configs/config.yaml`), with `card1` serving as the baseline proxy entity.",
        "- **Caveat**: As noted in our project plan, proxy entity keys may be coarse or noisy. If Set C exhibits no gain over Set B in RQ1, this will be reported honestly as an experimental finding.",
        "",
        "## 6. Documented Caveats",
        "1. **`TransactionDT` Offset**: `TransactionDT` is a relative offset in seconds from an unrevealed reference epoch, not a true timestamp. Any 'hour-of-day' feature is an approximation and documented as such.",
        "2. **Anonymised Features**: Columns C1-C14, D1-D15, M1-M9, and V1-V339 are masked and anonymised by Vesta Corporation. Their underlying business definitions are unknown.",
        "3. **Unverified Label**: The exact criteria used to flag `isFraud` (e.g. chargeback window, customer dispute, issuer return) are not documented in the dataset files. `isFraud` is treated strictly as provided ground truth.",
        "4. **No Account-to-Account Graph**: This dataset contains transaction records and device metadata, but no recipient/merchant account graph. No fraud-ring or mule-network assumptions are made.",
        "5. **Dataset Licensing**: The dataset was published for an open Kaggle research competition under competition rules. Its commercial licence terms are unverified.",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Generated %s successfully.", output_path)
