"""Time-ordered split logic for FraudShield.

Enforces strict chronological train/val/test splits (70/15/15) by TransactionDT.
Crucially, never splits inside a group of tied timestamps: if a split boundary
lands within identical TransactionDT values, it advances to the next distinct
TransactionDT to preserve temporal integrity.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("fraudshield.splits.temporal")


def find_next_distinct_boundary(df: pd.DataFrame, target_idx: int, time_col: str = "TransactionDT") -> int:
    """If target_idx falls inside a group of tied timestamps, advance to the next distinct timestamp."""
    n = len(df)
    if target_idx <= 0 or target_idx >= n:
        return target_idx

    current_time = df[time_col].iloc[target_idx]
    prev_time = df[time_col].iloc[target_idx - 1]

    # If the boundary lands between identical timestamps, advance until the timestamp changes
    if current_time == prev_time:
        logger.info(
            "Boundary at index %d falls inside tied timestamp %d. Advancing to next distinct timestamp...",
            target_idx,
            current_time,
        )
        idx = target_idx
        while idx < n and df[time_col].iloc[idx] == current_time:
            idx += 1
        return idx

    return target_idx


def create_temporal_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    time_col: str = "TransactionDT",
    id_col: str = "TransactionID",
    target_col: str = "isFraud",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create strictly time-ordered, non-overlapping train, val, and test splits."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio})")

    # Ensure chronological order by (time_col, id_col)
    logger.info("Verifying chronological sort by ('%s', '%s')...", time_col, id_col)
    df_sorted = df.sort_values(by=[time_col, id_col]).reset_index(drop=True)
    n = len(df_sorted)

    # Initial candidate boundaries
    raw_train_end = int(round(n * train_ratio))
    raw_val_end = int(round(n * (train_ratio + val_ratio)))

    # Adjust boundaries so no group of tied timestamps is split
    train_end = find_next_distinct_boundary(df_sorted, raw_train_end, time_col)
    val_end = find_next_distinct_boundary(df_sorted, raw_val_end, time_col)

    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

    # Verification checks
    assert len(train_df) + len(val_df) + len(test_df) == n, "Total split rows do not equal input rows!"
    assert set(train_df[id_col]).isdisjoint(set(val_df[id_col])), "Train and Val share TransactionIDs!"
    assert set(val_df[id_col]).isdisjoint(set(test_df[id_col])), "Val and Test share TransactionIDs!"
    assert set(train_df[id_col]).isdisjoint(set(test_df[id_col])), "Train and Test share TransactionIDs!"

    # Assert strict non-overlapping time ranges
    assert train_df[time_col].max() < val_df[time_col].min(), (
        f"Train max time ({train_df[time_col].max()}) not strictly less than Val min time ({val_df[time_col].min()})"
    )
    assert val_df[time_col].max() < test_df[time_col].min(), (
        f"Val max time ({val_df[time_col].max()}) not strictly less than Test min time ({test_df[time_col].min()})"
    )

    def split_summary(split_name: str, split_data: pd.DataFrame) -> Dict[str, Any]:
        count = len(split_data)
        fraud_cnt = int(split_data[target_col].sum()) if target_col in split_data.columns else 0
        fraud_rate = float(fraud_cnt / count) if count > 0 else 0.0
        min_t = int(split_data[time_col].min()) if count > 0 else 0
        max_t = int(split_data[time_col].max()) if count > 0 else 0
        return {
            "split": split_name,
            "rows": count,
            "rows_pct": round(count / n * 100, 2),
            "fraud_count": fraud_cnt,
            "fraud_rate_pct": round(fraud_rate * 100, 4),
            "min_time": min_t,
            "max_time": max_t,
            "time_span_days": round((max_t - min_t) / 86400.0, 2),
        }

    summary = {
        "train": split_summary("train", train_df),
        "val": split_summary("val", val_df),
        "test": split_summary("test", test_df),
    }

    # Print summary table
    print("\n" + "=" * 80)
    print("TIME-ORDERED TEMPORAL SPLIT SUMMARY (70 / 15 / 15)")
    print("=" * 80)
    print(f"{'Split':<10} {'Rows':<12} {'Share (%)':<10} {'Fraud Cnt':<12} {'Fraud Rate (%)':<16} {'Time Range (s)':<25}")
    print("-" * 80)
    for s_name in ["train", "val", "test"]:
        s = summary[s_name]
        t_range = f"[{s['min_time']:,} - {s['max_time']:,}]"
        print(f"{s['split']:<10} {s['rows']:<12,d} {s['rows_pct']:<10.2f} {s['fraud_count']:<12,d} {s['fraud_rate_pct']:<16.4f} {t_range:<25}")
    print("=" * 80 + "\n")

    return train_df, val_df, test_df, summary
