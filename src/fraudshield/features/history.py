"""Backward-looking entity-history feature engineering engine for Milestone 4.

Builds leakage-safe rolling features using strictly backward-looking time windows:
- Primary key: 'card1' (Set C1)
- Secondary key: ('card1', 'addr1') composite (Set C2)

Rules strictly enforced:
1. Timeline ordering: Full timeline sorted chronologically by (TransactionDT, TransactionID).
2. Strict inequality: Row i uses only rows j with TransactionDT_j < TransactionDT_i.
   Tied timestamps (TransactionDT_j == TransactionDT_i) never see each other.
3. Label-free: Features never use prior isFraud or any target labels.
4. Fixed feature list:
   - prior transaction count (all-time, 1h, 24h, 7d)
   - seconds since key's previous transaction
   - prior mean amount and ratio of current amount to it
   - distinct prior P_emaildomain count (and distinct prior addr1 count for card1 key only)
5. Cold start: NaN for feature plus a prior_count column.
"""

import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger("fraudshield.features.history")

C1_HISTORY_COLS = [
    "card1_prior_count",
    "card1_cnt_1h",
    "card1_cnt_24h",
    "card1_cnt_7d",
    "card1_seconds_since_prev",
    "card1_prior_amt_mean",
    "card1_amt_to_prior_mean_ratio",
    "card1_distinct_p_emaildomain_cnt",
    "card1_distinct_addr1_cnt",
]

C2_HISTORY_COLS = [
    "card1_addr1_prior_count",
    "card1_addr1_cnt_1h",
    "card1_addr1_cnt_24h",
    "card1_addr1_cnt_7d",
    "card1_addr1_seconds_since_prev",
    "card1_addr1_prior_amt_mean",
    "card1_addr1_amt_to_prior_mean_ratio",
    "card1_addr1_distinct_p_emaildomain_cnt",
]


def _build_key_series(df: pd.DataFrame, entity_key: Union[str, Tuple[str, ...], List[str]]) -> pd.Series:
    """Build a deterministic hashable series representing the entity key."""
    if isinstance(entity_key, (list, tuple)):
        # Composite key e.g. ('card1', 'addr1')
        k1 = df[entity_key[0]].astype(str)
        k2 = df[entity_key[1]].fillna(-1.0).astype(str)
        return k1 + "_" + k2
    return df[entity_key].astype(str)


def compute_entity_history_features(
    df: pd.DataFrame,
    entity_key: Union[str, Tuple[str, ...], List[str]] = "card1",
    include_addr1: bool = True,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Vectorized calculation of backward-looking entity history features.
    
    Guarantees:
    - Chronological timeline sorting by (TransactionDT, TransactionID).
    - Strict inequality: T_prior < T_current (tied timestamps never see each other).
    - Label-free: No target columns read or used.
    - Cold start: Returns NaN for ratio/mean/distinct/seconds plus prior_count == 0.
    - Order invariance: Returns features aligned to the original index of df.
    """
    if prefix is None:
        prefix = entity_key if isinstance(entity_key, str) else "card1_addr1"

    original_index = df.index
    # 1. Sort strictly by (TransactionDT, TransactionID)
    sort_cols = ["TransactionDT", "TransactionID"]
    df_sorted = df.sort_values(by=sort_cols).copy()
    n = len(df_sorted)

    key_series = _build_key_series(df_sorted, entity_key)
    df_sorted["_entity_key"] = key_series.values

    # Pre-allocate output feature arrays
    prior_count_arr = np.zeros(n, dtype=np.int32)
    cnt_1h_arr = np.zeros(n, dtype=np.int32)
    cnt_24h_arr = np.zeros(n, dtype=np.int32)
    cnt_7d_arr = np.zeros(n, dtype=np.int32)
    seconds_prev_arr = np.full(n, np.nan, dtype=np.float32)
    prior_mean_amt_arr = np.full(n, np.nan, dtype=np.float32)
    amt_ratio_arr = np.full(n, np.nan, dtype=np.float32)
    distinct_email_arr = np.full(n, np.nan, dtype=np.float32)
    distinct_addr_arr = np.full(n, np.nan, dtype=np.float32) if include_addr1 else None

    # Group by entity key; because df_sorted is already chronologically sorted,
    # each group's rows are in strictly chronological order.
    grouped = df_sorted.groupby("_entity_key", sort=False)
    group_indices_map = grouped.indices

    for key_val, pos in group_indices_map.items():
        m = len(pos)
        if m == 0:
            continue

        # Extract numpy slices for this entity
        dt_arr = df_sorted["TransactionDT"].values[pos]
        amt_arr = df_sorted["TransactionAmt"].values[pos].astype(np.float64)
        email_arr = df_sorted["P_emaildomain"].values[pos] if "P_emaildomain" in df_sorted.columns else None
        addr_arr = df_sorted["addr1"].values[pos] if (include_addr1 and "addr1" in df_sorted.columns) else None

        # Strict inequality search:
        # np.searchsorted(dt_arr, dt_arr, side='left') finds the first index with timestamp >= current.
        # All items strictly before this index have timestamp < current!
        # If timestamps are tied, both get the same right_idx, completely hiding each other.
        right_idx = np.searchsorted(dt_arr, dt_arr, side="left")
        left_1h = np.searchsorted(dt_arr, dt_arr - 3600, side="left")
        left_24h = np.searchsorted(dt_arr, dt_arr - 86400, side="left")
        left_7d = np.searchsorted(dt_arr, dt_arr - 604800, side="left")

        # Counts
        prior_cnt = right_idx
        c_1h = right_idx - left_1h
        c_24h = right_idx - left_24h
        c_7d = right_idx - left_7d

        prior_count_arr[pos] = prior_cnt
        cnt_1h_arr[pos] = c_1h
        cnt_24h_arr[pos] = c_24h
        cnt_7d_arr[pos] = c_7d

        has_prior = right_idx > 0
        if not np.any(has_prior):
            # All transactions in group are cold start (e.g. m=1 or all tied at same t)
            continue

        pos_has_prior = pos[has_prior]
        valid_right = right_idx[has_prior]

        # Seconds since previous transaction
        prev_idx = valid_right - 1
        seconds_prev_arr[pos_has_prior] = (dt_arr[has_prior] - dt_arr[prev_idx]).astype(np.float32)

        # Cumulative mean amount
        cumsum_amt = np.zeros(m + 1, dtype=np.float64)
        cumsum_amt[1:] = np.cumsum(amt_arr)
        prior_sums = cumsum_amt[valid_right]
        means = prior_sums / valid_right
        prior_mean_amt_arr[pos_has_prior] = means.astype(np.float32)

        # Ratio of current amount to prior mean amount
        cur_amt = amt_arr[has_prior]
        valid_mean_mask = means > 0
        ratios = np.full(len(valid_right), np.nan, dtype=np.float32)
        ratios[valid_mean_mask] = (cur_amt[valid_mean_mask] / means[valid_mean_mask]).astype(np.float32)
        amt_ratio_arr[pos_has_prior] = ratios

        # Distinct prior P_emaildomain count
        if email_arr is not None:
            cum_distinct_email = np.zeros(m + 1, dtype=np.float32)
            seen_emails = set()
            for j in range(m):
                val = email_arr[j]
                if pd.notna(val):
                    seen_emails.add(val)
                cum_distinct_email[j + 1] = len(seen_emails)
            distinct_email_arr[pos_has_prior] = cum_distinct_email[valid_right]

        # Distinct prior addr1 count (card1 only)
        if include_addr1 and addr_arr is not None:
            cum_distinct_addr = np.zeros(m + 1, dtype=np.float32)
            seen_addrs = set()
            for j in range(m):
                val = addr_arr[j]
                if pd.notna(val):
                    seen_addrs.add(val)
                cum_distinct_addr[j + 1] = len(seen_addrs)
            distinct_addr_arr[pos_has_prior] = cum_distinct_addr[valid_right]

    data = {
        f"{prefix}_prior_count": prior_count_arr,
        f"{prefix}_cnt_1h": cnt_1h_arr,
        f"{prefix}_cnt_24h": cnt_24h_arr,
        f"{prefix}_cnt_7d": cnt_7d_arr,
        f"{prefix}_seconds_since_prev": seconds_prev_arr,
        f"{prefix}_prior_amt_mean": prior_mean_amt_arr,
        f"{prefix}_amt_to_prior_mean_ratio": amt_ratio_arr,
        f"{prefix}_distinct_p_emaildomain_cnt": distinct_email_arr,
    }
    if include_addr1:
        data[f"{prefix}_distinct_addr1_cnt"] = distinct_addr_arr

    res_df = pd.DataFrame(data, index=df_sorted.index)
    # Re-align to original input dataframe index (Order Invariance guarantee)
    return res_df.loc[original_index].copy()


def compute_entity_history_naive(
    df: pd.DataFrame,
    entity_key: Union[str, Tuple[str, ...], List[str]] = "card1",
    include_addr1: bool = True,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Slow, naive brute-force loop for history feature calculation.
    
    Used strictly in unit testing to prove equivalence with the vectorized implementation.
    """
    if prefix is None:
        prefix = entity_key if isinstance(entity_key, str) else "card1_addr1"

    original_index = df.index
    df_sorted = df.sort_values(by=["TransactionDT", "TransactionID"]).copy()
    n = len(df_sorted)

    key_series = _build_key_series(df_sorted, entity_key)
    df_sorted["_entity_key"] = key_series.values

    prior_counts = np.zeros(n, dtype=np.int32)
    cnt_1h = np.zeros(n, dtype=np.int32)
    cnt_24h = np.zeros(n, dtype=np.int32)
    cnt_7d = np.zeros(n, dtype=np.int32)
    seconds_prev = np.full(n, np.nan, dtype=np.float32)
    prior_mean_amt = np.full(n, np.nan, dtype=np.float32)
    amt_ratio = np.full(n, np.nan, dtype=np.float32)
    distinct_email = np.full(n, np.nan, dtype=np.float32)
    distinct_addr = np.full(n, np.nan, dtype=np.float32) if include_addr1 else None

    for i in range(n):
        row_i = df_sorted.iloc[i]
        key_i = row_i["_entity_key"]
        t_i = row_i["TransactionDT"]
        amt_i = row_i["TransactionAmt"]

        # Naive linear scan for strictly prior rows:
        # Must match entity key AND TransactionDT < t_i strictly
        priors = [
            df_sorted.iloc[j]
            for j in range(n)
            if df_sorted.iloc[j]["_entity_key"] == key_i and df_sorted.iloc[j]["TransactionDT"] < t_i
        ]

        p_cnt = len(priors)
        prior_counts[i] = p_cnt
        if p_cnt == 0:
            continue

        cnt_1h[i] = sum(1 for p in priors if t_i - 3600 <= p["TransactionDT"] < t_i)
        cnt_24h[i] = sum(1 for p in priors if t_i - 86400 <= p["TransactionDT"] < t_i)
        cnt_7d[i] = sum(1 for p in priors if t_i - 604800 <= p["TransactionDT"] < t_i)

        max_prev_t = max(p["TransactionDT"] for p in priors)
        seconds_prev[i] = float(t_i - max_prev_t)

        mean_amt = sum(p["TransactionAmt"] for p in priors) / p_cnt
        prior_mean_amt[i] = float(mean_amt)
        if mean_amt > 0:
            amt_ratio[i] = float(amt_i / mean_amt)

        emails = set(p["P_emaildomain"] for p in priors if "P_emaildomain" in p and pd.notna(p["P_emaildomain"]))
        distinct_email[i] = float(len(emails))

        if include_addr1 and "addr1" in df_sorted.columns:
            addrs = set(p["addr1"] for p in priors if pd.notna(p["addr1"]))
            distinct_addr[i] = float(len(addrs))

    data = {
        f"{prefix}_prior_count": prior_counts,
        f"{prefix}_cnt_1h": cnt_1h,
        f"{prefix}_cnt_24h": cnt_24h,
        f"{prefix}_cnt_7d": cnt_7d,
        f"{prefix}_seconds_since_prev": seconds_prev,
        f"{prefix}_prior_amt_mean": prior_mean_amt,
        f"{prefix}_amt_to_prior_mean_ratio": amt_ratio,
        f"{prefix}_distinct_p_emaildomain_cnt": distinct_email,
    }
    if include_addr1:
        data[f"{prefix}_distinct_addr1_cnt"] = distinct_addr

    res_df = pd.DataFrame(data, index=df_sorted.index)
    return res_df.loc[original_index].copy()


def compute_reporting_statistics(
    full_df: pd.DataFrame,
    c1_feats: pd.DataFrame,
    c2_feats: pd.DataFrame,
    train_ids: set,
    val_ids: set,
    test_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Compute reporting statistics required by Milestone 4:
    
    1. Share of ROWS with at least 10 prior transactions in their entity (per key and per split).
    2. Card1 percentile table across unique entities (median, 75th, 90th, 95th, 99th, max).
    """
    txn_ids = full_df["TransactionID"].values
    c1_prior = c1_feats["card1_prior_count"].values
    c2_prior = c2_feats["card1_addr1_prior_count"].values

    splits = {
        "train": np.isin(txn_ids, list(train_ids)),
        "val": np.isin(txn_ids, list(val_ids)),
    }
    if test_ids is not None:
        splits["test"] = np.isin(txn_ids, list(test_ids))

    row_share_ge10 = {}
    for split_name, mask in splits.items():
        total_rows = int(np.sum(mask))
        c1_ge10 = int(np.sum((c1_prior >= 10) & mask))
        c2_ge10 = int(np.sum((c2_prior >= 10) & mask))

        row_share_ge10[split_name] = {
            "total_rows": total_rows,
            "card1_rows_ge_10": c1_ge10,
            "card1_share_pct": round(c1_ge10 / total_rows * 100, 2) if total_rows > 0 else 0.0,
            "card1_addr1_rows_ge_10": c2_ge10,
            "card1_addr1_share_pct": round(c2_ge10 / total_rows * 100, 2) if total_rows > 0 else 0.0,
        }

    # Card1 percentile table across unique entities
    card1_counts = full_df["card1"].value_counts()
    card1_percentiles = {
        "unique_entities": int(len(card1_counts)),
        "median": float(card1_counts.quantile(0.50)),
        "p75": float(card1_counts.quantile(0.75)),
        "p90": float(card1_counts.quantile(0.90)),
        "p95": float(card1_counts.quantile(0.95)),
        "p99": float(card1_counts.quantile(0.99)),
        "max": int(card1_counts.max()),
    }

    return {
        "row_share_with_ge_10_prior_txns": row_share_ge10,
        "card1_percentiles": card1_percentiles,
    }
