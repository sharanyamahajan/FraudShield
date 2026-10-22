"""Leakage verification suite for Milestone 4 entity history features.

Guarantees enforced before ANY model consumes Set C:
(a) Brute-force equivalence: Vectorized vs. naive loop equivalence on hand-crafted frame with tied timestamps.
(b) Future perturbation: Altering/deleting rows strictly after time t leaves rows <= t 100% unchanged.
(c) Order invariance: Shuffling input rows yields identical features after realignment.
(d) Cold start: NaN for ratio, mean, distinct, seconds; prior_count == 0.
(e) Label-free: No reliance or presence of target label column isFraud.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fraudshield.features.history import (
    compute_entity_history_features,
    compute_entity_history_naive,
    C1_HISTORY_COLS,
    C2_HISTORY_COLS,
)


@pytest.fixture
def hand_crafted_df():
    """Small hand-built dataframe containing:
    - Multiple entities (card1 and composite (card1, addr1))
    - Tied timestamps within the same entity
    - Cold start transactions
    - Multi-transaction histories
    - Distinct and repeated emails and addresses
    """
    return pd.DataFrame({
        "TransactionID": [101, 102, 103, 104, 105, 106, 107, 108],
        "TransactionDT": [1000, 2000, 2000, 2000, 5000, 1500, 5000, 8000],
        "card1": [10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 10.0],
        "addr1": [1.0, 2.0, 3.0, 2.0, 1.0, 5.0, 5.0, 2.0],
        "TransactionAmt": [100.0, 200.0, 300.0, 400.0, 500.0, 50.0, 150.0, 250.0],
        "P_emaildomain": ["gmail.com", "yahoo.com", "hotmail.com", "yahoo.com", "gmail.com", "corp.com", None, "yahoo.com"],
        "isFraud": [0, 0, 1, 0, 1, 0, 0, 0],
    })


def test_leakage_suite_a_brute_force_equivalence(hand_crafted_df):
    """(a) Brute-force equivalence: compare vectorized implementation against slow naive loop.
    
    Must match numerically on all features including tied timestamps and cold start NaNs.
    """
    # 1. Primary key (card1)
    vec_c1 = compute_entity_history_features(hand_crafted_df, entity_key="card1", include_addr1=True)
    naive_c1 = compute_entity_history_naive(hand_crafted_df, entity_key="card1", include_addr1=True)

    assert list(vec_c1.columns) == C1_HISTORY_COLS
    assert list(naive_c1.columns) == C1_HISTORY_COLS
    assert_frame_equal(vec_c1, naive_c1, check_dtype=False, check_exact=False, rtol=1e-5, atol=1e-5)

    # 2. Secondary composite key (card1, addr1)
    vec_c2 = compute_entity_history_features(
        hand_crafted_df, entity_key=["card1", "addr1"], include_addr1=False, prefix="card1_addr1"
    )
    naive_c2 = compute_entity_history_naive(
        hand_crafted_df, entity_key=["card1", "addr1"], include_addr1=False, prefix="card1_addr1"
    )

    assert list(vec_c2.columns) == C2_HISTORY_COLS
    assert list(naive_c2.columns) == C2_HISTORY_COLS
    assert_frame_equal(vec_c2, naive_c2, check_dtype=False, check_exact=False, rtol=1e-5, atol=1e-5)


def test_leakage_suite_b_future_perturbation(hand_crafted_df):
    """(b) Future perturbation: delete or alter all rows after time t (including labels);
    features for rows at or before t must be strictly unchanged.
    """
    t_cutoff = 2000

    # Baseline features on full data
    base_c1 = compute_entity_history_features(hand_crafted_df, entity_key="card1", include_addr1=True)
    base_c2 = compute_entity_history_features(hand_crafted_df, entity_key=["card1", "addr1"], include_addr1=False)

    rows_le_t = hand_crafted_df["TransactionDT"] <= t_cutoff

    # 1. Test deletion of future rows (rows > t completely deleted)
    df_truncated = hand_crafted_df[rows_le_t].copy()
    trunc_c1 = compute_entity_history_features(df_truncated, entity_key="card1", include_addr1=True)
    trunc_c2 = compute_entity_history_features(df_truncated, entity_key=["card1", "addr1"], include_addr1=False)

    assert_frame_equal(base_c1.loc[rows_le_t], trunc_c1, check_dtype=False)
    assert_frame_equal(base_c2.loc[rows_le_t], trunc_c2, check_dtype=False)

    # 2. Test alteration of future rows (changing amounts, emails, addrs, and labels)
    df_altered = hand_crafted_df.copy()
    future_mask = df_altered["TransactionDT"] > t_cutoff
    df_altered.loc[future_mask, "TransactionAmt"] = df_altered.loc[future_mask, "TransactionAmt"] * 100.0 + 999.0
    df_altered.loc[future_mask, "P_emaildomain"] = "fraudster.net"
    df_altered.loc[future_mask, "addr1"] = 9999.0
    df_altered.loc[future_mask, "isFraud"] = 1 - df_altered.loc[future_mask, "isFraud"]

    alt_c1 = compute_entity_history_features(df_altered, entity_key="card1", include_addr1=True)
    alt_c2 = compute_entity_history_features(df_altered, entity_key=["card1", "addr1"], include_addr1=False)

    assert_frame_equal(base_c1.loc[rows_le_t], alt_c1.loc[rows_le_t], check_dtype=False)
    assert_frame_equal(base_c2.loc[rows_le_t], alt_c2.loc[rows_le_t], check_dtype=False)


def test_leakage_suite_c_order_invariance(hand_crafted_df):
    """(c) Order invariance: shuffling input row order gives identical features."""
    orig_c1 = compute_entity_history_features(hand_crafted_df, entity_key="card1", include_addr1=True)
    orig_c2 = compute_entity_history_features(hand_crafted_df, entity_key=["card1", "addr1"], include_addr1=False)

    # Shuffle the input rows with random permutation
    shuffled_df = hand_crafted_df.sample(frac=1.0, random_state=42)
    assert not shuffled_df.index.equals(hand_crafted_df.index)

    shuffled_c1 = compute_entity_history_features(shuffled_df, entity_key="card1", include_addr1=True)
    shuffled_c2 = compute_entity_history_features(shuffled_df, entity_key=["card1", "addr1"], include_addr1=False)

    # Output features re-aligned to original index must match exactly
    assert_frame_equal(orig_c1, shuffled_c1.loc[orig_c1.index], check_dtype=False)
    assert_frame_equal(orig_c2, shuffled_c2.loc[orig_c2.index], check_dtype=False)


def test_tied_timestamps_never_see_each_other():
    """Verify that rows with identical TransactionDT never see each other's data."""
    tied_df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [100, 100, 100],  # All tied!
        "card1": [500.0, 500.0, 500.0],
        "addr1": [10.0, 10.0, 10.0],
        "TransactionAmt": [25.0, 50.0, 75.0],
        "P_emaildomain": ["a.com", "b.com", "c.com"],
    })

    c1 = compute_entity_history_features(tied_df, entity_key="card1", include_addr1=True)

    # All three rows must have prior_count == 0, count windows == 0, and all other features NaN
    for i in range(3):
        assert c1.iloc[i]["card1_prior_count"] == 0
        assert c1.iloc[i]["card1_cnt_1h"] == 0
        assert c1.iloc[i]["card1_cnt_24h"] == 0
        assert c1.iloc[i]["card1_cnt_7d"] == 0
        assert np.isnan(c1.iloc[i]["card1_seconds_since_prev"])
        assert np.isnan(c1.iloc[i]["card1_prior_amt_mean"])
        assert np.isnan(c1.iloc[i]["card1_amt_to_prior_mean_ratio"])
        assert np.isnan(c1.iloc[i]["card1_distinct_p_emaildomain_cnt"])
        assert np.isnan(c1.iloc[i]["card1_distinct_addr1_cnt"])


def test_cold_start_rules_strictly_enforced():
    """Verify cold start: prior_count == 0 and NaNs for ratio, mean, distinct, seconds."""
    cold_df = pd.DataFrame({
        "TransactionID": [10],
        "TransactionDT": [500],
        "card1": [999.0],
        "addr1": [42.0],
        "TransactionAmt": [123.45],
        "P_emaildomain": ["xyz.org"],
    })

    c1 = compute_entity_history_features(cold_df, entity_key="card1", include_addr1=True)
    assert c1["card1_prior_count"].iloc[0] == 0
    assert c1["card1_cnt_1h"].iloc[0] == 0
    assert c1["card1_cnt_24h"].iloc[0] == 0
    assert c1["card1_cnt_7d"].iloc[0] == 0
    assert np.isnan(c1["card1_seconds_since_prev"].iloc[0])
    assert np.isnan(c1["card1_prior_amt_mean"].iloc[0])
    assert np.isnan(c1["card1_amt_to_prior_mean_ratio"].iloc[0])
    assert np.isnan(c1["card1_distinct_p_emaildomain_cnt"].iloc[0])
    assert np.isnan(c1["card1_distinct_addr1_cnt"].iloc[0])


def test_label_free_requirement(hand_crafted_df):
    """Verify that features can be computed completely without target column isFraud."""
    df_no_label = hand_crafted_df.drop(columns=["isFraud"])
    # Must succeed without error
    c1 = compute_entity_history_features(df_no_label, entity_key="card1", include_addr1=True)
    c2 = compute_entity_history_features(df_no_label, entity_key=["card1", "addr1"], include_addr1=False)
    assert len(c1) == len(df_no_label)
    assert len(c2) == len(df_no_label)
