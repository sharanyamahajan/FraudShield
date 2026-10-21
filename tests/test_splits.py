"""Unit tests for time-ordered temporal splits and tied-timestamp boundary rules."""

import pytest
import numpy as np
import pandas as pd
from fraudshield.splits.temporal import create_temporal_splits, find_next_distinct_boundary


def test_split_boundary_advances_on_tied_timestamps():
    """Verify that if a boundary index lands on a tied timestamp, it advances to the next distinct timestamp."""
    # Timestamps: 10, 20, 20, 20, 30, 40
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "TransactionDT": [10, 20, 20, 20, 30, 40],
        "isFraud": [0, 0, 1, 0, 0, 1],
    })
    # Target index 2 lands inside timestamp 20 (indices 1, 2, 3 have timestamp 20)
    adjusted_idx = find_next_distinct_boundary(df, target_idx=2, time_col="TransactionDT")
    assert adjusted_idx == 4
    assert df["TransactionDT"].iloc[adjusted_idx] == 30


def test_temporal_split_strict_ordering_and_disjointness():
    """Verify strict chronological ordering, ID disjointness, and row conservation."""
    # Create 100 rows with varying timestamps and duplicate ties
    np.random.seed(42)
    n = 100
    times = np.sort(np.random.choice(range(100, 300), size=n, replace=True))
    df = pd.DataFrame({
        "TransactionID": np.arange(1000, 1000 + n),
        "TransactionDT": times,
        "isFraud": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    })

    train_df, val_df, test_df, summary = create_temporal_splits(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    # 1. Row conservation
    assert len(train_df) + len(val_df) + len(test_df) == n

    # 2. Strict ID disjointness
    train_ids = set(train_df["TransactionID"])
    val_ids = set(val_df["TransactionID"])
    test_ids = set(test_df["TransactionID"])
    assert train_ids.isdisjoint(val_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(test_ids)

    # 3. Strict time ordering
    assert train_df["TransactionDT"].max() < val_df["TransactionDT"].min()
    assert val_df["TransactionDT"].max() < test_df["TransactionDT"].min()

    # 4. No tied timestamp split across boundaries
    assert set(train_df["TransactionDT"]).isdisjoint(set(val_df["TransactionDT"]))
    assert set(val_df["TransactionDT"]).isdisjoint(set(test_df["TransactionDT"]))


def test_invalid_split_ratios_raise_error():
    """Verify that split ratios not summing to 1.0 raise an error."""
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [10, 20, 30],
        "isFraud": [0, 0, 0],
    })
    with pytest.raises(ValueError, match="must sum to 1.0"):
        create_temporal_splits(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)
