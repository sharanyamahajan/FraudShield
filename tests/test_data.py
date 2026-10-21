"""Unit tests for data ingestion, joins, and time-ordered verification."""

import pytest
import numpy as np
import pandas as pd
from fraudshield.data.loader import downcast_dtypes, join_and_process, verify_time_properties


@pytest.fixture
def sample_txn_and_identity():
    """Create a minimal, synthetic, hand-crafted dataset for data ingestion tests."""
    df_txn = pd.DataFrame({
        "TransactionID": [1001, 1002, 1003, 1004],
        "TransactionDT": [100, 200, 200, 300],  # note tie at 200
        "TransactionAmt": [50.0, 120.5, 99.9, 300.0],
        "card1": [101, 101, 102, 101],
        "isFraud": [0, 1, 0, 0],
    })
    # Identity table covers only a subset of transactions
    df_id = pd.DataFrame({
        "TransactionID": [1002, 1004],
        "DeviceType": ["mobile", "desktop"],
        "DeviceInfo": ["iOS", "Windows"],
    })
    return df_txn, df_id


def test_left_join_preserves_all_transactions(sample_txn_and_identity):
    """Assert left join keeps all transaction rows and matches identity where present."""
    df_txn, df_id = sample_txn_and_identity
    merged = join_and_process(df_txn, df_id)

    assert len(merged) == len(df_txn)
    assert set(merged["TransactionID"]) == {1001, 1002, 1003, 1004}
    
    # Check that unmatched rows have NaN for identity columns
    row_1001 = merged[merged["TransactionID"] == 1001].iloc[0]
    assert pd.isna(row_1001["DeviceType"])

    # Check matched rows
    row_1002 = merged[merged["TransactionID"] == 1002].iloc[0]
    assert row_1002["DeviceType"] == "mobile"
    assert row_1002["DeviceInfo"] == "iOS"


def test_duplicate_transaction_ids_raise_error():
    """Assert duplicate TransactionIDs in transaction table trigger an error."""
    df_txn_dup = pd.DataFrame({
        "TransactionID": [1001, 1001],
        "TransactionDT": [100, 200],
        "TransactionAmt": [50.0, 60.0],
        "isFraud": [0, 0],
    })
    df_id = pd.DataFrame({"TransactionID": [1001], "DeviceType": ["mobile"]})

    with pytest.raises(ValueError, match="duplicate"):
        join_and_process(df_txn_dup, df_id)


def test_verify_time_properties_and_ties():
    """Test time span, monotonicity, and tie reporting."""
    # Data with timestamp tie at 200 (2 rows tied)
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [100, 200, 200, 350],
    })
    stats = verify_time_properties(df)

    assert stats["min_transaction_dt"] == 100
    assert stats["max_transaction_dt"] == 350
    assert stats["span_seconds"] == 250
    assert stats["is_strictly_monotonic"] is False  # tie at 200
    assert stats["is_weakly_monotonic"] is True
    assert stats["unique_timestamps"] == 3
    assert stats["distinct_timestamps_gt1"] == 1
    assert stats["tied_rows"] == 2
    assert stats["surplus_timestamp_rows"] == 1
    assert stats["max_ties_in_single_second"] == 2
    assert "Strict-inequality filtering" in stats["tie_handling_rule"]


def test_strictly_monotonic_series():
    """Test strictly monotonic time sequence without ties."""
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [10, 20, 30],
    })
    stats = verify_time_properties(df)
    assert stats["is_strictly_monotonic"] is True
    assert stats["is_weakly_monotonic"] is True
    assert stats["distinct_timestamps_gt1"] == 0
    assert stats["tied_rows"] == 0
    assert stats["surplus_timestamp_rows"] == 0
    assert stats["max_ties_in_single_second"] == 1


def test_downcast_dtypes():
    """Test downcasting reduces memory without altering numerical values."""
    df = pd.DataFrame({
        "f_val": np.array([1.5, 2.5, 3.5], dtype=np.float64),
        "i_val": np.array([10, 20, 30], dtype=np.int64),
    })
    df_down = downcast_dtypes(df)
    assert df_down["f_val"].dtype == np.float32
    assert df_down["i_val"].dtype in (np.int16, np.int32)
    assert np.allclose(df_down["f_val"], [1.5, 2.5, 3.5])
    assert list(df_down["i_val"]) == [10, 20, 30]
