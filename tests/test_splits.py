"""Unit tests for time-ordered temporal splits and tied-timestamp boundary rules."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
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


def test_every_row_assigned_to_exactly_one_split():
    """Assert every single input row is assigned to exactly one split with zero duplicates and zero omissions."""
    n = 200
    df = pd.DataFrame({
        "TransactionID": [f"TXN_{i:04d}" for i in range(n)],
        "TransactionDT": np.repeat(np.arange(100, 150), 4),  # multiple ties
        "isFraud": np.zeros(n, dtype=int),
    })

    train_df, val_df, test_df, _ = create_temporal_splits(df, 0.70, 0.15, 0.15)

    all_split_ids = list(train_df["TransactionID"]) + list(val_df["TransactionID"]) + list(test_df["TransactionID"])

    # 1. Total row count matches
    assert len(all_split_ids) == n
    # 2. No duplicates across any split
    assert len(set(all_split_ids)) == n
    # 3. Exact set equality with original input rows
    assert set(all_split_ids) == set(df["TransactionID"])


def test_premature_test_split_access_guard_raises_error():
    """Verify that calling load_split_parquet for 'test' from an unauthorized caller raises PrematureTestAccessError."""
    from fraudshield.splits.guard import (
        load_split_parquet,
        PrematureTestAccessError,
    )

    with pytest.raises(PrematureTestAccessError, match="Premature access to the held-out test split is strictly forbidden"):
        load_split_parquet("test", feature_set="a")


def test_authorized_call_to_test_split_logs_and_loads(tmp_path):
    """Verify that an authorized evaluator module in the call stack can load a test split and logs access
    to a temporary log path without touching reports/test_access_log.txt.
    """
    from pathlib import Path
    from fraudshield.splits.guard import load_split_parquet

    # Create dummy test parquet in tmp_path (NEVER use real test split)
    dummy_df = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0, 1]})
    test_parquet_path = tmp_path / "test_set_a.parquet"
    dummy_df.to_parquet(test_parquet_path, index=False)
    temp_log_path = tmp_path / "temp_test_access_log.txt"

    # Execute a function inside a frame with filename 'final_evaluator.py'
    code = compile(
        "df = load_split_parquet('test', feature_set='a', base_dir=tmp_path, log_file=temp_log_path)",
        "final_evaluator.py",
        "exec"
    )
    exec_scope = {
        "load_split_parquet": load_split_parquet,
        "tmp_path": tmp_path,
        "temp_log_path": temp_log_path,
    }
    exec(code, exec_scope)
    loaded = exec_scope["df"]
    assert len(loaded) == 2

    # Verify that the temporary log was written and tracked log was NOT written to
    assert temp_log_path.exists()
    content = temp_log_path.read_text(encoding="utf-8")
    assert "AUTHORIZED TEST ACCESS" in content
    assert "final_evaluator.py" in content


def test_pytest_leaves_tracked_test_access_log_unchanged(tmp_path):
    """Assert that running tests leaves the tracked reports/test_access_log.txt strictly unchanged."""
    log_path = Path("reports/test_access_log.txt")
    if not log_path.exists():
        return
    initial_bytes = log_path.read_bytes()

    # Execute authorized test call with temporary directory fixture
    test_authorized_call_to_test_split_logs_and_loads(tmp_path)

    # Assert tracked production log remains byte-for-byte identical
    after_bytes = log_path.read_bytes()
    assert initial_bytes == after_bytes, "reports/test_access_log.txt was modified during test execution!"


def test_static_analysis_no_unauthorized_module_references_test_split():
    """Static analysis test: assert that no module except final evaluators references test split or test parquets."""
    from pathlib import Path

    src_root = Path("src/fraudshield")
    # Allowed modules to reference test splits/parquets:
    # guard.py (loader), temporal.py (ingestion partitioner), cli.py (export/pytest CLI), final_evaluator/evaluate_test (M5)
    allowed_modules = {
        "guard.py",
        "temporal.py",
        "cli.py",
        "final_evaluator.py",
        "evaluate_test.py",
    }

    violations = []
    for py_file in src_root.rglob("*.py"):
        if py_file.name in allowed_modules:
            continue
        text = py_file.read_text(encoding="utf-8")

        # Check for forbidden patterns
        if "test_set_" in text:
            violations.append(f"{py_file.as_posix()}: references 'test_set_'")
        if 'load_split_parquet("test"' in text or "load_split_parquet('test'" in text:
            violations.append(f"{py_file.as_posix()}: calls load_split_parquet with 'test'")
        if 'split_name="test"' in text or "split_name='test'" in text:
            violations.append(f"{py_file.as_posix()}: specifies split_name='test'")

    assert not violations, f"Unauthorized test split references found: {violations}"



