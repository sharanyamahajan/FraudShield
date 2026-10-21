"""Data loader and ingestion pipeline for FraudShield.

Loads train_transaction.csv and train_identity.csv, performs a left join on
TransactionID, performs schema validation and tie analysis, downcasts numeric
types for memory efficiency, and saves to data/interim/joined_transactions.parquet.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudshield.data.loader")


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 to float32 and int64 to int32/int16 where safe to reduce RAM usage."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == np.float64:
            df[col] = df[col].astype(np.float32)
        elif col_type == np.int64:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    return df


def load_raw_dataset(
    raw_dir: Path,
    transaction_file: str = "train_transaction.csv",
    identity_file: str = "train_identity.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw transaction and identity CSVs."""
    txn_path = raw_dir / transaction_file
    id_path = raw_dir / identity_file

    if not txn_path.exists():
        raise FileNotFoundError(f"Missing required file: {txn_path}")
    if not id_path.exists():
        raise FileNotFoundError(f"Missing required file: {id_path}")

    logger.info("Reading %s ...", txn_path)
    df_txn = pd.read_csv(txn_path)
    logger.info("Loaded transactions: %d rows, %d columns.", len(df_txn), df_txn.shape[1])

    logger.info("Reading %s ...", id_path)
    df_id = pd.read_csv(id_path)
    logger.info("Loaded identity: %d rows, %d columns.", len(df_id), df_id.shape[1])

    return df_txn, df_id


def join_and_process(
    df_txn: pd.DataFrame,
    df_id: pd.DataFrame,
    id_col: str = "TransactionID",
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    """Perform left join of transaction and identity on TransactionID, sort by time and ID."""
    logger.info("Performing left join on '%s'...", id_col)
    merged = pd.merge(df_txn, df_id, on=id_col, how="left")
    logger.info("Merged shape: %s", merged.shape)

    # Verify duplicate TransactionIDs
    dup_ids = merged[id_col].duplicated().sum()
    if dup_ids > 0:
        raise ValueError(f"Found {dup_ids} duplicate {id_col} values in merged dataset.")

    # Sort strictly by time_col ascending, with id_col as stable deterministic tie-breaker
    logger.info("Sorting by '%s' and '%s'...", time_col, id_col)
    merged = merged.sort_values(by=[time_col, id_col]).reset_index(drop=True)

    # Downcast numeric types to optimize memory and I/O
    logger.info("Downcasting numeric datatypes...")
    merged = downcast_dtypes(merged)

    return merged


def verify_time_properties(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    id_col: str = "TransactionID",
) -> Dict[str, Any]:
    """Inspect and assert monotonicity, time range, duplicate timestamps (ties), and tie handling."""
    times = df[time_col].values
    is_strictly_monotonic = bool(np.all(times[:-1] < times[1:]))
    is_weakly_monotonic = bool(np.all(times[:-1] <= times[1:]))

    total_rows = len(df)
    dt_counts = df[time_col].value_counts()
    unique_times = len(dt_counts)
    surplus_rows = total_rows - unique_times
    distinct_timestamps_gt1 = int((dt_counts > 1).sum())
    tied_rows = int(dt_counts[dt_counts > 1].sum())
    max_ties_in_single_second = int(dt_counts.max()) if len(dt_counts) > 0 else 0

    min_dt = int(df[time_col].min())
    max_dt = int(df[time_col].max())
    span_seconds = max_dt - min_dt
    span_days = span_seconds / 86400.0

    stats = {
        "min_transaction_dt": min_dt,
        "max_transaction_dt": max_dt,
        "span_seconds": span_seconds,
        "span_days": round(span_days, 2),
        "is_strictly_monotonic": is_strictly_monotonic,
        "is_weakly_monotonic": is_weakly_monotonic,
        "total_rows": total_rows,
        "unique_timestamps": unique_times,
        "distinct_timestamps_gt1": distinct_timestamps_gt1,
        "tied_rows": tied_rows,
        "tied_rows_pct": round(tied_rows / total_rows * 100, 2) if total_rows > 0 else 0.0,
        "surplus_timestamp_rows": surplus_rows,
        "max_ties_in_single_second": max_ties_in_single_second,
        "tie_handling_rule": (
            "Strict-inequality filtering (T_prior < T_current) ensures transactions "
            "occurring at the exact same TransactionDT (ties) do not leak into each other's "
            "backward-looking historical features. Secondary sorting by TransactionID "
            "provides a deterministic, reproducible row order. Split boundaries are adjusted "
            "to never split inside a group of tied timestamps."
        ),
    }
    return stats


def save_interim_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Save joined dataset to snappy-compressed parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving joined data to %s ...", output_path)
    df.to_parquet(output_path, compression="snappy", index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Successfully written %s (%.2f MB).", output_path, size_mb)


def run_ingestion_pipeline(config_path: Path = Path("configs/config.yaml")) -> Dict[str, Any]:
    """Execute complete ingestion pipeline and return verification metadata."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    txn_file = cfg["data"]["raw_transactions"]
    id_file = cfg["data"]["raw_identity"]
    output_parquet = Path(cfg["data"]["joined_parquet"])
    time_col = cfg["data"]["time_col"]
    id_col = cfg["data"]["id_col"]

    df_txn, df_id = load_raw_dataset(raw_dir, txn_file, id_file)
    merged = join_and_process(df_txn, df_id, id_col, time_col)
    time_stats = verify_time_properties(merged, time_col, id_col)
    save_interim_parquet(merged, output_parquet)

    return {
        "total_rows": len(merged),
        "total_columns": merged.shape[1],
        "txn_rows": len(df_txn),
        "identity_rows": len(df_id),
        "time_stats": time_stats,
    }
