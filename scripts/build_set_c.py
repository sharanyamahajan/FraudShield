"""Script to generate Set C1 and Set C2 parquets and compute Milestone 4 reporting statistics.

Full timeline sorted by (TransactionDT, TransactionID).
Strictly backward-looking, label-free.
"""

import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd

from fraudshield.features.history import (
    compute_entity_history_features,
    compute_reporting_statistics,
    C1_HISTORY_COLS,
    C2_HISTORY_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudshield.build_set_c")


def main():
    interim_path = Path("data/interim/joined_transactions.parquet")
    proc_dir = Path("data/processed")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading full joined timeline from %s ...", interim_path)
    full_df = pd.read_parquet(
        interim_path,
        columns=["TransactionID", "TransactionDT", "card1", "addr1", "TransactionAmt", "P_emaildomain"],
    )
    n_total = len(full_df)
    logger.info("Loaded %d transactions across full timeline.", n_total)

    # 1. Compute Primary Key History Features (card1) -> Set C1
    logger.info("Computing primary key (card1) backward-looking history features...")
    t0 = time.time()
    c1_feats = compute_entity_history_features(full_df, entity_key="card1", include_addr1=True)
    t1 = time.time()
    logger.info("Computed C1 history features (%d cols) in %.2f seconds.", len(c1_feats.columns), t1 - t0)
    c1_feats["TransactionID"] = full_df["TransactionID"].values

    # 2. Compute Secondary Composite Key History Features (card1, addr1) -> Set C2
    logger.info("Computing secondary composite key (card1, addr1) backward-looking history features...")
    t2 = time.time()
    c2_feats = compute_entity_history_features(
        full_df, entity_key=["card1", "addr1"], include_addr1=False, prefix="card1_addr1"
    )
    t3 = time.time()
    logger.info("Computed C2 history features (%d cols) in %.2f seconds.", len(c2_feats.columns), t3 - t2)
    c2_feats["TransactionID"] = full_df["TransactionID"].values

    # 3. Load splits to identify IDs and attach to Set B
    logger.info("Loading existing Set B splits to assemble C1 and C2...")
    df_train_b = pd.read_parquet(proc_dir / "train_set_b.parquet")
    df_val_b = pd.read_parquet(proc_dir / "val_set_b.parquet")
    df_test_b = pd.read_parquet(proc_dir / "test_set_b.parquet")

    train_ids = set(df_train_b["TransactionID"])
    val_ids = set(df_val_b["TransactionID"])
    test_ids = set(df_test_b["TransactionID"])

    # 4. Compute Milestone 4 Reporting Statistics
    logger.info("Computing Milestone 4 coverage and percentile statistics...")
    reporting_stats = compute_reporting_statistics(
        full_df=full_df,
        c1_feats=c1_feats,
        c2_feats=c2_feats,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )

    stats_path = reports_dir / "history_coverage.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(reporting_stats, f, indent=2)
    logger.info("Saved coverage and percentile statistics to %s", stats_path)

    # 5. Merge and export C1 splits
    logger.info("Merging and saving Set C1 splits...")
    train_c1 = df_train_b.merge(c1_feats, on="TransactionID", how="left")
    val_c1 = df_val_b.merge(c1_feats, on="TransactionID", how="left")
    test_c1 = df_test_b.merge(c1_feats, on="TransactionID", how="left")

    train_c1.to_parquet(proc_dir / "train_set_c1.parquet", compression="snappy", index=False)
    val_c1.to_parquet(proc_dir / "val_set_c1.parquet", compression="snappy", index=False)
    test_c1.to_parquet(proc_dir / "test_set_c1.parquet", compression="snappy", index=False)
    logger.info("Saved Set C1 (train=%s, val=%s, test=%s)", train_c1.shape, val_c1.shape, test_c1.shape)

    # 6. Merge and export C2 splits
    logger.info("Merging and saving Set C2 splits...")
    train_c2 = df_train_b.merge(c2_feats, on="TransactionID", how="left")
    val_c2 = df_val_b.merge(c2_feats, on="TransactionID", how="left")
    test_c2 = df_test_b.merge(c2_feats, on="TransactionID", how="left")

    train_c2.to_parquet(proc_dir / "train_set_c2.parquet", compression="snappy", index=False)
    val_c2.to_parquet(proc_dir / "val_set_c2.parquet", compression="snappy", index=False)
    test_c2.to_parquet(proc_dir / "test_set_c2.parquet", compression="snappy", index=False)
    logger.info("Saved Set C2 (train=%s, val=%s, test=%s)", train_c2.shape, val_c2.shape, test_c2.shape)

    print("\n" + "=" * 80)
    print("MILESTONE 4: ENTITY COVERAGE REPORTING")
    print("=" * 80)
    shares = reporting_stats["row_share_with_ge_10_prior_txns"]
    for split_name in ["train", "val", "test"]:
        s = shares[split_name]
        print(f"[{split_name.upper()} SPLIT - {s['total_rows']:,} rows]")
        print(f"  - card1 (Primary):           {s['card1_rows_ge_10']:,} rows ({s['card1_share_pct']}%) with >= 10 prior txns")
        print(f"  - (card1, addr1) (Secondary): {s['card1_addr1_rows_ge_10']:,} rows ({s['card1_addr1_share_pct']}%) with >= 10 prior txns")

    p = reporting_stats["card1_percentiles"]
    print("\n" + "=" * 80)
    print("CARD1 PERCENTILE TABLE (across unique entities)")
    print("=" * 80)
    print(f"Unique card1 Entities: {p['unique_entities']:,}")
    print(f"Median (50th):         {p['median']}")
    print(f"75th Percentile:       {p['p75']}")
    print(f"90th Percentile:       {p['p90']}")
    print(f"95th Percentile:       {p['p95']}")
    print(f"99th Percentile:       {p['p99']}")
    print(f"Max:                   {p['max']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
