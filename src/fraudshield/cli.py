"""Command-line interface and cross-platform task runner for FraudShield."""

import argparse
import logging
import os
import sys
import subprocess
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudshield")


def run_data(args: argparse.Namespace) -> int:
    """Acquire, verify, and join IEEE-CIS dataset."""
    import json
    from fraudshield.data.loader import load_raw_dataset, join_and_process, verify_time_properties, save_interim_parquet
    from fraudshield.data.verify import analyze_dataset, generate_markdown_report
    import yaml
    from pathlib import Path

    logger.info("Executing 'data' command (Milestone 1)...")
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    txn_file = cfg["data"]["raw_transactions"]
    id_file = cfg["data"]["raw_identity"]
    output_parquet = Path(cfg["data"]["joined_parquet"])
    time_col = cfg["data"]["time_col"]
    id_col = cfg["data"]["id_col"]

    df_txn, df_id = load_raw_dataset(raw_dir, txn_file, id_file)
    id_len = len(df_id)
    merged = join_and_process(df_txn, df_id, id_col, time_col)
    save_interim_parquet(merged, output_parquet)

    logger.info("Computing full data verification statistics on actual dataset...")
    data_report = analyze_dataset(merged, identity_df_len=id_len, id_col=id_col, time_col=time_col)

    # Save docs/DATA_REPORT.md
    docs_report_path = Path("docs/DATA_REPORT.md")
    generate_markdown_report(data_report, docs_report_path)

    # Save reports/data_summary.json
    summary_path = Path("reports/data_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(data_report, f, indent=2)
    logger.info("Saved data summary JSON to %s", summary_path)

    logger.info("Milestone 1 data ingestion and verification complete.")
    return 0


def run_features(args: argparse.Namespace) -> int:
    """Build temporal splits and feature sets A and B (Milestone 2)."""
    import json
    from fraudshield.splits.temporal import create_temporal_splits
    from fraudshield.features.pipeline import extract_set_a_raw, extract_set_b_raw, extract_set_b, TreePreprocessor
    import yaml
    from pathlib import Path

    logger.info("Executing 'features' command (Milestone 2)...")
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    interim_path = Path(cfg["data"]["joined_parquet"])
    if not interim_path.exists():
        logger.error("Interim parquet not found at %s. Please run 'python -m fraudshield data' first.", interim_path)
        return 1

    logger.info("Loading joined dataset from %s ...", interim_path)
    df = pd.read_parquet(interim_path)

    # 1. Temporal Splits
    logger.info("Generating time-ordered 70/15/15 splits with tied-timestamp protection...")
    train_df, val_df, test_df, split_summary = create_temporal_splits(
        df,
        train_ratio=cfg["splits"]["train_ratio"],
        val_ratio=cfg["splits"]["val_ratio"],
        test_ratio=cfg["splits"]["test_ratio"],
        time_col=cfg["data"]["time_col"],
        id_col=cfg["data"]["id_col"],
        target_col=cfg["data"]["target_col"],
    )

    # Save split summary
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    logger.info("Saved split summary to %s", summary_path)

    # 2. Extract Set A
    logger.info("Extracting Feature Set A (Transaction-only)...")
    train_a = extract_set_a_raw(train_df, time_col=cfg["data"]["time_col"])
    val_a = extract_set_a_raw(val_df, time_col=cfg["data"]["time_col"])
    test_a = extract_set_a_raw(test_df, time_col=cfg["data"]["time_col"])

    for split_data, df_feat in [(train_df, train_a), (val_df, val_a), (test_df, test_a)]:
        df_feat["TransactionID"] = split_data[cfg["data"]["id_col"]].values
        df_feat["TransactionDT"] = split_data[cfg["data"]["time_col"]].values
        df_feat["isFraud"] = split_data[cfg["data"]["target_col"]].values.astype(np.int8)

    # 3. Extract Set B
    logger.info("Extracting Feature Set B (Set A + C/D/M/V + Identity)...")
    train_b = extract_set_b_raw(train_df, time_col=cfg["data"]["time_col"])
    val_b = extract_set_b_raw(val_df, time_col=cfg["data"]["time_col"])
    test_b = extract_set_b_raw(test_df, time_col=cfg["data"]["time_col"])

    for split_data, df_feat in [(train_df, train_b), (val_df, val_b), (test_df, test_b)]:
        df_feat["TransactionID"] = split_data[cfg["data"]["id_col"]].values
        df_feat["TransactionDT"] = split_data[cfg["data"]["time_col"]].values
        df_feat["isFraud"] = split_data[cfg["data"]["target_col"]].values.astype(np.int8)

    # 4. Save to processed directory
    proc_dir = Path(cfg["paths"]["processed_data_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving unencoded processed datasets to %s ...", proc_dir)
    train_a.to_parquet(proc_dir / "train_set_a.parquet", compression="snappy", index=False)
    val_a.to_parquet(proc_dir / "val_set_a.parquet", compression="snappy", index=False)
    test_a.to_parquet(proc_dir / "test_set_a.parquet", compression="snappy", index=False)

    train_b.to_parquet(proc_dir / "train_set_b.parquet", compression="snappy", index=False)
    val_b.to_parquet(proc_dir / "val_set_b.parquet", compression="snappy", index=False)
    test_b.to_parquet(proc_dir / "test_set_b.parquet", compression="snappy", index=False)

    logger.info("Set A total columns (with meta): %d | Set B total columns: %d", train_a.shape[1], train_b.shape[1])
    logger.info("Milestone 2 feature pipeline execution complete.")
    return 0


def run_train(args: argparse.Namespace) -> int:
    """Train baseline models on Set A and Set B, or evaluate Set C history features."""
    if getattr(args, "history", False):
        from fraudshield.models.train_history import run_history_validation

        logger.info("Executing 'train --history' command (Milestone 4)...")
        df_results = run_history_validation()
        print("\n" + "=" * 100)
        print("MILESTONE 4: VALIDATION PERFORMANCE (Set B vs. Set C1 vs. Set C2)")
        print("=" * 100)
        display_cols = [
            "feature_set", "model_name", "pr_auc", "roc_auc",
            "recall_top_0.5pct", "recall_top_1.0pct", "recall_top_2.0pct", "recall_top_5.0pct",
            "precision_top_1.0pct", "fpr_top_1.0pct"
        ]
        print(df_results[display_cols].to_string(index=False))
        print("=" * 100 + "\n")
        logger.info("Milestone 4 history validation evaluation complete.")
        return 0

    from fraudshield.models.train_baselines import run_validation_baselines

    logger.info("Executing 'train' command (Milestone 3)...")
    df_results = run_validation_baselines()
    print("\n" + "=" * 90)
    print("MILESTONE 3: VALIDATION BASELINE PERFORMANCE SUMMARY")
    print("=" * 90)
    display_cols = [
        "feature_set", "model_name", "variant",
        "pr_auc", "roc_auc", "recall_top_1.0pct", "precision_top_1.0pct", "fpr_top_1.0pct"
    ]
    print(df_results[display_cols].to_string(index=False))
    print("=" * 90 + "\n")
    logger.info("Milestone 3 training and validation evaluation complete.")
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    """Evaluate models, compute bootstrap CIs, and save experiment artifacts."""
    logger.info("Executing 'evaluate' command (Milestone 5)...")
    logger.info("Evaluation pipeline entry point ready.")
    return 0


def run_test(args: argparse.Namespace) -> int:
    """Run pytest test suite across platforms."""
    logger.info("Running pytest suite...")
    cmd = [sys.executable, "-m", "pytest", "-v", "tests"]
    result = subprocess.run(cmd)
    return result.returncode


def run_serve(args: argparse.Namespace) -> int:
    """Launch FastAPI dashboard and REST server."""
    logger.info("Executing 'serve' command (Milestones 7 & 8)...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "fraudshield.api.app:app",
        f"--host={args.host}",
        f"--port={args.port}",
    ]
    if args.reload:
        cmd.append("--reload")
    result = subprocess.run(cmd)
    return result.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fraudshield",
        description="FraudShield: Reproducible, Leakage-Safe Transaction Fraud Risk Scoring",
    )
    subparsers = parser.add_subparsers(dest="command", help="Target task to execute")

    # Command: data
    parser_data = subparsers.add_parser("data", help="Acquire, verify, and join raw data")
    parser_data.set_defaults(func=run_data)

    # Command: features
    parser_features = subparsers.add_parser("features", help="Compute feature sets A, B, and C")
    parser_features.set_defaults(func=run_features)

    # Command: train
    parser_train = subparsers.add_parser("train", help="Train baseline and XGBoost models")
    parser_train.add_argument("--history", action="store_true", help="Evaluate Milestone 4 history feature sets (Set B vs. C1 vs. C2)")
    parser_train.set_defaults(func=run_train)

    # Command: evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate models and generate reports")
    parser_eval.set_defaults(func=run_evaluate)

    # Command: test
    parser_test = subparsers.add_parser("test", help="Run the test suite")
    parser_test.set_defaults(func=run_test)

    # Command: serve
    parser_serve = subparsers.add_parser("serve", help="Launch FastAPI server & dashboard")
    parser_serve.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port")
    parser_serve.add_argument("--reload", action="store_true", help="Enable live reload")
    parser_serve.set_defaults(func=run_serve)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
