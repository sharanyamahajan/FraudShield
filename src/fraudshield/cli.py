"""Command-line interface and cross-platform task runner for FraudShield."""

import argparse
import logging
import os
import sys
import subprocess

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
    """Build feature sets A, B, and C."""
    logger.info("Executing 'features' command (Milestones 2 & 4)...")
    logger.info("Feature engineering entry point ready.")
    return 0


def run_train(args: argparse.Namespace) -> int:
    """Train models across feature sets."""
    logger.info("Executing 'train' command (Milestones 3 & 5)...")
    logger.info("Training pipeline entry point ready.")
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
