"""Milestone 4 validation evaluation engine for Set B, Set C1, and Set C2.

Evaluates unweighted models on VALIDATION split only:
- Logistic Regression (class_weight=None)
- XGBoost (scale_pos_weight=None)

On:
- Set B: Baseline (434 features)
- Set C1: Set B + card1 history (443 features)
- Set C2: Set B + (card1, addr1) history (442 features)

Evaluates:
- PR-AUC, ROC-AUC, Brier score
- Review budgets: 0.5%, 1.0%, 2.0%, 5.0% (Recall, Precision, FPR, Max Achievable Recall)

Test split stays strictly sealed and is never loaded.
Results are saved to reports/history_val.csv and reports/history_val.json.
"""

import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import yaml

from fraudshield.splits.guard import load_split_parquet
from fraudshield.features.pipeline import (
    LinearPreprocessor,
    TreePreprocessor,
    get_model_feature_columns,
)
from fraudshield.models.train_baselines import (
    train_and_eval_logistic_regression,
    train_and_eval_xgboost,
    compute_file_sha256,
    get_git_commit_hash,
    is_git_dirty,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudshield.models.train_history")


def run_history_validation(config_path: Path = Path("configs/config.yaml")) -> pd.DataFrame:
    """Execute validation benchmark on Set B, Set C1, and Set C2 for unweighted LR and XGBoost."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    run_id = str(uuid.uuid4())[:8]
    git_hash = get_git_commit_hash()
    dirty_flag = is_git_dirty()

    logger.info("Starting Milestone 4 History Validation (Run ID: %s, Git: %s, Dirty: %s)...", run_id, git_hash, dirty_flag)

    proc_dir = Path(cfg["paths"]["processed_data_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Verify parquets exist before proceeding
    required_parquets = [
        proc_dir / "train_set_b.parquet",
        proc_dir / "val_set_b.parquet",
        proc_dir / "train_set_c1.parquet",
        proc_dir / "val_set_c1.parquet",
        proc_dir / "train_set_c2.parquet",
        proc_dir / "val_set_c2.parquet",
    ]
    for p in required_parquets:
        if not p.exists():
            raise FileNotFoundError(f"Required parquet not found: {p}. Run scripts/build_set_c.py first.")

    parquet_hashes = {p.stem: compute_file_sha256(p) for p in required_parquets}

    results = []

    def evaluate_feature_set(feature_set_name: str, split_key: str):
        logger.info("--- EVALUATING %s ---", feature_set_name.upper())
        df_train = load_split_parquet("train", feature_set=split_key)
        df_val = load_split_parquet("val", feature_set=split_key)

        feature_cols = get_model_feature_columns(df_train)
        y_train = df_train["isFraud"].values.astype(int)
        y_val = df_val["isFraud"].values.astype(int)

        n_val_total = len(y_val)
        n_val_pos = int(np.sum(y_val == 1))
        max_rec_05 = round(min(max(1, int(round(n_val_total * 0.005))), n_val_pos) / n_val_pos, 6)
        max_rec_10 = round(min(max(1, int(round(n_val_total * 0.010))), n_val_pos) / n_val_pos, 6)
        max_rec_20 = round(min(max(1, int(round(n_val_total * 0.020))), n_val_pos) / n_val_pos, 6)
        max_rec_50 = round(min(max(1, int(round(n_val_total * 0.050))), n_val_pos) / n_val_pos, 6)

        def build_row(model_name: str, variant: str, m: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "run_id": run_id,
                "git_hash": git_hash,
                "is_git_dirty": dirty_flag,
                "feature_set": feature_set_name,
                "model_name": model_name,
                "variant": variant,
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "brier_score": m["brier_score"],
                "n_iter": m.get("n_iter", None),
                "converged": m.get("converged", None),
                "best_iteration": m.get("best_iteration", None),
                "max_achievable_recall_top_0.5pct": max_rec_05,
                "max_achievable_recall_top_1.0pct": max_rec_10,
                "max_achievable_recall_top_2.0pct": max_rec_20,
                "max_achievable_recall_top_5.0pct": max_rec_50,
                "recall_top_0.5pct": m["budgets"]["top_0.5pct"]["recall"],
                "recall_top_1.0pct": m["budgets"]["top_1.0pct"]["recall"],
                "recall_top_2.0pct": m["budgets"]["top_2.0pct"]["recall"],
                "recall_top_5.0pct": m["budgets"]["top_5.0pct"]["recall"],
                "precision_top_0.5pct": m["budgets"]["top_0.5pct"]["precision"],
                "precision_top_1.0pct": m["budgets"]["top_1.0pct"]["precision"],
                "precision_top_2.0pct": m["budgets"]["top_2.0pct"]["precision"],
                "precision_top_5.0pct": m["budgets"]["top_5.0pct"]["precision"],
                "fpr_top_0.5pct": m["budgets"]["top_0.5pct"]["fpr"],
                "fpr_top_1.0pct": m["budgets"]["top_1.0pct"]["fpr"],
                "fpr_top_2.0pct": m["budgets"]["top_2.0pct"]["fpr"],
                "fpr_top_5.0pct": m["budgets"]["top_5.0pct"]["fpr"],
            }

        # 1. Logistic Regression (class_weight=None)
        logger.info("[%s] Training Logistic Regression (class_weight=None)...", feature_set_name)
        m_lr, _ = train_and_eval_logistic_regression(
            df_train[feature_cols], y_train, df_val[feature_cols], y_val, class_weight=None, random_state=seed
        )
        results.append(build_row("Logistic Regression", "class_weight=None", m_lr))

        # 2. XGBoost (scale_pos_weight=None)
        logger.info("[%s] Training XGBoost (scale_pos_weight=None)...", feature_set_name)
        m_xgb, _ = train_and_eval_xgboost(
            df_train[feature_cols], y_train, df_val[feature_cols], y_val, scale_pos_weight=None, random_state=seed
        )
        results.append(build_row("XGBoost", "scale_pos_weight=None", m_xgb))

    # Evaluate Set B, Set C1, and Set C2
    evaluate_feature_set("Set B", "b")
    evaluate_feature_set("Set C1", "c1")
    evaluate_feature_set("Set C2", "c2")

    df_results = pd.DataFrame(results)

    # Save results to reports
    csv_path = reports_dir / "history_val.csv"
    json_path = reports_dir / "history_val.json"

    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "git_commit_hash": git_hash,
            "is_git_dirty": dirty_flag,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parquet_sha256": parquet_hashes,
            "config_snapshot": cfg,
            "results": results,
        }, f, indent=2)

    logger.info("Saved Milestone 4 history validation results to %s and %s", csv_path, json_path)
    return df_results


if __name__ == "__main__":
    df_res = run_history_validation()
    print("\n" + "=" * 100)
    print("MILESTONE 4: VALIDATION PERFORMANCE (Set B vs. Set C1 vs. Set C2)")
    print("=" * 100)
    cols = [
        "feature_set", "model_name", "pr_auc", "roc_auc",
        "recall_top_0.5pct", "recall_top_1.0pct", "recall_top_2.0pct", "recall_top_5.0pct",
        "precision_top_1.0pct", "fpr_top_1.0pct"
    ]
    print(df_res[cols].to_string(index=False))
    print("=" * 100 + "\n")
