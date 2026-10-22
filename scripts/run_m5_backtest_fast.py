"""Run Fold 1 of the Rolling-Origin Backtest and assemble final m5 artifacts."""

import gc
import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from fraudshield.features.pipeline import TreePreprocessor, get_model_feature_columns
from fraudshield.evaluation.metrics import evaluate_model_performance
from fraudshield.splits.temporal import find_next_distinct_boundary
from fraudshield.models.train_baselines import compute_file_sha256, get_git_commit_hash, is_git_dirty

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("m5_backtest")

def run_backtest_and_finalize():
    t_start = time.time()
    reports_dir = Path("reports")
    ckpt_dir = reports_dir / "m5_checkpoints"
    proc_dir = Path("data/processed")

    # Load b_s42 and c1_s42 checkpoints
    with open(ckpt_dir / "b_s42_meta.json", "r") as f:
        m_b_42 = json.load(f)
    with open(ckpt_dir / "c1_s42_meta.json", "r") as f:
        m_c1_42 = json.load(f)
    with open(ckpt_dir / "c2_s42_meta.json", "r") as f:
        m_c2_42 = json.load(f)

    # 1. Load 85% timeline for backtest
    logger.info("Loading 85% timeline for backtest...")
    full_85_b = pd.concat([
        pd.read_parquet(proc_dir / "train_set_b.parquet"),
        pd.read_parquet(proc_dir / "val_set_b.parquet"),
    ], ignore_index=True).sort_values(by=["TransactionDT", "TransactionID"]).reset_index(drop=True)

    feats_b = get_model_feature_columns(full_85_b)
    n_85 = len(full_85_b)
    raw_tr_end = int(n_85 * 0.50)
    raw_vl_end = int(n_85 * 0.65)
    tr_end = find_next_distinct_boundary(full_85_b, raw_tr_end)
    vl_end = find_next_distinct_boundary(full_85_b, raw_vl_end)

    f1_tr_b = full_85_b.iloc[:tr_end]
    f1_vl_b = full_85_b.iloc[tr_end:vl_end]
    y_f1_tr = f1_tr_b["isFraud"].values.astype(np.int32)
    y_f1_vl = f1_vl_b["isFraud"].values.astype(np.int32)

    logger.info("Fold 1: Train rows = %d, Val rows = %d, Val frauds = %d", tr_end, vl_end - tr_end, int(np.sum(y_f1_vl)))

    # Fold 1 Set B
    logger.info("Fitting Fold 1 Set B...")
    p_f1_b = TreePreprocessor().fit(f1_tr_b[feats_b])
    X_tr = np.ascontiguousarray(p_f1_b.transform(f1_tr_b[feats_b]).values, dtype=np.float32)
    X_vl = np.ascontiguousarray(p_f1_b.transform(f1_vl_b[feats_b]).values, dtype=np.float32)
    del f1_tr_b, f1_vl_b, full_85_b
    gc.collect()

    dtr_b = xgb.DMatrix(X_tr, label=y_f1_tr)
    dvl_b = xgb.DMatrix(X_vl, label=y_f1_vl)
    del X_tr, X_vl
    gc.collect()

    params = {
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1.0,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": 6,
    }
    bst_b = xgb.train(params, dtr_b, num_boost_round=300, evals=[(dvl_b, "val")], early_stopping_rounds=30, verbose_eval=False)
    p_vl_b = bst_b.predict(dvl_b)
    m_bt_f1_b = evaluate_model_performance(y_f1_vl, p_vl_b)
    m_bt_f1_b["best_iteration"] = int(bst_b.best_iteration)
    del dtr_b, dvl_b, bst_b, p_vl_b
    gc.collect()

    # Fold 1 Set C1
    logger.info("Loading 85% timeline for Set C1...")
    full_85_c1 = pd.concat([
        pd.read_parquet(proc_dir / "train_set_c1.parquet"),
        pd.read_parquet(proc_dir / "val_set_c1.parquet"),
    ], ignore_index=True).sort_values(by=["TransactionDT", "TransactionID"]).reset_index(drop=True)
    feats_c1 = get_model_feature_columns(full_85_c1)

    f1_tr_c1 = full_85_c1.iloc[:tr_end]
    f1_vl_c1 = full_85_c1.iloc[tr_end:vl_end]
    p_f1_c1 = TreePreprocessor().fit(f1_tr_c1[feats_c1])
    X_tr = np.ascontiguousarray(p_f1_c1.transform(f1_tr_c1[feats_c1]).values, dtype=np.float32)
    X_vl = np.ascontiguousarray(p_f1_c1.transform(f1_vl_c1[feats_c1]).values, dtype=np.float32)
    del f1_tr_c1, f1_vl_c1, full_85_c1
    gc.collect()

    dtr_c1 = xgb.DMatrix(X_tr, label=y_f1_tr)
    dvl_c1 = xgb.DMatrix(X_vl, label=y_f1_vl)
    del X_tr, X_vl
    gc.collect()

    logger.info("Fitting Fold 1 Set C1...")
    bst_c1 = xgb.train(params, dtr_c1, num_boost_round=300, evals=[(dvl_c1, "val")], early_stopping_rounds=30, verbose_eval=False)
    p_vl_c1 = bst_c1.predict(dvl_c1)
    m_bt_f1_c1 = evaluate_model_performance(y_f1_vl, p_vl_c1)
    m_bt_f1_c1["best_iteration"] = int(bst_c1.best_iteration)
    del dtr_c1, dvl_c1, bst_c1, p_vl_c1
    gc.collect()

    # Save checkpoints for fold 1
    with open(ckpt_dir / "bt_f1_b_meta.json", "w") as f:
        json.dump(m_bt_f1_b, f, indent=2)
    with open(ckpt_dir / "bt_f1_c1_meta.json", "w") as f:
        json.dump(m_bt_f1_c1, f, indent=2)

    # 2. Assemble Backtest Records
    y_val = pd.read_parquet(proc_dir / "val_set_b.parquet", columns=["isFraud"])["isFraud"].values.astype(np.int32)
    y_tr = pd.read_parquet(proc_dir / "train_set_b.parquet", columns=["isFraud"])["isFraud"].values.astype(np.int32)

    backtest_records = [
        {
            "fold": "Fold 1 (50% Train / 15% Val)",
            "train_rows": tr_end,
            "val_rows": vl_end - tr_end,
            "val_frauds": int(np.sum(y_f1_vl)),
            "b_pr_auc": m_bt_f1_b["pr_auc"],
            "c1_pr_auc": m_bt_f1_c1["pr_auc"],
            "c1_minus_b_pr_auc": round(m_bt_f1_c1["pr_auc"] - m_bt_f1_b["pr_auc"], 6),
            "b_roc_auc": m_bt_f1_b["roc_auc"],
            "c1_roc_auc": m_bt_f1_c1["roc_auc"],
            "c1_minus_b_roc_auc": round(m_bt_f1_c1["roc_auc"] - m_bt_f1_b["roc_auc"], 6),
            "b_recall_1.0pct": m_bt_f1_b["budgets"]["top_1.0pct"]["recall"],
            "c1_recall_1.0pct": m_bt_f1_c1["budgets"]["top_1.0pct"]["recall"],
            "c1_minus_b_recall_1.0pct": round(m_bt_f1_c1["budgets"]["top_1.0pct"]["recall"] - m_bt_f1_b["budgets"]["top_1.0pct"]["recall"], 6),
            "b_recall_2.0pct": m_bt_f1_b["budgets"]["top_2.0pct"]["recall"],
            "c1_recall_2.0pct": m_bt_f1_c1["budgets"]["top_2.0pct"]["recall"],
            "c1_minus_b_recall_2.0pct": round(m_bt_f1_c1["budgets"]["top_2.0pct"]["recall"] - m_bt_f1_b["budgets"]["top_2.0pct"]["recall"], 6),
            "b_recall_5.0pct": m_bt_f1_b["budgets"]["top_5.0pct"]["recall"],
            "c1_recall_5.0pct": m_bt_f1_c1["budgets"]["top_5.0pct"]["recall"],
            "c1_minus_b_recall_5.0pct": round(m_bt_f1_c1["budgets"]["top_5.0pct"]["recall"] - m_bt_f1_b["budgets"]["top_5.0pct"]["recall"], 6),
        },
        {
            "fold": "Fold 2 (70% Train / 15% Val)",
            "train_rows": len(y_tr),
            "val_rows": len(y_val),
            "val_frauds": int(np.sum(y_val)),
            "b_pr_auc": m_b_42["pr_auc"],
            "c1_pr_auc": m_c1_42["pr_auc"],
            "c1_minus_b_pr_auc": round(m_c1_42["pr_auc"] - m_b_42["pr_auc"], 6),
            "b_roc_auc": m_b_42["roc_auc"],
            "c1_roc_auc": m_c1_42["roc_auc"],
            "c1_minus_b_roc_auc": round(m_c1_42["roc_auc"] - m_b_42["roc_auc"], 6),
            "b_recall_1.0pct": m_b_42["budgets"]["top_1.0pct"]["recall"],
            "c1_recall_1.0pct": m_c1_42["budgets"]["top_1.0pct"]["recall"],
            "c1_minus_b_recall_1.0pct": round(m_c1_42["budgets"]["top_1.0pct"]["recall"] - m_b_42["budgets"]["top_1.0pct"]["recall"], 6),
            "b_recall_2.0pct": m_b_42["budgets"]["top_2.0pct"]["recall"],
            "c1_recall_2.0pct": m_c1_42["budgets"]["top_2.0pct"]["recall"],
            "c1_minus_b_recall_2.0pct": round(m_c1_42["budgets"]["top_2.0pct"]["recall"] - m_b_42["budgets"]["top_2.0pct"]["recall"], 6),
            "b_recall_5.0pct": m_b_42["budgets"]["top_5.0pct"]["recall"],
            "c1_recall_5.0pct": m_c1_42["budgets"]["top_5.0pct"]["recall"],
            "c1_minus_b_recall_5.0pct": round(m_c1_42["budgets"]["top_5.0pct"]["recall"] - m_b_42["budgets"]["top_5.0pct"]["recall"], 6),
        },
    ]

    parquet_hashes = {
        "train_set_b": compute_file_sha256(proc_dir / "train_set_b.parquet"),
        "val_set_b": compute_file_sha256(proc_dir / "val_set_b.parquet"),
        "train_set_c1": compute_file_sha256(proc_dir / "train_set_c1.parquet"),
        "val_set_c1": compute_file_sha256(proc_dir / "val_set_c1.parquet"),
        "train_set_c2": compute_file_sha256(proc_dir / "train_set_c2.parquet"),
        "val_set_c2": compute_file_sha256(proc_dir / "val_set_c2.parquet"),
    }

    reduced_scope_metadata = {
        "seeds_evaluated": [42],
        "seeds_count": 1,
        "bootstrap_resamples": 500,
        "rolling_backtest_folds": 2,
        "dropped_items": ["(d) seeds 123 and 7", "(f) C1 sensitivity run"],
        "dropped_rationale": "Dropped per priority order in rule 5 and user explicit instruction to finish within 10 minutes for presentation.",
        "hyperparameters": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": 6,
            "early_stopping_rounds": 30,
        },
    }

    df_bt = pd.DataFrame(backtest_records)
    df_bt.to_csv(reports_dir / "m5_rolling_backtest.csv", index=False)
    with open(reports_dir / "m5_rolling_backtest.json", "w") as f:
        json.dump({
            "run_id": "0f15c8f9",
            "git_commit_hash": get_git_commit_hash(),
            "is_git_dirty": is_git_dirty(),
            "parquet_sha256": parquet_hashes,
            "folds": backtest_records,
            "reduced_scope_metadata": reduced_scope_metadata,
        }, f, indent=2)

    # 3. Assemble Seed Stability (Seed 42)
    def flatten_metrics(m, fs, s):
        return {
            "feature_set": fs,
            "seed": s,
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "brier_score": m["brier_score"],
            "best_iteration": m.get("best_iteration"),
            "fit_time_seconds": m.get("fit_time_seconds"),
            "recall_top_0.5pct": m["budgets"]["top_0.5pct"]["recall"],
            "precision_top_0.5pct": m["budgets"]["top_0.5pct"]["precision"],
            "fpr_top_0.5pct": m["budgets"]["top_0.5pct"]["fpr"],
            "recall_top_1.0pct": m["budgets"]["top_1.0pct"]["recall"],
            "precision_top_1.0pct": m["budgets"]["top_1.0pct"]["precision"],
            "fpr_top_1.0pct": m["budgets"]["top_1.0pct"]["fpr"],
            "recall_top_2.0pct": m["budgets"]["top_2.0pct"]["recall"],
            "precision_top_2.0pct": m["budgets"]["top_2.0pct"]["precision"],
            "fpr_top_2.0pct": m["budgets"]["top_2.0pct"]["fpr"],
            "recall_top_5.0pct": m["budgets"]["top_5.0pct"]["recall"],
            "precision_top_5.0pct": m["budgets"]["top_5.0pct"]["precision"],
            "fpr_top_5.0pct": m["budgets"]["top_5.0pct"]["fpr"],
        }

    seed_rows = [
        flatten_metrics(m_b_42, "Set B", 42),
        flatten_metrics(m_c1_42, "Set C1", 42),
        flatten_metrics(m_c2_42, "Set C2", 42),
    ]
    df_seeds = pd.DataFrame(seed_rows)
    df_seeds.to_csv(reports_dir / "m5_seed_stability.csv", index=False)

    summary_seeds = []
    for row in seed_rows:
        summary_seeds.append({
            "feature_set": row["feature_set"],
            "seeds_count": 1,
            "pr_auc_mean": row["pr_auc"],
            "pr_auc_std": 0.0,
            "roc_auc_mean": row["roc_auc"],
            "roc_auc_std": 0.0,
            "recall_top_1.0pct_mean": row["recall_top_1.0pct"],
            "recall_top_1.0pct_std": 0.0,
            "recall_top_2.0pct_mean": row["recall_top_2.0pct"],
            "recall_top_2.0pct_std": 0.0,
            "recall_top_5.0pct_mean": row["recall_top_5.0pct"],
            "recall_top_5.0pct_std": 0.0,
        })

    with open(reports_dir / "m5_seed_stability.json", "w") as f:
        json.dump({
            "run_id": "0f15c8f9",
            "git_commit_hash": get_git_commit_hash(),
            "is_git_dirty": is_git_dirty(),
            "parquet_sha256": parquet_hashes,
            "summary": summary_seeds,
            "all_runs": seed_rows,
            "reduced_scope_metadata": reduced_scope_metadata,
        }, f, indent=2)

    logger.info("================================================================")
    logger.info("FINAL BACKTEST & SEED ARTIFACTS WRITTEN SUCCESSFULLY (%.2f s)!", time.time() - t_start)
    logger.info("================================================================")

if __name__ == "__main__":
    run_backtest_and_finalize()
