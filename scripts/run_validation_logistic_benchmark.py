"""Validation-Only Experiment: XGBoost with objective='binary:logistic'.

Evaluates Set B and Set C1 on Validation (N=88,581), computes 500-resample paired
bootstrap CI for C1 - B, and recomputes the RQ2 random vs temporal split gap.
DOES NOT ACCESS OR EVALUATE THE TEST SPLIT.
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

from fraudshield.features.pipeline import TreePreprocessor, get_model_feature_columns
from fraudshield.evaluation.metrics import (
    evaluate_model_performance,
    compute_pr_auc,
    compute_roc_auc,
    compute_budget_metrics,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("val_logistic")


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_c1: np.ndarray,
    y_b: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = 42,
    budgets: List[float] = [0.005, 0.01, 0.02, 0.05],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    diff_pr = compute_pr_auc(y_true, y_c1) - compute_pr_auc(y_true, y_b)
    diff_roc = compute_roc_auc(y_true, y_c1) - compute_roc_auc(y_true, y_b)
    bm_c1 = compute_budget_metrics(y_true, y_c1, budgets=budgets)
    bm_b = compute_budget_metrics(y_true, y_b, budgets=budgets)
    diff_budgets = {k: bm_c1[k]["recall"] - bm_b[k]["recall"] for k in bm_c1}

    boot_pr = []
    boot_roc = []
    boot_budgets = {k: [] for k in bm_c1}

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        yc = y_c1[idx]
        yb = y_b[idx]

        boot_pr.append(compute_pr_auc(yt, yc) - compute_pr_auc(yt, yb))
        boot_roc.append(compute_roc_auc(yt, yc) - compute_roc_auc(yt, yb))
        bc = compute_budget_metrics(yt, yc, budgets=budgets)
        bb = compute_budget_metrics(yt, yb, budgets=budgets)
        for k in bc:
            boot_budgets[k].append(bc[k]["recall"] - bb[k]["recall"])

    def summarize(point: float, boots: List[float]):
        arr = np.array(boots)
        ci_l = float(np.percentile(arr, 2.5))
        ci_u = float(np.percentile(arr, 97.5))
        return {
            "point_estimate": round(float(point), 6),
            "mean_diff": round(float(np.mean(arr)), 6),
            "std_error": round(float(np.std(arr, ddof=1)), 6),
            "ci_95_lower": round(ci_l, 6),
            "ci_95_upper": round(ci_u, 6),
            "strictly_positive": bool(ci_l > 0),
        }

    res = {
        "pr_auc": summarize(diff_pr, boot_pr),
        "roc_auc": summarize(diff_roc, boot_roc),
    }
    for k in bm_c1:
        res[f"recall_{k}"] = summarize(diff_budgets[k], boot_budgets[k])
    return res


def run_logistic_experiment():
    t0 = time.time()
    proc_dir = Path("data/processed")
    reports_dir = Path("reports")

    # Load targets
    y_tr = pd.read_parquet(proc_dir / "train_set_b.parquet", columns=["isFraud"])["isFraud"].values.astype(np.int32)
    y_vl = pd.read_parquet(proc_dir / "val_set_b.parquet", columns=["isFraud"])["isFraud"].values.astype(np.int32)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 6,
    }

    # =========================================================================
    # 1. SET B (TEMPORAL VALIDATION)
    # =========================================================================
    logger.info("Training Set B with objective='binary:logistic'...")
    df_tr_b = pd.read_parquet(proc_dir / "train_set_b.parquet")
    df_vl_b = pd.read_parquet(proc_dir / "val_set_b.parquet")
    feats_b = get_model_feature_columns(df_tr_b)

    prep_b = TreePreprocessor().fit(df_tr_b[feats_b])
    X_tr_b = np.ascontiguousarray(prep_b.transform(df_tr_b[feats_b]).values, dtype=np.float32)
    X_vl_b = np.ascontiguousarray(prep_b.transform(df_vl_b[feats_b]).values, dtype=np.float32)
    del df_tr_b, df_vl_b
    gc.collect()

    dtr_b = xgb.DMatrix(X_tr_b, label=y_tr)
    dvl_b = xgb.DMatrix(X_vl_b, label=y_vl)
    del X_tr_b, X_vl_b
    gc.collect()

    bst_b = xgb.train(params, dtr_b, num_boost_round=300, evals=[(dvl_b, "val")], early_stopping_rounds=30, verbose_eval=False)
    p_vl_b = bst_b.predict(dvl_b)
    m_b = evaluate_model_performance(y_vl, p_vl_b)
    m_b["best_iteration"] = int(bst_b.best_iteration)
    del dtr_b, dvl_b, bst_b
    gc.collect()
    logger.info("Set B (binary:logistic) Validation: PR-AUC = %.6f, ROC-AUC = %.6f, Best Iter = %d",
                m_b["pr_auc"], m_b["roc_auc"], m_b["best_iteration"])

    # =========================================================================
    # 2. SET C1 (TEMPORAL VALIDATION)
    # =========================================================================
    logger.info("Training Set C1 with objective='binary:logistic'...")
    df_tr_c1 = pd.read_parquet(proc_dir / "train_set_c1.parquet")
    df_vl_c1 = pd.read_parquet(proc_dir / "val_set_c1.parquet")
    feats_c1 = get_model_feature_columns(df_tr_c1)

    prep_c1 = TreePreprocessor().fit(df_tr_c1[feats_c1])
    X_tr_c1 = np.ascontiguousarray(prep_c1.transform(df_tr_c1[feats_c1]).values, dtype=np.float32)
    X_vl_c1 = np.ascontiguousarray(prep_c1.transform(df_vl_c1[feats_c1]).values, dtype=np.float32)
    del df_tr_c1, df_vl_c1
    gc.collect()

    dtr_c1 = xgb.DMatrix(X_tr_c1, label=y_tr)
    dvl_c1 = xgb.DMatrix(X_vl_c1, label=y_vl)
    del X_tr_c1, X_vl_c1
    gc.collect()

    bst_c1 = xgb.train(params, dtr_c1, num_boost_round=300, evals=[(dvl_c1, "val")], early_stopping_rounds=30, verbose_eval=False)
    p_vl_c1 = bst_c1.predict(dvl_c1)
    m_c1 = evaluate_model_performance(y_vl, p_vl_c1)
    m_c1["best_iteration"] = int(bst_c1.best_iteration)
    del dtr_c1, dvl_c1, bst_c1
    gc.collect()
    logger.info("Set C1 (binary:logistic) Validation: PR-AUC = %.6f, ROC-AUC = %.6f, Best Iter = %d",
                m_c1["pr_auc"], m_c1["roc_auc"], m_c1["best_iteration"])

    # Paired Bootstrap on Validation
    logger.info("Computing 500-sample paired bootstrap on validation (C1 - B)...")
    boot_res = compute_bootstrap_ci(y_vl, p_vl_c1, p_vl_b, n_bootstraps=500, seed=42)

    # =========================================================================
    # 3. RQ2 RANDOM SPLIT ON SET B (WITH binary:logistic)
    # =========================================================================
    logger.info("Training Set B on 70/15 random split with objective='binary:logistic'...")
    full_85_b = pd.concat([
        pd.read_parquet(proc_dir / "train_set_b.parquet"),
        pd.read_parquet(proc_dir / "val_set_b.parquet"),
    ], ignore_index=True)

    idx_all = np.arange(len(full_85_b))
    idx_tr, idx_vl = train_test_split(idx_all, test_size=len(y_vl), random_state=42, shuffle=True)
    df_rnd_tr = full_85_b.iloc[idx_tr]
    df_rnd_vl = full_85_b.iloc[idx_vl]
    del full_85_b
    gc.collect()

    y_rnd_tr = df_rnd_tr["isFraud"].values.astype(np.int32)
    y_rnd_vl = df_rnd_vl["isFraud"].values.astype(np.int32)

    prep_rnd = TreePreprocessor().fit(df_rnd_tr[feats_b])
    X_tr_rnd = np.ascontiguousarray(prep_rnd.transform(df_rnd_tr[feats_b]).values, dtype=np.float32)
    X_vl_rnd = np.ascontiguousarray(prep_rnd.transform(df_rnd_vl[feats_b]).values, dtype=np.float32)
    del df_rnd_tr, df_rnd_vl
    gc.collect()

    dtr_rnd = xgb.DMatrix(X_tr_rnd, label=y_rnd_tr)
    dvl_rnd = xgb.DMatrix(X_vl_rnd, label=y_rnd_vl)
    del X_tr_rnd, X_vl_rnd
    gc.collect()

    bst_rnd = xgb.train(params, dtr_rnd, num_boost_round=300, evals=[(dvl_rnd, "val")], early_stopping_rounds=30, verbose_eval=False)
    p_vl_rnd = bst_rnd.predict(dvl_rnd)
    m_rnd = evaluate_model_performance(y_rnd_vl, p_vl_rnd)
    m_rnd["best_iteration"] = int(bst_rnd.best_iteration)
    del dtr_rnd, dvl_rnd, bst_rnd
    gc.collect()

    logger.info("Random Split Set B (binary:logistic): PR-AUC = %.6f, ROC-AUC = %.6f, Best Iter = %d",
                m_rnd["pr_auc"], m_rnd["roc_auc"], m_rnd["best_iteration"])

    rq2_gap = {
        "temporal_pr_auc": m_b["pr_auc"],
        "random_pr_auc": m_rnd["pr_auc"],
        "pr_auc_gap": round(m_rnd["pr_auc"] - m_b["pr_auc"], 6),
        "temporal_roc_auc": m_b["roc_auc"],
        "random_roc_auc": m_rnd["roc_auc"],
        "roc_auc_gap": round(m_rnd["roc_auc"] - m_b["roc_auc"], 6),
        "temporal_recall_top_1.0pct": m_b["budgets"]["top_1.0pct"]["recall"],
        "random_recall_top_1.0pct": m_rnd["budgets"]["top_1.0pct"]["recall"],
        "recall_1.0pct_gap": round(m_rnd["budgets"]["top_1.0pct"]["recall"] - m_b["budgets"]["top_1.0pct"]["recall"], 6),
        "temporal_recall_top_5.0pct": m_b["budgets"]["top_5.0pct"]["recall"],
        "random_recall_top_5.0pct": m_rnd["budgets"]["top_5.0pct"]["recall"],
        "recall_5.0pct_gap": round(m_rnd["budgets"]["top_5.0pct"]["recall"] - m_b["budgets"]["top_5.0pct"]["recall"], 6),
    }

    # Save artifact
    output_artifact = {
        "objective": "binary:logistic",
        "eval_split": "validation_strictly_only",
        "validation_samples": len(y_vl),
        "validation_frauds": int(np.sum(y_vl)),
        "model_set_b": m_b,
        "model_set_c1": m_c1,
        "paired_bootstrap_c1_minus_b": boot_res,
        "rq2_gap": rq2_gap,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    with open(reports_dir / "validation_logistic_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(output_artifact, f, indent=2)

    logger.info("==================================================================")
    logger.info("COMPLETED VALIDATION LOGISTIC BENCHMARK IN %.2f SECONDS!", time.time() - t0)
    logger.info("Set B PR-AUC:               %.6f", m_b["pr_auc"])
    logger.info("Set C1 PR-AUC:              %.6f", m_c1["pr_auc"])
    logger.info("Delta PR-AUC (C1 - B):      %.6f (95%% CI: [%.6f, %.6f])",
                boot_res["pr_auc"]["point_estimate"], boot_res["pr_auc"]["ci_95_lower"], boot_res["pr_auc"]["ci_95_upper"])
    logger.info("RQ2 PR-AUC Gap:             +%.6f (Temporal: %.6f, Random: %.6f)",
                rq2_gap["pr_auc_gap"], rq2_gap["temporal_pr_auc"], rq2_gap["random_pr_auc"])
    logger.info("==================================================================")


if __name__ == "__main__":
    run_logistic_experiment()
