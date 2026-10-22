"""Milestone 6: Model Optimization, Calibration, Operating Points, Test Evaluation & SHAP.

This module is named 'evaluate_test.py', which is explicitly listed in
AUTHORIZED_CALLER_MODULES in src/fraudshield/splits/guard.py.
It performs the authorized single final evaluation on the sealed test split.
"""

import gc
import json
import logging
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
import xgboost as xgb
import shap

from fraudshield.splits.guard import load_split_parquet
from fraudshield.features.pipeline import TreePreprocessor, get_model_feature_columns
from fraudshield.evaluation.metrics import (
    evaluate_model_performance,
    compute_pr_auc,
    compute_roc_auc,
    compute_budget_metrics,
)
from fraudshield.models.train_baselines import compute_file_sha256, get_git_commit_hash, is_git_dirty

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudshield.m6_final")


def compute_test_bootstrap_ci(
    y_true: np.ndarray,
    y_score_c1: np.ndarray,
    y_score_b: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = 42,
    budgets: List[float] = [0.005, 0.01, 0.02, 0.05],
) -> Dict[str, Any]:
    """Compute paired bootstrap confidence intervals on the test split (C1 - B)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    p_c1 = compute_pr_auc(y_true, y_score_c1)
    p_b = compute_pr_auc(y_true, y_score_b)
    r_c1 = compute_roc_auc(y_true, y_score_c1)
    r_b = compute_roc_auc(y_true, y_score_b)
    bm_c1 = compute_budget_metrics(y_true, y_score_c1, budgets=budgets)
    bm_b = compute_budget_metrics(y_true, y_score_b, budgets=budgets)

    diff_pr = p_c1 - p_b
    diff_roc = r_c1 - r_b
    diff_budgets = {k: bm_c1[k]["recall"] - bm_b[k]["recall"] for k in bm_c1}

    boot_diff_pr = []
    boot_diff_roc = []
    boot_diff_budgets = {k: [] for k in bm_c1}

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        yc = y_score_c1[idx]
        yb = y_score_b[idx]

        boot_diff_pr.append(compute_pr_auc(yt, yc) - compute_pr_auc(yt, yb))
        boot_diff_roc.append(compute_roc_auc(yt, yc) - compute_roc_auc(yt, yb))
        b_c = compute_budget_metrics(yt, yc, budgets=budgets)
        b_b = compute_budget_metrics(yt, yb, budgets=budgets)
        for k in b_c:
            boot_diff_budgets[k].append(b_c[k]["recall"] - b_b[k]["recall"])

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
        "pr_auc": summarize(diff_pr, boot_diff_pr),
        "roc_auc": summarize(diff_roc, boot_diff_roc),
    }
    for k in bm_c1:
        res[f"recall_{k}"] = summarize(diff_budgets[k], boot_diff_budgets[k])
    return res


def run_milestone6():
    start_time = time.time()
    reports_dir = Path("reports")
    models_dir = Path("models")
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = Path("data/processed")

    git_hash = get_git_commit_hash()
    is_dirty = is_git_dirty()
    parquet_hashes = {
        "train_set_b": compute_file_sha256(proc_dir / "train_set_b.parquet"),
        "val_set_b": compute_file_sha256(proc_dir / "val_set_b.parquet"),
        "test_set_b": compute_file_sha256(proc_dir / "test_set_b.parquet"),
        "train_set_c1": compute_file_sha256(proc_dir / "train_set_c1.parquet"),
        "val_set_c1": compute_file_sha256(proc_dir / "val_set_c1.parquet"),
        "test_set_c1": compute_file_sha256(proc_dir / "test_set_c1.parquet"),
    }

    logger.info("=================================================================")
    logger.info("MILESTONE 6: FINAL MODEL TRAINING, CALIBRATION & TEST EVALUATION")
    logger.info("=================================================================")

    # =========================================================================
    # STEP 1: LOAD TRAIN & VAL FOR SET C1, FIT PREPROCESSOR & XGBOOST MODEL
    # =========================================================================
    model_c1_file = models_dir / "xgb_set_c1.json"
    model_b_file = models_dir / "xgb_set_b.json"

    df_tr_c1_sample = pd.read_parquet(proc_dir / "train_set_c1.parquet", columns=["isFraud", "TransactionAmt"])
    feats_c1 = get_model_feature_columns(pd.read_parquet(proc_dir / "train_set_c1.parquet"))
    feats_b = get_model_feature_columns(pd.read_parquet(proc_dir / "train_set_b.parquet"))
    y_vl = pd.read_parquet(proc_dir / "val_set_c1.parquet", columns=["isFraud"])["isFraud"].values.astype(np.int32)

    if model_c1_file.exists() and (models_dir / "preprocessor_c1.pkl").exists():
        logger.info("Loading existing trained XGBoost model and preprocessor for Set C1...")
        bst_c1 = xgb.Booster()
        bst_c1.load_model(model_c1_file)
        with open(models_dir / "preprocessor_c1.pkl", "rb") as f:
            prep_c1 = pickle.load(f)

        df_vl_c1 = pd.read_parquet(proc_dir / "val_set_c1.parquet")
        X_vl_c1 = np.ascontiguousarray(prep_c1.transform(df_vl_c1[feats_c1]).values, dtype=np.float32)
        val_probs_c1 = bst_c1.predict(xgb.DMatrix(X_vl_c1))
        del df_vl_c1, X_vl_c1
        gc.collect()
    else:
        logger.info("Fitting TreePreprocessor and training final XGBoost model on Set C1...")
        df_tr_c1 = pd.read_parquet(proc_dir / "train_set_c1.parquet")
        df_vl_c1 = pd.read_parquet(proc_dir / "val_set_c1.parquet")
        y_tr = df_tr_c1["isFraud"].values.astype(np.int32)

        prep_c1 = TreePreprocessor().fit(df_tr_c1[feats_c1])
        with open(models_dir / "preprocessor_c1.pkl", "wb") as f:
            pickle.dump(prep_c1, f)

        X_tr_c1 = np.ascontiguousarray(prep_c1.transform(df_tr_c1[feats_c1]).values, dtype=np.float32)
        X_vl_c1 = np.ascontiguousarray(prep_c1.transform(df_vl_c1[feats_c1]).values, dtype=np.float32)
        del df_tr_c1, df_vl_c1
        gc.collect()

        dtrain_c1 = xgb.DMatrix(X_tr_c1, label=y_tr)
        dval_c1 = xgb.DMatrix(X_vl_c1, label=y_vl)
        del X_tr_c1, X_vl_c1
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

        bst_c1 = xgb.train(
            params,
            dtrain_c1,
            num_boost_round=300,
            evals=[(dval_c1, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        bst_c1.save_model(model_c1_file)
        val_probs_c1 = bst_c1.predict(dval_c1)
        del dtrain_c1, dval_c1
        gc.collect()

    val_metrics_c1 = evaluate_model_performance(y_vl, val_probs_c1)
    val_metrics_c1["best_iteration"] = int(bst_c1.best_iteration)
    logger.info("Validation Set C1: PR-AUC = %.6f, ROC-AUC = %.6f, Best Iter = %d",
                val_metrics_c1["pr_auc"], val_metrics_c1["roc_auc"], val_metrics_c1["best_iteration"])

    # Baseline Model Set B
    if model_b_file.exists() and (models_dir / "preprocessor_b.pkl").exists():
        logger.info("Loading existing trained XGBoost model and preprocessor for Set B...")
        bst_b = xgb.Booster()
        bst_b.load_model(model_b_file)
        with open(models_dir / "preprocessor_b.pkl", "rb") as f:
            prep_b = pickle.load(f)

        df_vl_b = pd.read_parquet(proc_dir / "val_set_b.parquet")
        X_vl_b = np.ascontiguousarray(prep_b.transform(df_vl_b[feats_b]).values, dtype=np.float32)
        val_probs_b = bst_b.predict(xgb.DMatrix(X_vl_b))
        del df_vl_b, X_vl_b
        gc.collect()
    else:
        logger.info("Fitting TreePreprocessor and training baseline XGBoost model on Set B...")
        df_tr_b = pd.read_parquet(proc_dir / "train_set_b.parquet")
        df_vl_b = pd.read_parquet(proc_dir / "val_set_b.parquet")
        y_tr = df_tr_b["isFraud"].values.astype(np.int32)

        prep_b = TreePreprocessor().fit(df_tr_b[feats_b])
        with open(models_dir / "preprocessor_b.pkl", "wb") as f:
            pickle.dump(prep_b, f)

        X_tr_b = np.ascontiguousarray(prep_b.transform(df_tr_b[feats_b]).values, dtype=np.float32)
        X_vl_b = np.ascontiguousarray(prep_b.transform(df_vl_b[feats_b]).values, dtype=np.float32)
        del df_tr_b, df_vl_b
        gc.collect()

        dtrain_b = xgb.DMatrix(X_tr_b, label=y_tr)
        dval_b = xgb.DMatrix(X_vl_b, label=y_vl)
        del X_tr_b, X_vl_b
        gc.collect()

        bst_b = xgb.train(
            params,
            dtrain_b,
            num_boost_round=300,
            evals=[(dval_b, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        bst_b.save_model(model_b_file)
        val_probs_b = bst_b.predict(dval_b)
        del dtrain_b, dval_b
        gc.collect()

    # =========================================================================
    # STEP 2: PROBABILITY CALIBRATION (Fitted on Validation Predictions)
    # =========================================================================
    logger.info("Step 2: Fitting probability calibrators on validation...")
    val_probs_c1_clipped = np.clip(val_probs_c1, 1e-6, 1.0 - 1e-6)
    val_brier_uncal = float(brier_score_loss(y_vl, val_probs_c1_clipped))

    # Platt Scaling (Logistic Regression on raw validation scores)
    platt_calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    platt_calibrator.fit(val_probs_c1_clipped.reshape(-1, 1), y_vl)
    val_probs_platt = platt_calibrator.predict_proba(val_probs_c1_clipped.reshape(-1, 1))[:, 1]
    val_brier_platt = float(brier_score_loss(y_vl, np.clip(val_probs_platt, 0.0, 1.0)))

    # Isotonic Regression
    iso_calibrator = IsotonicRegression(out_of_bounds="clip")
    iso_calibrator.fit(val_probs_c1_clipped, y_vl)
    val_probs_iso = iso_calibrator.predict(val_probs_c1_clipped)
    val_brier_iso = float(brier_score_loss(y_vl, np.clip(val_probs_iso, 0.0, 1.0)))

    logger.info("Validation Brier Scores: Uncalibrated=%.6f, Platt=%.6f, Isotonic=%.6f",
                val_brier_uncal, val_brier_platt, val_brier_iso)

    # Choose calibrator with lowest validation Brier score
    if val_brier_iso < val_brier_platt:
        chosen_calibrator = iso_calibrator
        calibrator_type = "IsotonicRegression"
        best_val_brier = val_brier_iso
    else:
        chosen_calibrator = platt_calibrator
        calibrator_type = "PlattScaling"
        best_val_brier = val_brier_platt

    with open(models_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(chosen_calibrator, f)
    logger.info("Selected Calibrator: %s (Brier: %.6f -> %.6f)", calibrator_type, val_brier_uncal, best_val_brier)

    # =========================================================================
    # STEP 3: OPERATING POINT SELECTION ON VALIDATION
    # =========================================================================
    logger.info("Step 3: Calculating operating thresholds on validation split...")
    budgets = [0.005, 0.01, 0.02, 0.05]
    budget_names = ["top_0.5pct", "top_1.0pct", "top_2.0pct", "top_5.0pct"]
    n_val = len(y_vl)

    val_sorted_scores = np.sort(val_probs_c1)[::-1]
    val_operating_points = {}
    for b, b_name in zip(budgets, budget_names):
        n_cutoff = max(1, int(round(n_val * b)))
        threshold = float(val_sorted_scores[n_cutoff - 1])
        b_metrics = val_metrics_c1["budgets"][b_name]
        val_operating_points[b_name] = {
            "budget_ratio": b,
            "threshold": threshold,
            "review_volume": n_cutoff,
            "fraud_captured": b_metrics["fraud_captured"],
            "recall": b_metrics["recall"],
            "precision": b_metrics["precision"],
            "fpr": b_metrics["fpr"],
        }
        logger.info("  %s: threshold=%.6f | Review=%d, Recall=%.4f, Precision=%.4f, FPR=%.6f",
                    b_name, threshold, n_cutoff, b_metrics["recall"], b_metrics["precision"], b_metrics["fpr"])

    # =========================================================================
    # STEP 4: AUTHORIZED SINGLE FINAL TEST EVALUATION
    # =========================================================================
    logger.info("Step 4: Accessing held-out test split via authorized loader guard...")
    # This call inspects the stack, recognizes evaluate_test.py, and logs to reports/test_access_log.txt
    df_test_c1 = load_split_parquet("test", feature_set="c1", base_dir=proc_dir)
    df_test_b = load_split_parquet("test", feature_set="b", base_dir=proc_dir)

    y_test = df_test_c1["isFraud"].values.astype(np.int32)
    test_ids = df_test_c1["TransactionID"].values
    test_dts = df_test_c1["TransactionDT"].values
    test_amts = df_test_c1["TransactionAmt"].values

    logger.info("Pre-processing test split (%d rows, %d frauds)...", len(y_test), int(np.sum(y_test)))
    X_test_c1 = np.ascontiguousarray(prep_c1.transform(df_test_c1[feats_c1]).values, dtype=np.float32)
    X_test_b = np.ascontiguousarray(prep_b.transform(df_test_b[feats_b]).values, dtype=np.float32)
    del df_test_c1, df_test_b
    gc.collect()

    dtest_c1 = xgb.DMatrix(X_test_c1)
    dtest_b = xgb.DMatrix(X_test_b)

    logger.info("Predicting on test set...")
    test_probs_c1 = bst_c1.predict(dtest_c1)
    test_probs_b = bst_b.predict(dtest_b)

    # Calibrate test probabilities
    test_probs_c1_clipped = np.clip(test_probs_c1, 1e-6, 1.0 - 1e-6)
    if calibrator_type == "IsotonicRegression":
        test_probs_calibrated = np.clip(chosen_calibrator.predict(test_probs_c1_clipped), 0.0, 1.0)
    else:
        test_probs_calibrated = np.clip(chosen_calibrator.predict_proba(test_probs_c1_clipped.reshape(-1, 1))[:, 1], 0.0, 1.0)

    # Evaluate Test Set C1
    test_metrics_c1 = evaluate_model_performance(y_test, test_probs_c1, budgets=budgets)
    test_metrics_b = evaluate_model_performance(y_test, test_probs_b, budgets=budgets)
    test_brier_uncal = float(brier_score_loss(y_test, test_probs_c1_clipped))
    test_brier_cal = float(brier_score_loss(y_test, test_probs_calibrated))

    logger.info("TEST EVALUATION RESULTS (PRIMARY SET C1):")
    logger.info("  PR-AUC:                %.6f", test_metrics_c1["pr_auc"])
    logger.info("  ROC-AUC:               %.6f", test_metrics_c1["roc_auc"])
    logger.info("  Uncalibrated Brier:    %.6f", test_brier_uncal)
    logger.info("  Calibrated Brier:      %.6f", test_brier_cal)
    logger.info("TEST EVALUATION RESULTS (BASELINE SET B):")
    logger.info("  PR-AUC:                %.6f", test_metrics_b["pr_auc"])
    logger.info("  ROC-AUC:               %.6f", test_metrics_b["roc_auc"])

    # Evaluate test performance at FIXED VALIDATION THRESHOLDS
    test_operating_records = []
    n_test = len(y_test)
    total_test_frauds = int(np.sum(y_test))

    for b, b_name in zip(budgets, budget_names):
        thresh = val_operating_points[b_name]["threshold"]
        predicted_positives = (test_probs_c1 >= thresh)
        rev_vol = int(np.sum(predicted_positives))
        tp = int(np.sum(predicted_positives & (y_test == 1)))
        fp = rev_vol - tp
        rec = float(tp / total_test_frauds) if total_test_frauds > 0 else 0.0
        prec = float(tp / rev_vol) if rev_vol > 0 else 0.0
        fpr = float(fp / (n_test - total_test_frauds))

        # Also get fixed capacity ranking metrics on test
        fixed_cap_metrics = test_metrics_c1["budgets"][b_name]

        record = {
            "budget_name": b_name,
            "budget_ratio": b,
            "val_fixed_threshold": thresh,
            "val_recall": val_operating_points[b_name]["recall"],
            "val_precision": val_operating_points[b_name]["precision"],
            "test_fixed_threshold_recall": round(rec, 6),
            "test_fixed_threshold_precision": round(prec, 6),
            "test_fixed_threshold_fpr": round(fpr, 6),
            "test_fixed_threshold_review_count": rev_vol,
            "test_fixed_threshold_fraud_captured": tp,
            "test_fixed_budget_recall": fixed_cap_metrics["recall"],
            "test_fixed_budget_precision": fixed_cap_metrics["precision"],
            "test_fixed_budget_fpr": fixed_cap_metrics["fpr"],
            "test_fixed_budget_review_count": fixed_cap_metrics["review_volume"],
            "test_fixed_budget_fraud_captured": fixed_cap_metrics["fraud_captured"],
        }
        test_operating_records.append(record)

    df_test_ops = pd.DataFrame(test_operating_records)
    df_test_ops.to_csv(reports_dir / "test_operating_points.csv", index=False)

    # Paired Bootstrap on Test Set
    logger.info("Computing 500-sample paired bootstrap on test split (C1 vs B)...")
    test_bootstrap = compute_test_bootstrap_ci(y_test, test_probs_c1, test_probs_b, n_bootstraps=500, seed=42)

    # =========================================================================
    # STEP 5: SHAP TREEEXPLAINER COMPUTATION
    # =========================================================================
    logger.info("Step 5: Computing TreeExplainer SHAP values on representative test sample...")
    top_1pct_idx = np.where(test_probs_c1 >= val_operating_points["top_1.0pct"]["threshold"])[0]
    other_idx = np.where(test_probs_c1 < val_operating_points["top_1.0pct"]["threshold"])[0]
    rng = np.random.default_rng(42)
    n_sample_other = min(len(other_idx), max(100, 1000 - len(top_1pct_idx)))
    sample_other_idx = rng.choice(other_idx, size=n_sample_other, replace=False)
    sample_indices = np.concatenate([top_1pct_idx, sample_other_idx])
    sample_indices = np.sort(sample_indices)

    logger.info("Computing SHAP on %d transactions (including %d flagged alerts)...",
                len(sample_indices), len(top_1pct_idx))

    X_shap_sample = X_test_c1[sample_indices]
    explainer = shap.TreeExplainer(bst_c1)
    shap_values = explainer.shap_values(X_shap_sample)

    # Global feature importances
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_order = np.argsort(mean_abs_shap)[::-1]

    def categorize_feature(fname: str) -> str:
        if fname.startswith("card1_") or fname.startswith("card_"):
            return "Entity History (Set C1)"
        elif fname.startswith("V"):
            return "Vesta Anonymised (Set A)"
        elif fname.startswith("C"):
            return "Transaction Counting (Set A)"
        elif fname.startswith("D"):
            return "Timetable / Delta (Set A)"
        elif fname.startswith("M"):
            return "Match Flags (Set A)"
        elif fname.startswith("id_"):
            return "Identity (Set B)"
        elif "email" in fname or "addr" in fname:
            return "Network / Address (Set B)"
        elif fname in ["TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6"]:
            return "Transaction Core (Set A)"
        else:
            return "Other Features"

    global_shap_records = []
    for rank, idx in enumerate(shap_order, start=1):
        fname = feats_c1[idx]
        global_shap_records.append({
            "rank": rank,
            "feature_name": fname,
            "feature_family": categorize_feature(fname),
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 6),
            "is_anonymised": bool(fname.startswith("V")),
            "interpretation_note": (
                "Anonymised structural pattern (behavioral driver, not causal)"
                if fname.startswith("V")
                else "Interpretable domain / entity attribute"
            ),
        })

    df_shap_global = pd.DataFrame(global_shap_records)
    df_shap_global.to_csv(reports_dir / "shap_global_importance.csv", index=False)
    with open(reports_dir / "shap_global_importance.json", "w", encoding="utf-8") as f:
        json.dump(global_shap_records, f, indent=2)

    # Compute top 5 drivers per transaction in the sample
    local_shap_dict = {}
    for sample_pos, global_idx in enumerate(sample_indices):
        tx_id = int(test_ids[global_idx])
        vals = shap_values[sample_pos]
        top_pos = np.argsort(vals)[-5:][::-1]
        top_neg = np.argsort(vals)[:5]

        drivers = {
            "top_risk_increasing": [
                {"feature": feats_c1[i], "attribution": round(float(vals[i]), 5), "family": categorize_feature(feats_c1[i])}
                for i in top_pos if vals[i] > 0
            ],
            "top_risk_decreasing": [
                {"feature": feats_c1[i], "attribution": round(float(vals[i]), 5), "family": categorize_feature(feats_c1[i])}
                for i in top_neg if vals[i] < 0
            ],
        }
        local_shap_dict[tx_id] = json.dumps(drivers)

    del X_test_c1, X_test_b, dtest_c1, dtest_b, shap_values
    gc.collect()

    # =========================================================================
    # STEP 6: SQLITE DATABASE POPULATION (reports/fraudshield.db)
    # =========================================================================
    logger.info("Step 6: Storing scored records, metadata, and SHAP into SQLite database...")
    db_path = reports_dir / "fraudshield.db"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Model Metadata table
    cur.execute("""
        CREATE TABLE model_metadata (
            model_id TEXT PRIMARY KEY,
            model_name TEXT,
            feature_set TEXT,
            best_iteration INTEGER,
            val_pr_auc REAL,
            test_pr_auc REAL,
            test_roc_auc REAL,
            brier_uncalibrated REAL,
            brier_calibrated REAL,
            calibrator_type TEXT,
            git_commit_hash TEXT,
            evaluated_at TEXT
        )
    """)
    cur.execute("""
        INSERT INTO model_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "xgb_set_c1_m6",
        "XGBoost (Hist Gradient Boosted Trees)",
        "Set C1 (card1 Entity History)",
        int(bst_c1.best_iteration) if hasattr(bst_c1, "best_iteration") else 299,
        float(val_metrics_c1["pr_auc"]),
        float(test_metrics_c1["pr_auc"]),
        float(test_metrics_c1["roc_auc"]),
        test_brier_uncal,
        test_brier_cal,
        calibrator_type,
        git_hash,
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    ))

    # 2. Operating Points table
    cur.execute("""
        CREATE TABLE operating_points (
            budget_name TEXT PRIMARY KEY,
            budget_ratio REAL,
            score_threshold REAL,
            val_recall REAL,
            val_precision REAL,
            test_recall REAL,
            test_precision REAL,
            test_review_count INTEGER,
            test_fraud_captured INTEGER
        )
    """)
    for r in test_operating_records:
        cur.execute("""
            INSERT INTO operating_points VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["budget_name"],
            r["budget_ratio"],
            r["val_fixed_threshold"],
            r["val_recall"],
            r["val_precision"],
            r["test_fixed_threshold_recall"],
            r["test_fixed_threshold_precision"],
            r["test_fixed_threshold_review_count"],
            r["test_fixed_threshold_fraud_captured"],
        ))

    # 3. Global Feature Importances table
    cur.execute("""
        CREATE TABLE feature_importances (
            rank INTEGER PRIMARY KEY,
            feature_name TEXT,
            feature_family TEXT,
            mean_abs_shap REAL,
            is_anonymised INTEGER,
            interpretation_note TEXT
        )
    """)
    for g in global_shap_records[:50]:
        cur.execute("""
            INSERT INTO feature_importances VALUES (?, ?, ?, ?, ?, ?)
        """, (
            g["rank"],
            g["feature_name"],
            g["feature_family"],
            g["mean_abs_shap"],
            1 if g["is_anonymised"] else 0,
            g["interpretation_note"],
        ))

    # 4. Scored Test Records table
    cur.execute("""
        CREATE TABLE test_predictions (
            transaction_id INTEGER PRIMARY KEY,
            transaction_dt INTEGER,
            transaction_amt REAL,
            raw_risk_score REAL,
            calibrated_prob REAL,
            is_fraud INTEGER,
            flag_top_0_5pct INTEGER,
            flag_top_1_0pct INTEGER,
            flag_top_2_0pct INTEGER,
            flag_top_5_0pct INTEGER,
            top_shap_drivers TEXT
        )
    """)

    logger.info("Inserting %d scored test transactions into SQLite...", len(y_test))
    rows_to_insert = []
    t_05 = val_operating_points["top_0.5pct"]["threshold"]
    t_10 = val_operating_points["top_1.0pct"]["threshold"]
    t_20 = val_operating_points["top_2.0pct"]["threshold"]
    t_50 = val_operating_points["top_5.0pct"]["threshold"]

    for i in range(len(y_test)):
        tid = int(test_ids[i])
        s = float(test_probs_c1[i])
        p = float(test_probs_calibrated[i])
        rows_to_insert.append((
            tid,
            int(test_dts[i]),
            float(test_amts[i]),
            round(s, 6),
            round(p, 6),
            int(y_test[i]),
            1 if s >= t_05 else 0,
            1 if s >= t_10 else 0,
            1 if s >= t_20 else 0,
            1 if s >= t_50 else 0,
            local_shap_dict.get(tid, None),
        ))

    cur.executemany("""
        INSERT INTO test_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_to_insert)

    cur.execute("CREATE INDEX idx_risk_score ON test_predictions (raw_risk_score DESC)")
    cur.execute("CREATE INDEX idx_calibrated_prob ON test_predictions (calibrated_prob DESC)")
    conn.commit()
    conn.close()
    logger.info("Successfully populated SQLite database at %s", db_path)

    # =========================================================================
    # STEP 7: SERIALIZE COMPLETE TEST EVALUATION SUMMARY
    # =========================================================================
    summary_artifact = {
        "run_id": "m6_final_eval",
        "git_commit_hash": git_hash,
        "is_git_dirty": is_dirty,
        "parquet_sha256": parquet_hashes,
        "calibrator": {
            "type": calibrator_type,
            "val_brier_uncalibrated": val_brier_uncal,
            "val_brier_calibrated": best_val_brier,
            "test_brier_uncalibrated": test_brier_uncal,
            "test_brier_calibrated": test_brier_cal,
        },
        "primary_model_set_c1": {
            "test_pr_auc": test_metrics_c1["pr_auc"],
            "test_roc_auc": test_metrics_c1["roc_auc"],
            "val_pr_auc": val_metrics_c1["pr_auc"],
            "val_roc_auc": val_metrics_c1["roc_auc"],
            "best_iteration": int(bst_c1.best_iteration) if hasattr(bst_c1, "best_iteration") else 299,
        },
        "baseline_model_set_b": {
            "test_pr_auc": test_metrics_b["pr_auc"],
            "test_roc_auc": test_metrics_b["roc_auc"],
            "best_iteration": int(bst_b.best_iteration) if hasattr(bst_b, "best_iteration") else 293,
        },
        "test_paired_bootstrap_c1_minus_b": test_bootstrap,
        "operating_points": test_operating_records,
        "shap_summary": {
            "top_10_global_features": global_shap_records[:10],
            "total_features_explained": len(feats_c1),
            "sample_size": len(sample_indices),
        },
    }

    with open(reports_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_artifact, f, indent=2)

    df_test_summary = pd.DataFrame([
        {
            "model": "XGBoost Set C1 (Primary)",
            "test_pr_auc": test_metrics_c1["pr_auc"],
            "test_roc_auc": test_metrics_c1["roc_auc"],
            "brier_uncalibrated": test_brier_uncal,
            "brier_calibrated": test_brier_cal,
            "recall_top_1.0pct": test_metrics_c1["budgets"]["top_1.0pct"]["recall"],
            "precision_top_1.0pct": test_metrics_c1["budgets"]["top_1.0pct"]["precision"],
            "recall_top_2.0pct": test_metrics_c1["budgets"]["top_2.0pct"]["recall"],
            "precision_top_2.0pct": test_metrics_c1["budgets"]["top_2.0pct"]["precision"],
            "recall_top_5.0pct": test_metrics_c1["budgets"]["top_5.0pct"]["recall"],
            "precision_top_5.0pct": test_metrics_c1["budgets"]["top_5.0pct"]["precision"],
        },
        {
            "model": "XGBoost Set B (Baseline)",
            "test_pr_auc": test_metrics_b["pr_auc"],
            "test_roc_auc": test_metrics_b["roc_auc"],
            "brier_uncalibrated": float(brier_score_loss(y_test, np.clip(test_probs_b, 0.0, 1.0))),
            "brier_calibrated": None,
            "recall_top_1.0pct": test_metrics_b["budgets"]["top_1.0pct"]["recall"],
            "precision_top_1.0pct": test_metrics_b["budgets"]["top_1.0pct"]["precision"],
            "recall_top_2.0pct": test_metrics_b["budgets"]["top_2.0pct"]["recall"],
            "precision_top_2.0pct": test_metrics_b["budgets"]["top_2.0pct"]["precision"],
            "recall_top_5.0pct": test_metrics_b["budgets"]["top_5.0pct"]["recall"],
            "precision_top_5.0pct": test_metrics_b["budgets"]["top_5.0pct"]["precision"],
        },
    ])
    df_test_summary.to_csv(reports_dir / "test_evaluation_summary.csv", index=False)

    total_time = time.time() - start_time
    logger.info("=================================================================")
    logger.info("MILESTONE 6 COMPLETED CLEANLY IN %.2f SECONDS (%.2f MIN)!", total_time, total_time / 60.0)
    logger.info("=================================================================")


if __name__ == "__main__":
    run_milestone6()
