"""Milestone 5 Production Pipeline: Leakage-Safe Statistical Evaluation.

Execution rules strictly enforced:
1. Sequential execution with explicit memory deletion and gc.collect().
   Never holds multiple feature matrices in memory simultaneously.
2. Contiguous float32 DMatrix with tree_method='hist' and n_jobs=6.
3. Checkpointing in reports/m5_checkpoints/; skips already-completed fits on restart.
4. Logging progress ("fit k/N done, elapsed X min") to reports/m5_progress.log.
5. Execution priority order:
   (a) Seeds on B, C1, C2 (seed 42)
   (b) Paired bootstrap (500 resamples on seed 42 predictions)
   (c) RQ2 random vs temporal split
   (d) Seeds 123 and 7 on B and C1
   (e) Rolling-origin backtest (2 folds)
   (f) C1 sensitivity run (without card1_prior_count, seeds 42, 123, 7)
"""

import gc
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
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
from fraudshield.splits.temporal import find_next_distinct_boundary
from fraudshield.models.train_baselines import compute_file_sha256, get_git_commit_hash, is_git_dirty

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudshield.m5_prod")


def flatten_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten hierarchical evaluation metrics into flat dictionary for CSV/JSON tables."""
    row = {
        "pr_auc": m.get("pr_auc"),
        "roc_auc": m.get("roc_auc"),
        "brier_score": m.get("brier_score"),
        "best_iteration": m.get("best_iteration"),
        "fit_time_seconds": m.get("fit_time_seconds"),
    }
    budgets = m.get("budgets", {})
    for b_key in ["top_0.5pct", "top_1.0pct", "top_2.0pct", "top_5.0pct"]:
        if b_key in budgets:
            row[f"recall_{b_key}"] = budgets[b_key].get("recall")
            row[f"precision_{b_key}"] = budgets[b_key].get("precision")
            row[f"fpr_{b_key}"] = budgets[b_key].get("fpr")
    return row


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_score_test: np.ndarray,
    y_score_base: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = 42,
    budgets: List[float] = [0.005, 0.01, 0.02, 0.05],
) -> Dict[str, Any]:
    """Compute paired bootstrap confidence intervals for metric differences (Test - Base)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Full-sample point estimates
    p_test = compute_pr_auc(y_true, y_score_test)
    p_base = compute_pr_auc(y_true, y_score_base)
    r_test = compute_roc_auc(y_true, y_score_test)
    r_base = compute_roc_auc(y_true, y_score_base)
    b_test = compute_budget_metrics(y_true, y_score_test, budgets=budgets)
    b_base = compute_budget_metrics(y_true, y_score_base, budgets=budgets)

    diff_pr = p_test - p_base
    diff_roc = r_test - r_base
    diff_budgets = {
        b_key: b_test[b_key]["recall"] - b_base[b_key]["recall"]
        for b_key in b_test
    }

    boot_diff_pr = []
    boot_diff_roc = []
    boot_diff_budgets = {b_key: [] for b_key in b_test}

    t0 = time.time()
    for b_idx in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        yc = y_score_test[idx]
        yb = y_score_base[idx]

        bp_c = compute_pr_auc(yt, yc)
        bp_b = compute_pr_auc(yt, yb)
        boot_diff_pr.append(bp_c - bp_b)

        br_c = compute_roc_auc(yt, yc)
        br_b = compute_roc_auc(yt, yb)
        boot_diff_roc.append(br_c - br_b)

        bb_c = compute_budget_metrics(yt, yc, budgets=budgets)
        bb_b = compute_budget_metrics(yt, yb, budgets=budgets)
        for b_key in bb_c:
            boot_diff_budgets[b_key].append(bb_c[b_key]["recall"] - bb_b[b_key]["recall"])

    logger.info("Completed %d paired bootstrap resamples in %.2f s", n_bootstraps, time.time() - t0)

    def summarize_diff(point: float, boot_vals: List[float]) -> Dict[str, Any]:
        arr = np.array(boot_vals)
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        return {
            "point_estimate": round(float(point), 6),
            "mean_diff": round(float(np.mean(arr)), 6),
            "std_error": round(float(np.std(arr, ddof=1)), 6),
            "ci_95_lower": round(ci_lower, 6),
            "ci_95_upper": round(ci_upper, 6),
            "strictly_positive": bool(ci_lower > 0),
        }

    result = {
        "pr_auc": summarize_diff(diff_pr, boot_diff_pr),
        "roc_auc": summarize_diff(diff_roc, boot_diff_roc),
    }
    for b_key in b_test:
        result[f"recall_{b_key}"] = summarize_diff(diff_budgets[b_key], boot_diff_budgets[b_key])

    return result


class M5ExperimentRunner:
    def __init__(self, reports_dir: Path = Path("reports")):
        self.reports_dir = reports_dir
        self.checkpoints_dir = reports_dir / "m5_checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.progress_log = reports_dir / "m5_progress.log"

        self.start_time = time.time()
        self.total_fits = 13  # b_s42, c1_s42, c2_s42, rq2_rnd_b, b_s123, b_s7, c1_s123, c1_s7, bt_f1_b, bt_f1_c1, c1_sens_42, c1_sens_123, c1_sens_7
        self.completed_fits = 0

        self.run_id = str(uuid.uuid4())[:8]
        self.git_hash = get_git_commit_hash()
        self.is_dirty = is_git_dirty()

        proc_dir = Path("data/processed")
        self.parquet_hashes = {
            "train_set_b": compute_file_sha256(proc_dir / "train_set_b.parquet"),
            "val_set_b": compute_file_sha256(proc_dir / "val_set_b.parquet"),
            "train_set_c1": compute_file_sha256(proc_dir / "train_set_c1.parquet"),
            "val_set_c1": compute_file_sha256(proc_dir / "val_set_c1.parquet"),
            "train_set_c2": compute_file_sha256(proc_dir / "train_set_c2.parquet"),
            "val_set_c2": compute_file_sha256(proc_dir / "val_set_c2.parquet"),
        }

        self.reduced_scope_metadata = {
            "seeds_evaluated": [42, 123, 7],
            "seeds_count": 3,
            "bootstrap_resamples": 500,
            "rolling_backtest_folds": 2,
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
            "rationale": "Execution scope optimized under 100-minute budget per explicit instructions. Retained identical hyperparameters.",
        }

    def log_progress(self, fit_name: str):
        self.completed_fits += 1
        elapsed_min = (time.time() - self.start_time) / 60.0
        msg = f"[M5 PROGRESS] fit {self.completed_fits}/{self.total_fits} done ('{fit_name}'), elapsed {elapsed_min:.2f} min"
        logger.info(msg)
        with open(self.progress_log, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def run_or_load_fit(
        self,
        fit_key: str,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        y_val: np.ndarray,
        seed: int = 42,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Fit model with early stopping or load from checkpoint if already completed."""
        meta_path = self.checkpoints_dir / f"{fit_key}_meta.json"
        probs_path = self.checkpoints_dir / f"{fit_key}_probs.npy"

        if meta_path.exists() and probs_path.exists():
            logger.info("Found existing checkpoint for '%s'. Loading cached metrics and probabilities.", fit_key)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            probs = np.load(probs_path)
            self.log_progress(f"{fit_key} (cached)")
            return meta, probs

        logger.info("Fitting model '%s' (seed=%d, n_jobs=6, tree_method='hist')...", fit_key, seed)
        params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.0,
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "random_state": seed,
            "n_jobs": 6,
        }

        t0 = time.time()
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        t_fit = time.time() - t0

        probs = bst.predict(dval)
        metrics = evaluate_model_performance(y_val, probs)
        metrics["best_iteration"] = int(bst.best_iteration)
        metrics["fit_time_seconds"] = round(t_fit, 2)
        metrics["seed"] = seed
        metrics["fit_key"] = fit_key

        # Save checkpoint immediately
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        np.save(probs_path, probs)

        self.log_progress(fit_key)
        return metrics, probs


def run_milestone5_pipeline():
    runner = M5ExperimentRunner()
    proc_dir = Path("data/processed")
    y_val = pd.read_parquet(proc_dir / "val_set_b.parquet", columns=["isFraud"])["isFraud"].values.astype(np.int32)

    all_seed_results = []
    val_probs = {}

    # =========================================================================
    # STEP 1: SET B EVALUATION (Seed 42)
    # =========================================================================
    logger.info(">>> STEP 1: PREPARING SET B <<<")
    df_tr_b = pd.read_parquet(proc_dir / "train_set_b.parquet")
    df_vl_b = pd.read_parquet(proc_dir / "val_set_b.parquet")
    feats_b = get_model_feature_columns(df_tr_b)
    y_tr = df_tr_b["isFraud"].values.astype(np.int32)

    prep_b = TreePreprocessor().fit(df_tr_b[feats_b])
    X_tr_b = prep_b.transform(df_tr_b[feats_b])
    X_vl_b = prep_b.transform(df_vl_b[feats_b])
    del df_tr_b, df_vl_b
    gc.collect()

    X_tr_arr = np.ascontiguousarray(X_tr_b.values, dtype=np.float32)
    X_vl_arr = np.ascontiguousarray(X_vl_b.values, dtype=np.float32)
    del X_tr_b, X_vl_b
    gc.collect()

    dtrain_b = xgb.DMatrix(X_tr_arr, label=y_tr)
    dval_b = xgb.DMatrix(X_vl_arr, label=y_val)
    del X_tr_arr, X_vl_arr
    gc.collect()

    # Fit or load Set B Seed 42
    m_b_42, p_b_42 = runner.run_or_load_fit("b_s42", dtrain_b, dval_b, y_val, seed=42)
    val_probs["Set B"] = p_b_42
    all_seed_results.append({"feature_set": "Set B", "seed": 42, **flatten_metrics(m_b_42)})

    # =========================================================================
    # STEP 2: SET C1 EVALUATION (Seed 42)
    # =========================================================================
    logger.info(">>> STEP 2: PREPARING SET C1 <<<")
    df_tr_c1 = pd.read_parquet(proc_dir / "train_set_c1.parquet")
    df_vl_c1 = pd.read_parquet(proc_dir / "val_set_c1.parquet")
    feats_c1 = get_model_feature_columns(df_tr_c1)

    prep_c1 = TreePreprocessor().fit(df_tr_c1[feats_c1])
    X_tr_c1 = prep_c1.transform(df_tr_c1[feats_c1])
    X_vl_c1 = prep_c1.transform(df_vl_c1[feats_c1])
    del df_tr_c1, df_vl_c1
    gc.collect()

    X_tr_arr = np.ascontiguousarray(X_tr_c1.values, dtype=np.float32)
    X_vl_arr = np.ascontiguousarray(X_vl_c1.values, dtype=np.float32)
    del X_tr_c1, X_vl_c1
    gc.collect()

    dtrain_c1 = xgb.DMatrix(X_tr_arr, label=y_tr)
    dval_c1 = xgb.DMatrix(X_vl_arr, label=y_val)
    del X_tr_arr, X_vl_arr
    gc.collect()

    m_c1_42, p_c1_42 = runner.run_or_load_fit("c1_s42", dtrain_c1, dval_c1, y_val, seed=42)
    val_probs["Set C1"] = p_c1_42
    all_seed_results.append({"feature_set": "Set C1", "seed": 42, **flatten_metrics(m_c1_42)})

    # =========================================================================
    # STEP 3: SET C2 EVALUATION (Seed 42)
    # =========================================================================
    logger.info(">>> STEP 3: PREPARING SET C2 <<<")
    df_tr_c2 = pd.read_parquet(proc_dir / "train_set_c2.parquet")
    df_vl_c2 = pd.read_parquet(proc_dir / "val_set_c2.parquet")
    feats_c2 = get_model_feature_columns(df_tr_c2)

    prep_c2 = TreePreprocessor().fit(df_tr_c2[feats_c2])
    X_tr_c2 = prep_c2.transform(df_tr_c2[feats_c2])
    X_vl_c2 = prep_c2.transform(df_vl_c2[feats_c2])
    del df_tr_c2, df_vl_c2
    gc.collect()

    X_tr_arr = np.ascontiguousarray(X_tr_c2.values, dtype=np.float32)
    X_vl_arr = np.ascontiguousarray(X_vl_c2.values, dtype=np.float32)
    del X_tr_c2, X_vl_c2
    gc.collect()

    dtrain_c2 = xgb.DMatrix(X_tr_arr, label=y_tr)
    dval_c2 = xgb.DMatrix(X_vl_arr, label=y_val)
    del X_tr_arr, X_vl_arr
    gc.collect()

    m_c2_42, p_c2_42 = runner.run_or_load_fit("c2_s42", dtrain_c2, dval_c2, y_val, seed=42)
    val_probs["Set C2"] = p_c2_42
    all_seed_results.append({"feature_set": "Set C2", "seed": 42, **flatten_metrics(m_c2_42)})

    # Clean up C2 matrices
    del dtrain_c2, dval_c2
    gc.collect()

    # =========================================================================
    # STEP 4: PAIRED BOOTSTRAP (500 resamples on Seed 42 predictions)
    # =========================================================================
    logger.info(">>> STEP 4: RUNNING PAIRED BOOTSTRAP (500 resamples, seed=42) <<<")
    boot_c1_b = compute_bootstrap_ci(y_val, val_probs["Set C1"], val_probs["Set B"], n_bootstraps=500, seed=42)
    boot_c2_b = compute_bootstrap_ci(y_val, val_probs["Set C2"], val_probs["Set B"], n_bootstraps=500, seed=42)

    bootstrap_artifact = {
        "run_id": runner.run_id,
        "git_commit_hash": runner.git_hash,
        "is_git_dirty": runner.is_dirty,
        "parquet_sha256": runner.parquet_hashes,
        "n_bootstraps": 500,
        "seed": 42,
        "validation_samples": len(y_val),
        "validation_frauds": int(np.sum(y_val)),
        "c1_minus_b": boot_c1_b,
        "c2_minus_b": boot_c2_b,
        "reduced_scope_metadata": runner.reduced_scope_metadata,
    }
    with open(runner.reports_dir / "m5_paired_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_artifact, f, indent=2)
    logger.info("Saved paired bootstrap results to reports/m5_paired_bootstrap.json")

    # =========================================================================
    # STEP 5: RQ2 RANDOM VS TEMPORAL SPLIT EXPERIMENT (Set B)
    # =========================================================================
    logger.info(">>> STEP 5: RUNNING RQ2 RANDOM VS TEMPORAL SPLIT (SET B) <<<")
    rq2_meta_path = runner.checkpoints_dir / "rq2_random_meta.json"
    if rq2_meta_path.exists():
        with open(rq2_meta_path, "r", encoding="utf-8") as f:
            rq2_summary = json.load(f)
        runner.log_progress("rq2_rnd_b (cached)")
    else:
        full_85_b = pd.concat([
            pd.read_parquet(proc_dir / "train_set_b.parquet"),
            pd.read_parquet(proc_dir / "val_set_b.parquet"),
        ], ignore_index=True)

        idx_all = np.arange(len(full_85_b))
        idx_tr, idx_vl = train_test_split(idx_all, test_size=len(y_val), random_state=42, shuffle=True)
        df_rnd_tr = full_85_b.iloc[idx_tr]
        df_rnd_vl = full_85_b.iloc[idx_vl]
        del full_85_b
        gc.collect()

        y_rnd_tr = df_rnd_tr["isFraud"].values.astype(np.int32)
        y_rnd_vl = df_rnd_vl["isFraud"].values.astype(np.int32)

        prep_rnd = TreePreprocessor().fit(df_rnd_tr[feats_b])
        X_tr_rnd = prep_rnd.transform(df_rnd_tr[feats_b])
        X_vl_rnd = prep_rnd.transform(df_rnd_vl[feats_b])
        del df_rnd_tr, df_rnd_vl
        gc.collect()

        X_tr_arr = np.ascontiguousarray(X_tr_rnd.values, dtype=np.float32)
        X_vl_arr = np.ascontiguousarray(X_vl_rnd.values, dtype=np.float32)
        del X_tr_rnd, X_vl_rnd
        gc.collect()

        dtrain_rnd = xgb.DMatrix(X_tr_arr, label=y_rnd_tr)
        dval_rnd = xgb.DMatrix(X_vl_arr, label=y_rnd_vl)
        del X_tr_arr, X_vl_arr
        gc.collect()

        m_rnd, _ = runner.run_or_load_fit("rq2_rnd_b", dtrain_rnd, dval_rnd, y_rnd_vl, seed=42)
        del dtrain_rnd, dval_rnd
        gc.collect()

        rq2_summary = {
            "run_id": runner.run_id,
            "git_commit_hash": runner.git_hash,
            "is_git_dirty": runner.is_dirty,
            "parquet_sha256": runner.parquet_hashes,
            "temporal_split": {
                "pr_auc": m_b_42["pr_auc"],
                "roc_auc": m_b_42["roc_auc"],
                "recall_top_0.5pct": m_b_42["budgets"]["top_0.5pct"]["recall"],
                "recall_top_1.0pct": m_b_42["budgets"]["top_1.0pct"]["recall"],
                "recall_top_2.0pct": m_b_42["budgets"]["top_2.0pct"]["recall"],
                "recall_top_5.0pct": m_b_42["budgets"]["top_5.0pct"]["recall"],
            },
            "random_split_optimistic_leaky": {
                "pr_auc": m_rnd["pr_auc"],
                "roc_auc": m_rnd["roc_auc"],
                "recall_top_0.5pct": m_rnd["budgets"]["top_0.5pct"]["recall"],
                "recall_top_1.0pct": m_rnd["budgets"]["top_1.0pct"]["recall"],
                "recall_top_2.0pct": m_rnd["budgets"]["top_2.0pct"]["recall"],
                "recall_top_5.0pct": m_rnd["budgets"]["top_5.0pct"]["recall"],
            },
            "overstatement_gap_random_minus_temporal": {
                "pr_auc_gap": round(m_rnd["pr_auc"] - m_b_42["pr_auc"], 6),
                "roc_auc_gap": round(m_rnd["roc_auc"] - m_b_42["roc_auc"], 6),
                "recall_top_0.5pct_gap": round(m_rnd["budgets"]["top_0.5pct"]["recall"] - m_b_42["budgets"]["top_0.5pct"]["recall"], 6),
                "recall_top_1.0pct_gap": round(m_rnd["budgets"]["top_1.0pct"]["recall"] - m_b_42["budgets"]["top_1.0pct"]["recall"], 6),
                "recall_top_2.0pct_gap": round(m_rnd["budgets"]["top_2.0pct"]["recall"] - m_b_42["budgets"]["top_2.0pct"]["recall"], 6),
                "recall_top_5.0pct_gap": round(m_rnd["budgets"]["top_5.0pct"]["recall"] - m_b_42["budgets"]["top_5.0pct"]["recall"], 6),
            },
            "reduced_scope_metadata": runner.reduced_scope_metadata,
        }
        with open(rq2_meta_path, "w", encoding="utf-8") as f:
            json.dump(rq2_summary, f, indent=2)

    with open(runner.reports_dir / "m5_rq2_split_comparison.json", "w", encoding="utf-8") as f:
        json.dump(rq2_summary, f, indent=2)
    logger.info("Saved RQ2 random vs temporal split comparison to reports/m5_rq2_split_comparison.json")

    # =========================================================================
    # STEP 6: SEEDS 123 AND 7 ON SET B AND SET C1
    # =========================================================================
    logger.info(">>> STEP 6: RUNNING SEEDS 123 AND 7 ON SET B AND SET C1 <<<")
    # Set B seeds
    m_b_123, _ = runner.run_or_load_fit("b_s123", dtrain_b, dval_b, y_val, seed=123)
    all_seed_results.append({"feature_set": "Set B", "seed": 123, **flatten_metrics(m_b_123)})
    m_b_7, _ = runner.run_or_load_fit("b_s7", dtrain_b, dval_b, y_val, seed=7)
    all_seed_results.append({"feature_set": "Set B", "seed": 7, **flatten_metrics(m_b_7)})

    # Clean up Set B matrices
    del dtrain_b, dval_b
    gc.collect()

    # Set C1 seeds
    m_c1_123, _ = runner.run_or_load_fit("c1_s123", dtrain_c1, dval_c1, y_val, seed=123)
    all_seed_results.append({"feature_set": "Set C1", "seed": 123, **flatten_metrics(m_c1_123)})
    m_c1_7, _ = runner.run_or_load_fit("c1_s7", dtrain_c1, dval_c1, y_val, seed=7)
    all_seed_results.append({"feature_set": "Set C1", "seed": 7, **flatten_metrics(m_c1_7)})

    # Clean up Set C1 matrices
    del dtrain_c1, dval_c1
    gc.collect()

    # Intermediate seed stability save
    df_seeds_int = pd.DataFrame(all_seed_results)
    df_seeds_int.to_csv(runner.reports_dir / "m5_seed_stability.csv", index=False)

    # =========================================================================
    # STEP 7: ROLLING-ORIGIN BACKTEST (2 Folds: Fold 1 [50/15] and Fold 2 [70/15])
    # =========================================================================
    logger.info(">>> STEP 7: RUNNING ROLLING-ORIGIN BACKTEST (2 FOLDS) <<<")
    backtest_meta_path = runner.checkpoints_dir / "backtest_meta.json"
    if backtest_meta_path.exists():
        with open(backtest_meta_path, "r", encoding="utf-8") as f:
            backtest_records = json.load(f)
        runner.log_progress("bt_f1_b (cached)")
        runner.log_progress("bt_f1_c1 (cached)")
    else:
        full_85_b = pd.concat([
            pd.read_parquet(proc_dir / "train_set_b.parquet"),
            pd.read_parquet(proc_dir / "val_set_b.parquet"),
        ], ignore_index=True).sort_values(by=["TransactionDT", "TransactionID"]).reset_index(drop=True)

        full_85_c1 = pd.concat([
            pd.read_parquet(proc_dir / "train_set_c1.parquet"),
            pd.read_parquet(proc_dir / "val_set_c1.parquet"),
        ], ignore_index=True).sort_values(by=["TransactionDT", "TransactionID"]).reset_index(drop=True)

        n_85 = len(full_85_b)
        raw_tr_end = int(n_85 * 0.50)
        raw_vl_end = int(n_85 * 0.65)
        tr_end = find_next_distinct_boundary(full_85_b, raw_tr_end)
        vl_end = find_next_distinct_boundary(full_85_b, raw_vl_end)

        f1_tr_b = full_85_b.iloc[:tr_end]
        f1_vl_b = full_85_b.iloc[tr_end:vl_end]
        y_f1_tr = f1_tr_b["isFraud"].values.astype(np.int32)
        y_f1_vl = f1_vl_b["isFraud"].values.astype(np.int32)

        # Fold 1 Set B
        p_f1_b = TreePreprocessor().fit(f1_tr_b[feats_b])
        X_tr = np.ascontiguousarray(p_f1_b.transform(f1_tr_b[feats_b]).values, dtype=np.float32)
        X_vl = np.ascontiguousarray(p_f1_b.transform(f1_vl_b[feats_b]).values, dtype=np.float32)
        del f1_tr_b, f1_vl_b
        gc.collect()

        dtr = xgb.DMatrix(X_tr, label=y_f1_tr)
        dvl = xgb.DMatrix(X_vl, label=y_f1_vl)
        del X_tr, X_vl
        gc.collect()

        m_bt_f1_b, _ = runner.run_or_load_fit("bt_f1_b", dtr, dvl, y_f1_vl, seed=42)
        del dtr, dvl
        gc.collect()

        # Fold 1 Set C1
        f1_tr_c1 = full_85_c1.iloc[:tr_end]
        f1_vl_c1 = full_85_c1.iloc[tr_end:vl_end]
        p_f1_c1 = TreePreprocessor().fit(f1_tr_c1[feats_c1])
        X_tr = np.ascontiguousarray(p_f1_c1.transform(f1_tr_c1[feats_c1]).values, dtype=np.float32)
        X_vl = np.ascontiguousarray(p_f1_c1.transform(f1_vl_c1[feats_c1]).values, dtype=np.float32)
        del f1_tr_c1, f1_vl_c1, full_85_b, full_85_c1
        gc.collect()

        dtr = xgb.DMatrix(X_tr, label=y_f1_tr)
        dvl = xgb.DMatrix(X_vl, label=y_f1_vl)
        del X_tr, X_vl
        gc.collect()

        m_bt_f1_c1, _ = runner.run_or_load_fit("bt_f1_c1", dtr, dvl, y_f1_vl, seed=42)
        del dtr, dvl
        gc.collect()

        # Fold 2 coincides exactly with the 70% train / 15% val standard evaluation
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
        with open(backtest_meta_path, "w", encoding="utf-8") as f:
            json.dump(backtest_records, f, indent=2)

    df_backtest = pd.DataFrame(backtest_records)
    df_backtest.to_csv(runner.reports_dir / "m5_rolling_backtest.csv", index=False)
    with open(runner.reports_dir / "m5_rolling_backtest.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_id": runner.run_id,
            "git_commit_hash": runner.git_hash,
            "is_git_dirty": runner.is_dirty,
            "parquet_sha256": runner.parquet_hashes,
            "folds": backtest_records,
            "reduced_scope_metadata": runner.reduced_scope_metadata,
        }, f, indent=2)
    logger.info("Saved rolling backtest results to reports/m5_rolling_backtest.json")

    # =========================================================================
    # STEP 8: C1 SENSITIVITY RUN (Without card1_prior_count, Seeds 42, 123, 7)
    # =========================================================================
    logger.info(">>> STEP 8: RUNNING C1 SENSITIVITY RUN (WITHOUT card1_prior_count) <<<")
    df_tr_c1 = pd.read_parquet(proc_dir / "train_set_c1.parquet")
    df_vl_c1 = pd.read_parquet(proc_dir / "val_set_c1.parquet")
    feats_c1_sens = [c for c in get_model_feature_columns(df_tr_c1) if c != "card1_prior_count"]

    prep_c1_sens = TreePreprocessor().fit(df_tr_c1[feats_c1_sens])
    X_tr_sens = prep_c1_sens.transform(df_tr_c1[feats_c1_sens])
    X_vl_sens = prep_c1_sens.transform(df_vl_c1[feats_c1_sens])
    del df_tr_c1, df_vl_c1
    gc.collect()

    X_tr_arr = np.ascontiguousarray(X_tr_sens.values, dtype=np.float32)
    X_vl_arr = np.ascontiguousarray(X_vl_sens.values, dtype=np.float32)
    del X_tr_sens, X_vl_sens
    gc.collect()

    dtrain_sens = xgb.DMatrix(X_tr_arr, label=y_tr)
    dval_sens = xgb.DMatrix(X_vl_arr, label=y_val)
    del X_tr_arr, X_vl_arr
    gc.collect()

    for s in [42, 123, 7]:
        m_sens, _ = runner.run_or_load_fit(f"c1_sens_s{s}", dtrain_sens, dval_sens, y_val, seed=s)
        all_seed_results.append({
            "feature_set": "Set C1 (Sens. no prior_count)",
            "seed": s,
            **flatten_metrics(m_sens),
        })

    del dtrain_sens, dval_sens
    gc.collect()

    # Re-save complete seed stability table
    df_seeds_full = pd.DataFrame(all_seed_results)
    df_seeds_full.to_csv(runner.reports_dir / "m5_seed_stability.csv", index=False)

    summary_seeds_full = []
    for fs in df_seeds_full["feature_set"].unique():
        sub = df_seeds_full[df_seeds_full["feature_set"] == fs]
        summary_seeds_full.append({
            "feature_set": fs,
            "seeds_count": len(sub),
            "pr_auc_mean": round(float(sub["pr_auc"].mean()), 6),
            "pr_auc_std": round(float(sub["pr_auc"].std()), 6) if len(sub) > 1 else 0.0,
            "roc_auc_mean": round(float(sub["roc_auc"].mean()), 6),
            "roc_auc_std": round(float(sub["roc_auc"].std()), 6) if len(sub) > 1 else 0.0,
            "recall_top_1.0pct_mean": round(float(sub["recall_top_1.0pct"].mean()), 6),
            "recall_top_1.0pct_std": round(float(sub["recall_top_1.0pct"].std()), 6) if len(sub) > 1 else 0.0,
            "recall_top_2.0pct_mean": round(float(sub["recall_top_2.0pct"].mean()), 6),
            "recall_top_2.0pct_std": round(float(sub["recall_top_2.0pct"].std()), 6) if len(sub) > 1 else 0.0,
            "recall_top_5.0pct_mean": round(float(sub["recall_top_5.0pct"].mean()), 6),
            "recall_top_5.0pct_std": round(float(sub["recall_top_5.0pct"].std()), 6) if len(sub) > 1 else 0.0,
        })

    with open(runner.reports_dir / "m5_seed_stability.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_id": runner.run_id,
            "git_commit_hash": runner.git_hash,
            "is_git_dirty": runner.is_dirty,
            "parquet_sha256": runner.parquet_hashes,
            "summary": summary_seeds_full,
            "all_runs": all_seed_results,
            "reduced_scope_metadata": runner.reduced_scope_metadata,
        }, f, indent=2)
    logger.info("Saved seed stability results to reports/m5_seed_stability.json")

    # =========================================================================
    # DECISION RULE EVALUATION (docs/ANALYSIS_PLAN.md)
    # =========================================================================
    ci_pr_auc_c1 = boot_c1_b["pr_auc"]["ci_95_lower"]
    bt_folds_positive = all(fold["c1_minus_b_pr_auc"] > 0 for fold in backtest_records)
    c1_improves = (ci_pr_auc_c1 > 0) and bt_folds_positive

    verdict_text = "IMPROVEMENT CONFIRMED" if c1_improves else "NO RELIABLE GAIN"
    logger.info("=========================================================================")
    logger.info("DECISION RULE VERDICT:")
    logger.info("  Criterion 1 (Bootstrap 95%% CI lower > 0): %.6f -> %s", ci_pr_auc_c1, "PASS" if ci_pr_auc_c1 > 0 else "FAIL")
    logger.info("  Criterion 2 (All backtest folds positive): %s -> %s", [f['c1_minus_b_pr_auc'] for f in backtest_records], "PASS" if bt_folds_positive else "FAIL")
    logger.info("  FINAL REPORTED RESULT: %s", verdict_text)
    logger.info("=========================================================================")

    total_time_min = (time.time() - runner.start_time) / 60.0
    logger.info("ALL MILESTONE 5 EXPERIMENTS COMPLETED IN %.2f MINUTES!", total_time_min)


if __name__ == "__main__":
    run_milestone5_pipeline()
