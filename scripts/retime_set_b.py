"""Retime ONE Set B XGBoost fit with M3/M4 hyperparameters.

Reports: max_depth, rounds, wall time, peak RAM.
Uses contiguous float32 DMatrix, tree_method="hist", n_jobs=6.
Saves checkpoint to reports/m5_checkpoints/b_s42_meta.json and b_s42_probs.npy.
"""

import gc
import json
import os
import time
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from fraudshield.features.pipeline import TreePreprocessor, get_model_feature_columns
from fraudshield.evaluation.metrics import evaluate_model_performance


def retime_set_b():
    tracemalloc.start()
    t_start = time.time()

    reports_dir = Path("reports")
    ckpt_dir = reports_dir / "m5_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    proc_dir = Path("data/processed")
    print("Loading train_set_b and val_set_b parquets...")
    df_tr = pd.read_parquet(proc_dir / "train_set_b.parquet")
    df_vl = pd.read_parquet(proc_dir / "val_set_b.parquet")

    feats = get_model_feature_columns(df_tr)
    y_tr = df_tr["isFraud"].values.astype(np.int32)
    y_vl = df_vl["isFraud"].values.astype(np.int32)

    print(f"Fitting TreePreprocessor on {len(feats)} features...")
    prep = TreePreprocessor().fit(df_tr[feats])
    X_tr_df = prep.transform(df_tr[feats])
    X_vl_df = prep.transform(df_vl[feats])

    del df_tr, df_vl
    gc.collect()

    print("Converting to contiguous float32 DMatrix...")
    X_tr_arr = np.ascontiguousarray(X_tr_df.values, dtype=np.float32)
    X_vl_arr = np.ascontiguousarray(X_vl_df.values, dtype=np.float32)
    del X_tr_df, X_vl_df
    gc.collect()

    dtrain = xgb.DMatrix(X_tr_arr, label=y_tr)
    dval = xgb.DMatrix(X_vl_arr, label=y_vl)
    del X_tr_arr, X_vl_arr
    gc.collect()

    max_depth = 6
    num_boost_round = 300
    early_stopping_rounds = 30
    params = {
        "max_depth": max_depth,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1.0,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": 6,
    }

    print(f"Starting XGBoost fit (max_depth={max_depth}, max_rounds={num_boost_round}, n_jobs=6)...")
    t0_fit = time.time()
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    t_fit_wall = time.time() - t0_fit
    t_total_wall = time.time() - t_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_mem / (1024 * 1024)

    probs = bst.predict(dval)
    metrics = evaluate_model_performance(y_vl, probs)
    metrics["best_iteration"] = int(bst.best_iteration)
    metrics["num_boost_round"] = num_boost_round
    metrics["max_depth"] = max_depth
    metrics["fit_time_seconds"] = round(t_fit_wall, 2)
    metrics["total_time_seconds"] = round(t_total_wall, 2)
    metrics["peak_ram_mb"] = round(peak_ram_mb, 2)
    metrics["seed"] = 42
    metrics["fit_key"] = "b_s42"

    # Save checkpoint immediately
    meta_path = ckpt_dir / "b_s42_meta.json"
    probs_path = ckpt_dir / "b_s42_probs.npy"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(probs_path, probs)

    print("\n" + "=" * 60)
    print("RETIME BENCHMARK RESULTS (SET B XGBOOST SEED 42):")
    print(f"  - max_depth:        {max_depth}")
    print(f"  - max_rounds:       {num_boost_round}")
    print(f"  - best_iteration:   {bst.best_iteration}")
    print(f"  - fit wall time:    {t_fit_wall:.2f} s ({t_fit_wall / 60.0:.2f} min)")
    print(f"  - total wall time:  {t_total_wall:.2f} s ({t_total_wall / 60.0:.2f} min)")
    print(f"  - peak RAM:         {peak_ram_mb:.2f} MB")
    print(f"  - PR-AUC:           {metrics['pr_auc']:.6f}")
    print(f"  - ROC-AUC:          {metrics['roc_auc']:.6f}")
    print(f"  - Checkpoint saved: {meta_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    retime_set_b()
