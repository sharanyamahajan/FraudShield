"""Rerun all Logistic Regression fits (Sets A, B, C1, C2; class_weight=None)
with max_iter=2000 to test convergence.
"""

import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraudshield.splits.guard import load_split_parquet
from fraudshield.features.pipeline import LinearPreprocessor, get_model_feature_columns
from fraudshield.evaluation.metrics import evaluate_model_performance

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudshield.lr_2000")


def eval_lr_set(feature_set: str, max_iter: int = 2000):
    logger.info("--- Rerunning LR on Set %s (max_iter=%d) ---", feature_set.upper(), max_iter)
    df_train = load_split_parquet("train", feature_set=feature_set)
    df_val = load_split_parquet("val", feature_set=feature_set)

    feat_cols = get_model_feature_columns(df_train)
    y_train = df_train["isFraud"].values.astype(int)
    y_val = df_val["isFraud"].values.astype(int)

    prep = LinearPreprocessor().fit(df_train[feat_cols])
    X_tr_proc = prep.transform(df_train[feat_cols])
    X_vl_proc = prep.transform(df_val[feat_cols])

    t0 = time.time()
    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_tr_proc, y_train)
    t1 = time.time()

    val_probs = lr.predict_proba(X_vl_proc)[:, 1]
    metrics = evaluate_model_performance(y_val, val_probs)

    n_iter = int(lr.n_iter_[0])
    converged = bool(n_iter < max_iter)

    logger.info(
        "Set %s finished in %.1fs: n_iter=%d, converged=%s, PR-AUC=%.6f, ROC-AUC=%.6f",
        feature_set.upper(), t1 - t0, n_iter, converged, metrics["pr_auc"], metrics["roc_auc"]
    )

    return {
        "feature_set": f"Set {feature_set.upper()}",
        "model_name": "Logistic Regression",
        "variant": "class_weight=None",
        "max_iter": max_iter,
        "n_iter": n_iter,
        "converged": converged,
        "fit_time_seconds": round(t1 - t0, 1),
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "brier_score": metrics["brier_score"],
        "recall_top_0.5pct": metrics["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": metrics["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": metrics["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": metrics["budgets"]["top_5.0pct"]["recall"],
        "precision_top_1.0pct": metrics["budgets"]["top_1.0pct"]["precision"],
        "fpr_top_1.0pct": metrics["budgets"]["top_1.0pct"]["fpr"],
    }


def main():
    results = []
    for s in ["a", "b", "c1", "c2"]:
        res = eval_lr_set(s, max_iter=2000)
        results.append(res)

    df_res = pd.DataFrame(results)
    out_csv = Path("reports/lr_convergence_2000.csv")
    out_json = Path("reports/lr_convergence_2000.json")
    df_res.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 105)
    print("LOGISTIC REGRESSION CONVERGENCE BENCHMARK (max_iter=2000, solver=lbfgs, class_weight=None)")
    print("=" * 105)
    cols = [
        "feature_set", "n_iter", "converged", "fit_time_seconds",
        "pr_auc", "roc_auc", "recall_top_1.0pct", "precision_top_1.0pct"
    ]
    print(df_res[cols].to_string(index=False))
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
