"""Baseline training and validation evaluation engine for Milestone 3.

Evaluates:
- Logistic Regression (class_weight=None vs 'balanced') on Set A & B
- Isolation Forest (unsupervised on train features, score = -score_samples) on Set A & B
- XGBoost (scale_pos_weight=None vs train ratio) on Set A & B

All evaluations are conducted STRICTLY on the VALIDATION split.
The test split is never loaded or accessed.
Results are saved to reports/baselines_val.csv and reports/baselines_val.json.
"""

import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
import xgboost as xgb

from fraudshield.splits.guard import load_split_parquet
from fraudshield.features.pipeline import (
    LinearPreprocessor,
    TreePreprocessor,
    get_model_feature_columns,
)
from fraudshield.evaluation.metrics import evaluate_model_performance

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudshield.models.train_baselines")


def get_git_commit_hash() -> str:
    """Get the current short git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def train_and_eval_logistic_regression(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_val_raw: pd.DataFrame,
    y_val: np.ndarray,
    class_weight: Any = None,
    random_state: int = 42,
    max_iter: int = 300,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Train Logistic Regression with train-fitted LinearPreprocessor and evaluate on validation."""
    logger.info("Training Logistic Regression (class_weight=%s)...", class_weight)
    prep = LinearPreprocessor().fit(X_train_raw)
    X_tr_proc = prep.transform(X_train_raw)
    X_vl_proc = prep.transform(X_val_raw)

    lr = LogisticRegression(
        C=1.0,
        class_weight=class_weight,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
    )
    lr.fit(X_tr_proc, y_train)
    val_probs = lr.predict_proba(X_vl_proc)[:, 1]

    metrics = evaluate_model_performance(y_val, val_probs)
    return metrics, val_probs


def train_and_eval_isolation_forest(
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    y_val: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Train unsupervised Isolation Forest on train features only, score = -score_samples (higher = riskier)."""
    logger.info("Training Isolation Forest (unsupervised)...")
    prep = LinearPreprocessor().fit(X_train_raw)
    X_tr_proc = prep.transform(X_train_raw)
    X_vl_proc = prep.transform(X_val_raw)

    # Subsample 20,000 points for tree construction to maintain high speed and low memory
    max_samples = min(20000, len(X_tr_proc))
    iforest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=0.035,
        random_state=random_state,
        n_jobs=-1,
    )
    iforest.fit(X_tr_proc)

    # IsolationForest.score_samples returns lower values for more anomalous instances
    # We negate it so higher values indicate greater anomaly risk
    raw_scores = iforest.score_samples(X_vl_proc)
    val_anomaly_scores = -1.0 * raw_scores

    metrics = evaluate_model_performance(y_val, val_anomaly_scores)
    return metrics, val_anomaly_scores


def train_and_eval_xgboost(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_val_raw: pd.DataFrame,
    y_val: np.ndarray,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Train XGBoost with early stopping on validation and evaluate."""
    spw_str = f"{scale_pos_weight:.2f}" if scale_pos_weight else "None"
    logger.info("Training XGBoost (scale_pos_weight=%s)...", spw_str)
    prep = TreePreprocessor().fit(X_train_raw)
    X_tr_proc = prep.transform(X_train_raw)
    X_vl_proc = prep.transform(X_val_raw)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_tr_proc,
        y_train,
        eval_set=[(X_vl_proc, y_val)],
        verbose=False,
    )

    val_probs = model.predict_proba(X_vl_proc)[:, 1]
    metrics = evaluate_model_performance(y_val, val_probs)
    metrics["best_iteration"] = int(model.best_iteration)
    return metrics, val_probs


def run_validation_baselines(config_path: Path = Path("configs/config.yaml")) -> pd.DataFrame:
    """Execute validation benchmark across all specified baselines on Sets A and B."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    run_id = str(uuid.uuid4())[:8]
    git_hash = get_git_commit_hash()

    logger.info("Starting Baseline Evaluation on VALIDATION split (Run ID: %s, Git: %s)...", run_id, git_hash)

    # 1. Load Train and Validation for Set A
    df_train_a = load_split_parquet("train", feature_set="a", allow_test_access=False)
    df_val_a = load_split_parquet("val", feature_set="a", allow_test_access=False)

    feature_cols_a = get_model_feature_columns(df_train_a)
    y_train = df_train_a["isFraud"].values.astype(int)
    y_val = df_val_a["isFraud"].values.astype(int)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    train_spw = float(n_neg / n_pos)
    logger.info("Train class ratio: %d non-fraud, %d fraud -> scale_pos_weight = %.2f", n_neg, n_pos, train_spw)

    results = []

    # ==========================
    # SET A BASELINES
    # ==========================
    logger.info("--- EVALUATING SET A BASELINES ---")

    # 1. Logistic Regression (no class_weight)
    m_lr_none, _ = train_and_eval_logistic_regression(
        df_train_a[feature_cols_a], y_train, df_val_a[feature_cols_a], y_val, class_weight=None, random_state=seed
    )
    results.append({
        "run_id": run_id, "git_hash": git_hash, "feature_set": "Set A",
        "model_name": "Logistic Regression", "variant": "class_weight=None",
        "pr_auc": m_lr_none["pr_auc"], "roc_auc": m_lr_none["roc_auc"], "brier_score": m_lr_none["brier_score"],
        "recall_top_0.5pct": m_lr_none["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": m_lr_none["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": m_lr_none["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": m_lr_none["budgets"]["top_5.0pct"]["recall"],
        "fpr_top_1.0pct": m_lr_none["budgets"]["top_1.0pct"]["fpr"],
        "precision_top_1.0pct": m_lr_none["budgets"]["top_1.0pct"]["precision"],
    })

    # 2. Logistic Regression (balanced)
    m_lr_bal, _ = train_and_eval_logistic_regression(
        df_train_a[feature_cols_a], y_train, df_val_a[feature_cols_a], y_val, class_weight="balanced", random_state=seed
    )
    results.append({
        "run_id": run_id, "git_hash": git_hash, "feature_set": "Set A",
        "model_name": "Logistic Regression", "variant": "class_weight=balanced",
        "pr_auc": m_lr_bal["pr_auc"], "roc_auc": m_lr_bal["roc_auc"], "brier_score": m_lr_bal["brier_score"],
        "recall_top_0.5pct": m_lr_bal["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": m_lr_bal["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": m_lr_bal["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": m_lr_bal["budgets"]["top_5.0pct"]["recall"],
        "fpr_top_1.0pct": m_lr_bal["budgets"]["top_1.0pct"]["fpr"],
        "precision_top_1.0pct": m_lr_bal["budgets"]["top_1.0pct"]["precision"],
    })

    # 3. Isolation Forest (unsupervised)
    m_if, _ = train_and_eval_isolation_forest(
        df_train_a[feature_cols_a], df_val_a[feature_cols_a], y_val, random_state=seed
    )
    results.append({
        "run_id": run_id, "git_hash": git_hash, "feature_set": "Set A",
        "model_name": "Isolation Forest", "variant": "unsupervised",
        "pr_auc": m_if["pr_auc"], "roc_auc": m_if["roc_auc"], "brier_score": m_if["brier_score"],
        "recall_top_0.5pct": m_if["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": m_if["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": m_if["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": m_if["budgets"]["top_5.0pct"]["recall"],
        "fpr_top_1.0pct": m_if["budgets"]["top_1.0pct"]["fpr"],
        "precision_top_1.0pct": m_if["budgets"]["top_1.0pct"]["precision"],
    })

    # 4. XGBoost (scale_pos_weight=None)
    m_xgb_none, _ = train_and_eval_xgboost(
        df_train_a[feature_cols_a], y_train, df_val_a[feature_cols_a], y_val, scale_pos_weight=None, random_state=seed
    )
    results.append({
        "run_id": run_id, "git_hash": git_hash, "feature_set": "Set A",
        "model_name": "XGBoost", "variant": "scale_pos_weight=None",
        "pr_auc": m_xgb_none["pr_auc"], "roc_auc": m_xgb_none["roc_auc"], "brier_score": m_xgb_none["brier_score"],
        "recall_top_0.5pct": m_xgb_none["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": m_xgb_none["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": m_xgb_none["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": m_xgb_none["budgets"]["top_5.0pct"]["recall"],
        "fpr_top_1.0pct": m_xgb_none["budgets"]["top_1.0pct"]["fpr"],
        "precision_top_1.0pct": m_xgb_none["budgets"]["top_1.0pct"]["precision"],
    })

    # 5. XGBoost (scale_pos_weight=train_spw)
    m_xgb_spw, _ = train_and_eval_xgboost(
        df_train_a[feature_cols_a], y_train, df_val_a[feature_cols_a], y_val, scale_pos_weight=train_spw, random_state=seed
    )
    results.append({
        "run_id": run_id, "git_hash": git_hash, "feature_set": "Set A",
        "model_name": "XGBoost", "variant": f"scale_pos_weight={train_spw:.2f}",
        "pr_auc": m_xgb_spw["pr_auc"], "roc_auc": m_xgb_spw["roc_auc"], "brier_score": m_xgb_spw["brier_score"],
        "recall_top_0.5pct": m_xgb_spw["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": m_xgb_spw["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": m_xgb_spw["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": m_xgb_spw["budgets"]["top_5.0pct"]["recall"],
        "fpr_top_1.0pct": m_xgb_spw["budgets"]["top_1.0pct"]["fpr"],
        "precision_top_1.0pct": m_xgb_spw["budgets"]["top_1.0pct"]["precision"],
    })

    # Clean up Set A data from memory before loading Set B
    del df_train_a, df_val_a

    # ==========================
    # SET B BASELINES
    # ==========================
    logger.info("--- EVALUATING SET B BASELINES ---")
    df_train_b = load_split_parquet("train", feature_set="b", allow_test_access=False)
    df_val_b = load_split_parquet("val", feature_set="b", allow_test_access=False)
    feature_cols_b = get_model_feature_columns(df_train_b)

    # 6. Logistic Regression on Set B (no class_weight)
    m_lr_b_none, _ = train_and_eval_logistic_regression(
        df_train_b[feature_cols_b], y_train, df_val_b[feature_cols_b], y_val, class_weight=None, random_state=seed
    )
    results.append({
        "run_id": run_id, "git_hash": git_hash, "feature_set": "Set B",
        "model_name": "Logistic Regression", "variant": "class_weight=None",
        "pr_auc": m_lr_b_none["pr_auc"], "roc_auc": m_lr_b_none["roc_auc"], "brier_score": m_lr_b_none["brier_score"],
        "recall_top_0.5pct": m_lr_b_none["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": m_lr_b_none["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": m_lr_b_none["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": m_lr_b_none["budgets"]["top_5.0pct"]["recall"],
        "fpr_top_1.0pct": m_lr_b_none["budgets"]["top_1.0pct"]["fpr"],
        "precision_top_1.0pct": m_lr_b_none["budgets"]["top_1.0pct"]["precision"],
    })

    # 7. Logistic Regression on Set B (balanced)
    m_lr_b_bal, _ = train_and_eval_logistic_regression(
        df_train_b[feature_cols_b], y_train, df_val_b[feature_cols_b], y_val, class_weight="balanced", random_state=seed
    )
    results.append({
        "run_id": run_id, "git_hash": git_hash, "feature_set": "Set B",
        "model_name": "Logistic Regression", "variant": "class_weight=balanced",
        "pr_auc": m_lr_b_bal["pr_auc"], "roc_auc": m_lr_b_bal["roc_auc"], "brier_score": m_lr_b_bal["brier_score"],
        "recall_top_0.5pct": m_lr_b_bal["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": m_lr_b_bal["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": m_lr_b_bal["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": m_lr_b_bal["budgets"]["top_5.0pct"]["recall"],
        "fpr_top_1.0pct": m_lr_b_bal["budgets"]["top_1.0pct"]["fpr"],
        "precision_top_1.0pct": m_lr_b_bal["budgets"]["top_1.0pct"]["precision"],
    })

    # 8. Isolation Forest on Set B (unsupervised)
    m_if_b, _ = train_and_eval_isolation_forest(
        df_train_b[feature_cols_b], df_val_b[feature_cols_b], y_val, random_state=seed
    )
    results.append({
        "run_id": run_id, "git_hash": git_hash, "feature_set": "Set B",
        "model_name": "Isolation Forest", "variant": "unsupervised",
        "pr_auc": m_if_b["pr_auc"], "roc_auc": m_if_b["roc_auc"], "brier_score": m_if_b["brier_score"],
        "recall_top_0.5pct": m_if_b["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": m_if_b["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": m_if_b["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": m_if_b["budgets"]["top_5.0pct"]["recall"],
        "fpr_top_1.0pct": m_if_b["budgets"]["top_1.0pct"]["fpr"],
        "precision_top_1.0pct": m_if_b["budgets"]["top_1.0pct"]["precision"],
    })

    # 9. XGBoost on Set B (scale_pos_weight=None)
    m_xgb_b_none, _ = train_and_eval_xgboost(
        df_train_b[feature_cols_b], y_train, df_val_b[feature_cols_b], y_val, scale_pos_weight=None, random_state=seed
    )
    results.append({
        "run_id": run_id, "git_hash": git_hash, "feature_set": "Set B",
        "model_name": "XGBoost", "variant": "scale_pos_weight=None",
        "pr_auc": m_xgb_b_none["pr_auc"], "roc_auc": m_xgb_b_none["roc_auc"], "brier_score": m_xgb_b_none["brier_score"],
        "recall_top_0.5pct": m_xgb_b_none["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": m_xgb_b_none["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": m_xgb_b_none["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": m_xgb_b_none["budgets"]["top_5.0pct"]["recall"],
        "fpr_top_1.0pct": m_xgb_b_none["budgets"]["top_1.0pct"]["fpr"],
        "precision_top_1.0pct": m_xgb_b_none["budgets"]["top_1.0pct"]["precision"],
    })

    # 10. XGBoost on Set B (scale_pos_weight=train_spw)
    m_xgb_b_spw, _ = train_and_eval_xgboost(
        df_train_b[feature_cols_b], y_train, df_val_b[feature_cols_b], y_val, scale_pos_weight=train_spw, random_state=seed
    )
    results.append({
        "run_id": run_id, "git_hash": git_hash, "feature_set": "Set B",
        "model_name": "XGBoost", "variant": f"scale_pos_weight={train_spw:.2f}",
        "pr_auc": m_xgb_b_spw["pr_auc"], "roc_auc": m_xgb_b_spw["roc_auc"], "brier_score": m_xgb_b_spw["brier_score"],
        "recall_top_0.5pct": m_xgb_b_spw["budgets"]["top_0.5pct"]["recall"],
        "recall_top_1.0pct": m_xgb_b_spw["budgets"]["top_1.0pct"]["recall"],
        "recall_top_2.0pct": m_xgb_b_spw["budgets"]["top_2.0pct"]["recall"],
        "recall_top_5.0pct": m_xgb_b_spw["budgets"]["top_5.0pct"]["recall"],
        "fpr_top_1.0pct": m_xgb_b_spw["budgets"]["top_1.0pct"]["fpr"],
        "precision_top_1.0pct": m_xgb_b_spw["budgets"]["top_1.0pct"]["precision"],
    })

    df_results = pd.DataFrame(results)

    # Save to reports
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "baselines_val.csv"
    json_path = reports_dir / "baselines_val.json"

    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "git_commit_hash": git_hash,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_snapshot": cfg,
            "results": results,
        }, f, indent=2)

    logger.info("Saved baseline validation results to %s and %s", csv_path, json_path)
    return df_results
