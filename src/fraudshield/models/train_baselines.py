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
from typing import Dict, Any, List, Tuple, Optional
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


def is_git_dirty() -> bool:
    """Check if the git working tree has unstaged or uncommitted changes."""
    try:
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if res.returncode == 0:
            return bool(res.stdout.strip())
    except Exception:
        pass
    return False


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
    # TreePreprocessor converts categoricals to ordinal codes for tree construction
    prep = TreePreprocessor().fit(X_train_raw)
    X_tr_proc = prep.transform(X_train_raw)
    X_vl_proc = prep.transform(X_val_raw)

    # Impute remaining NaNs with train median for Isolation Forest
    fill_vals = X_tr_proc.median()
    X_tr_filled = X_tr_proc.fillna(fill_vals)
    X_vl_filled = X_vl_proc.fillna(fill_vals)

    max_samples = min(20000, len(X_tr_filled))
    iforest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=0.035,
        random_state=random_state,
        n_jobs=-1,
    )
    iforest.fit(X_tr_filled)

    # IsolationForest.score_samples returns lower values for more anomalous instances; negate so higher = riskier
    raw_scores = iforest.score_samples(X_vl_filled)
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
    dirty_flag = is_git_dirty()

    logger.info("Starting Baseline Evaluation on VALIDATION split (Run ID: %s, Git: %s, Dirty: %s)...", run_id, git_hash, dirty_flag)

    # 1. Load Train and Validation for Set A
    df_train_a = load_split_parquet("train", feature_set="a")
    df_val_a = load_split_parquet("val", feature_set="a")

    feature_cols_a = get_model_feature_columns(df_train_a)
    y_train = df_train_a["isFraud"].values.astype(int)
    y_val = df_val_a["isFraud"].values.astype(int)

    n_val_total = len(y_val)
    n_val_pos = int(np.sum(y_val == 1))
    max_rec_05 = round(min(max(1, int(round(n_val_total * 0.005))), n_val_pos) / n_val_pos, 6)
    max_rec_10 = round(min(max(1, int(round(n_val_total * 0.010))), n_val_pos) / n_val_pos, 6)
    max_rec_20 = round(min(max(1, int(round(n_val_total * 0.020))), n_val_pos) / n_val_pos, 6)
    max_rec_50 = round(min(max(1, int(round(n_val_total * 0.050))), n_val_pos) / n_val_pos, 6)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    train_spw = float(n_neg / n_pos)
    logger.info("Train class ratio: %d non-fraud, %d fraud -> scale_pos_weight = %.2f", n_neg, n_pos, train_spw)

    results = []

    def build_row(feature_set: str, model_name: str, variant: str, m: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "git_hash": git_hash,
            "is_git_dirty": dirty_flag,
            "feature_set": feature_set,
            "model_name": model_name,
            "variant": variant,
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "brier_score": m["brier_score"],
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

    # ==========================
    # SET A BASELINES
    # ==========================
    logger.info("--- EVALUATING SET A BASELINES ---")

    # 1. Logistic Regression (class_weight=None)
    m_lr_none, _ = train_and_eval_logistic_regression(
        df_train_a[feature_cols_a], y_train, df_val_a[feature_cols_a], y_val, class_weight=None, random_state=seed
    )
    results.append(build_row("Set A", "Logistic Regression", "class_weight=None", m_lr_none))

    # 2. Logistic Regression (balanced)
    m_lr_bal, _ = train_and_eval_logistic_regression(
        df_train_a[feature_cols_a], y_train, df_val_a[feature_cols_a], y_val, class_weight="balanced", random_state=seed
    )
    results.append(build_row("Set A", "Logistic Regression", "class_weight=balanced", m_lr_bal))

    # 3. Isolation Forest (unsupervised)
    m_if, _ = train_and_eval_isolation_forest(
        df_train_a[feature_cols_a], df_val_a[feature_cols_a], y_val, random_state=seed
    )
    results.append(build_row("Set A", "Isolation Forest", "unsupervised", m_if))

    # 4. XGBoost (scale_pos_weight=None)
    m_xgb_none, _ = train_and_eval_xgboost(
        df_train_a[feature_cols_a], y_train, df_val_a[feature_cols_a], y_val, scale_pos_weight=None, random_state=seed
    )
    results.append(build_row("Set A", "XGBoost", "scale_pos_weight=None", m_xgb_none))

    # 5. XGBoost (scale_pos_weight=train_spw)
    m_xgb_spw, _ = train_and_eval_xgboost(
        df_train_a[feature_cols_a], y_train, df_val_a[feature_cols_a], y_val, scale_pos_weight=train_spw, random_state=seed
    )
    results.append(build_row("Set A", "XGBoost", f"scale_pos_weight={train_spw:.2f}", m_xgb_spw))

    del df_train_a, df_val_a

    # ==========================
    # SET B BASELINES
    # ==========================
    logger.info("--- EVALUATING SET B BASELINES ---")
    df_train_b = load_split_parquet("train", feature_set="b")
    df_val_b = load_split_parquet("val", feature_set="b")
    feature_cols_b = get_model_feature_columns(df_train_b)

    # 6. Logistic Regression on Set B (class_weight=None)
    m_lr_b_none, _ = train_and_eval_logistic_regression(
        df_train_b[feature_cols_b], y_train, df_val_b[feature_cols_b], y_val, class_weight=None, random_state=seed
    )
    results.append(build_row("Set B", "Logistic Regression", "class_weight=None", m_lr_b_none))

    # 7. Logistic Regression on Set B (balanced)
    m_lr_b_bal, _ = train_and_eval_logistic_regression(
        df_train_b[feature_cols_b], y_train, df_val_b[feature_cols_b], y_val, class_weight="balanced", random_state=seed
    )
    results.append(build_row("Set B", "Logistic Regression", "class_weight=balanced", m_lr_b_bal))

    # 8. Isolation Forest on Set B (unsupervised)
    m_if_b, _ = train_and_eval_isolation_forest(
        df_train_b[feature_cols_b], df_val_b[feature_cols_b], y_val, random_state=seed
    )
    results.append(build_row("Set B", "Isolation Forest", "unsupervised", m_if_b))

    # 9. XGBoost on Set B (scale_pos_weight=None)
    m_xgb_b_none, _ = train_and_eval_xgboost(
        df_train_b[feature_cols_b], y_train, df_val_b[feature_cols_b], y_val, scale_pos_weight=None, random_state=seed
    )
    results.append(build_row("Set B", "XGBoost", "scale_pos_weight=None", m_xgb_b_none))

    # 10. XGBoost on Set B (scale_pos_weight=train_spw)
    m_xgb_b_spw, _ = train_and_eval_xgboost(
        df_train_b[feature_cols_b], y_train, df_val_b[feature_cols_b], y_val, scale_pos_weight=train_spw, random_state=seed
    )
    results.append(build_row("Set B", "XGBoost", f"scale_pos_weight={train_spw:.2f}", m_xgb_b_spw))

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
            "is_git_dirty": dirty_flag,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_snapshot": cfg,
            "results": results,
        }, f, indent=2)

    logger.info("Saved baseline validation results to %s and %s", csv_path, json_path)
    return df_results
