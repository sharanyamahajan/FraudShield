"""Comprehensive evaluation metrics for transaction fraud risk scoring.

Metrics include:
- PR-AUC (Average Precision) - Primary metric
- ROC-AUC - Secondary metric
- Operational Review Budget Metrics: Precision, Recall (fraud capture rate),
  False Positive Rate (FPR), and review count at top 0.5%, 1%, 2%, 5%
- Deterministic tie-breaking for score ranking
- Threshold sweeps (0.05 to 0.95)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

logger = logging.getLogger("fraudshield.evaluation.metrics")


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Precision-Recall Area Under the Curve (Average Precision)."""
    y_t = np.asarray(y_true).astype(int)
    y_s = np.asarray(y_score).astype(float)
    if len(np.unique(y_t)) < 2:
        return 0.0
    return float(average_precision_score(y_t, y_s))


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Receiver Operating Characteristic Area Under the Curve."""
    y_t = np.asarray(y_true).astype(int)
    y_s = np.asarray(y_score).astype(float)
    if len(np.unique(y_t)) < 2:
        return 0.5
    return float(roc_auc_score(y_t, y_s))


def compute_budget_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    budgets: List[float] = [0.005, 0.01, 0.02, 0.05],
    tie_breaker: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute operational metrics at fixed manual review capacity budgets (e.g. top 0.5%, 1%, 2%, 5%).
    
    Ranking rules:
    - Transactions are ordered by risk score descending.
    - Ties in risk score are broken deterministically using tie_breaker (or array index).
    - Review capacity at budget b is: n_budget = max(1, int(round(n_total * b))).
    """
    y_t = np.asarray(y_true).astype(int)
    y_s = np.asarray(y_score).astype(float)
    n_total = len(y_t)
    total_positives = int(np.sum(y_t))
    total_negatives = n_total - total_positives

    if tie_breaker is None:
        tie_breaker = np.arange(n_total)
    else:
        tie_breaker = np.asarray(tie_breaker)

    # Sort descending by score; break ties ascending by tie_breaker
    # Lexsort sorts primarily by the LAST key in the tuple
    order = np.lexsort((tie_breaker, -y_s))
    y_t_sorted = y_t[order]
    y_s_sorted = y_s[order]

    results = {}
    for b in budgets:
        # Documented rounding: at least 1 transaction reviewed if budget > 0
        n_budget = max(1, int(round(n_total * b)))
        n_budget = min(n_budget, n_total)

        reviewed_labels = y_t_sorted[:n_budget]
        tp = int(np.sum(reviewed_labels))
        fp = n_budget - tp
        fn = total_positives - tp
        tn = total_negatives - fp

        recall = float(tp / total_positives) if total_positives > 0 else 0.0
        precision = float(tp / n_budget) if n_budget > 0 else 0.0
        fpr = float(fp / total_negatives) if total_negatives > 0 else 0.0

        budget_key = f"top_{round(b * 100, 2)}pct"
        results[budget_key] = {
            "budget_ratio": float(b),
            "review_volume": int(n_budget),
            "fraud_captured": tp,
            "false_alerts": fp,
            "recall": round(recall, 6),
            "precision": round(precision, 6),
            "fpr": round(fpr, 6),
            "score_threshold_at_budget": float(y_s_sorted[n_budget - 1]),
        }

    return results


def evaluate_model_performance(
    y_true: np.ndarray,
    y_score: np.ndarray,
    budgets: List[float] = [0.005, 0.01, 0.02, 0.05],
    tie_breaker: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate overall model evaluation summary: PR-AUC, ROC-AUC, Brier score, and budget metrics."""
    pr_auc = compute_pr_auc(y_true, y_score)
    roc_auc = compute_roc_auc(y_true, y_score)
    brier = float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0)))
    budget_stats = compute_budget_metrics(y_true, y_score, budgets=budgets, tie_breaker=tie_breaker)

    return {
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "brier_score": round(brier, 6),
        "budgets": budget_stats,
    }
