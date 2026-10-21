"""Unit tests for evaluation metrics with hand-built arrays and known values."""

import pytest
import numpy as np
from fraudshield.evaluation.metrics import (
    compute_pr_auc,
    compute_roc_auc,
    compute_budget_metrics,
    evaluate_model_performance,
)


def test_pr_auc_and_roc_auc_known_values():
    """Verify PR-AUC and ROC-AUC on small hand-crafted binary predictions with known outcomes."""
    # Perfect ranking: positive items have strictly higher scores than negative items
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_roc_auc(y_true, y_score) == 1.0
    assert compute_pr_auc(y_true, y_score) == 1.0

    # Inverted ranking
    y_score_inv = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_roc_auc(y_true, y_score_inv) == 0.0
    # Average precision of inverted ranking: positives ranked 3rd (P=1/3, R=1/2) and 4th (P=2/4=0.5, R=2/2=1)
    # AP = (1/3)*(1/2 - 0) + (1/2)*(1 - 1/2) = 1/6 + 1/4 = 5/12 ~ 0.416667
    ap = compute_pr_auc(y_true, y_score_inv)
    assert pytest.approx(ap, 1e-4) == 5.0 / 12.0


def test_review_budget_metrics_hand_built():
    """Verify review budget metrics on an exact 100-sample array with known fraud locations."""
    # 100 samples total, 5 frauds (5% base rate)
    # Frauds are at indices 0, 1, 2, 5, 10
    y_true = np.zeros(100, dtype=int)
    y_true[[0, 1, 2, 5, 10]] = 1

    # Scores strictly descending by index: 0 is highest risk, 99 is lowest risk
    y_score = 100 - np.arange(100)

    # Review budgets: top 1% (1 item), top 2% (2 items), top 5% (5 items)
    budgets = [0.01, 0.02, 0.05]
    res = compute_budget_metrics(y_true, y_score, budgets=budgets)

    # Top 1%: 1 sample reviewed (index 0). It is a fraud (y_true[0]=1).
    # TP = 1, FP = 0. Recall = 1/5 = 0.20, Precision = 1/1 = 1.0, FPR = 0 / 95 = 0.0
    top1 = res["top_1.0pct"]
    assert top1["review_volume"] == 1
    assert top1["fraud_captured"] == 1
    assert top1["false_alerts"] == 0
    assert pytest.approx(top1["recall"], 1e-5) == 0.20
    assert pytest.approx(top1["precision"], 1e-5) == 1.0
    assert pytest.approx(top1["fpr"], 1e-5) == 0.0

    # Top 2%: 2 samples reviewed (indices 0, 1). Both are frauds.
    # TP = 2, FP = 0. Recall = 2/5 = 0.40, Precision = 2/2 = 1.0, FPR = 0.0
    top2 = res["top_2.0pct"]
    assert top2["review_volume"] == 2
    assert top2["fraud_captured"] == 2
    assert pytest.approx(top2["recall"], 1e-5) == 0.40
    assert pytest.approx(top2["precision"], 1e-5) == 1.0

    # Top 5%: 5 samples reviewed (indices 0, 1, 2, 3, 4).
    # Frauds captured: 0, 1, 2 (3 frauds). Indices 3 and 4 are legitimate (FP = 2).
    # TP = 3, FP = 2. Recall = 3/5 = 0.60, Precision = 3/5 = 0.60, FPR = 2 / 95
    top5 = res["top_5.0pct"]
    assert top5["review_volume"] == 5
    assert top5["fraud_captured"] == 3
    assert top5["false_alerts"] == 2
    assert pytest.approx(top5["recall"], abs=1e-5) == 0.60
    assert pytest.approx(top5["precision"], abs=1e-5) == 0.60
    assert pytest.approx(top5["fpr"], abs=1e-4) == 2.0 / 95.0


def test_deterministic_score_tie_breaking():
    """Verify that score ties are broken deterministically by tie_breaker index."""
    # 4 samples with identical scores: [0.5, 0.5, 0.5, 0.5]
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])

    # Tie breaker ascending: 1, 2, 3, 4
    # Top 50% budget reviews 2 samples: will take indices 0 and 1
    tb1 = np.array([1, 2, 3, 4])
    res1 = compute_budget_metrics(y_true, y_score, budgets=[0.5], tie_breaker=tb1)
    # Samples reviewed: index 0 (y=0) and index 1 (y=1) -> TP = 1
    assert res1["top_50.0pct"]["fraud_captured"] == 1

    # Inverted tie breaker: 4, 3, 2, 1
    # Top 50% budget reviews 2 samples: will take indices 3 and 2
    tb2 = np.array([4, 3, 2, 1])
    res2 = compute_budget_metrics(y_true, y_score, budgets=[0.5], tie_breaker=tb2)
    # Samples reviewed: index 3 (y=1) and index 2 (y=0) -> TP = 1
    assert res2["top_50.0pct"]["fraud_captured"] == 1


def test_budget_rounding():
    """Verify budget volume rounding when total rows times budget is non-integer."""
    # N = 25, budget = 0.05 -> 25 * 0.05 = 1.25 -> rounds to 1
    y_true = np.zeros(25, dtype=int)
    y_true[0] = 1
    y_score = np.linspace(0, 1, 25)

    res = compute_budget_metrics(y_true, y_score, budgets=[0.05])
    assert res["top_5.0pct"]["review_volume"] == 1
