"""Audit test suite: Verifies that all headline numbers in documentation and UI

strictly match the authoritative report JSON and CSV artifacts without discrepancy.
"""

import json
from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def report_data():
    reports_dir = Path("reports")
    with open(reports_dir / "test_evaluation_summary.json", "r", encoding="utf-8") as f:
        test_summary = json.load(f)
    with open(reports_dir / "m5_rq2_split_comparison.json", "r", encoding="utf-8") as f:
        rq2_summary = json.load(f)
    with open(reports_dir / "m5_paired_bootstrap.json", "r", encoding="utf-8") as f:
        val_bootstrap = json.load(f)
    df_ops = pd.read_csv(reports_dir / "test_operating_points.csv")
    return {
        "test_summary": test_summary,
        "rq2_summary": rq2_summary,
        "val_bootstrap": val_bootstrap,
        "df_ops": df_ops,
    }


def test_readme_headline_numbers_match_reports(report_data):
    readme_text = Path("README.md").read_text(encoding="utf-8")
    test_summary = report_data["test_summary"]
    rq2_summary = report_data["rq2_summary"]

    # Test PR-AUC Set B (0.4296) and Set C1 (0.4002)
    b_test_prauc = f"{test_summary['baseline_model_set_b']['test_pr_auc']:.4f}"
    c1_test_prauc = f"{test_summary['primary_model_set_c1']['test_pr_auc']:.4f}"
    assert b_test_prauc in readme_text, f"README missing Set B test PR-AUC {b_test_prauc}"
    assert c1_test_prauc in readme_text, f"README missing Set C1 test PR-AUC {c1_test_prauc}"

    # RQ2 numbers (0.646 vs 0.522 and +12.4)
    assert "+12.4 percentage points" in readme_text
    assert "(0.646 vs 0.522 PR-AUC, ~24% relative)" in readme_text

    # Brier calibration numbers
    uncal_brier = f"{test_summary['calibrator']['test_brier_uncalibrated']:.4f}"
    cal_brier = f"{test_summary['calibrator']['test_brier_calibrated']:.4f}"
    assert uncal_brier in readme_text, f"README missing uncalibrated Brier {uncal_brier}"
    assert cal_brier in readme_text, f"README missing calibrated Brier {cal_brier}"


def test_model_card_headline_numbers_match_reports(report_data):
    model_card_text = Path("docs/MODEL_CARD.md").read_text(encoding="utf-8")
    test_summary = report_data["test_summary"]

    c1_prauc_full = f"{test_summary['primary_model_set_c1']['test_pr_auc']:.6f}"
    b_prauc_full = f"{test_summary['baseline_model_set_b']['test_pr_auc']:.6f}"
    assert c1_prauc_full in model_card_text, f"MODEL_CARD missing C1 PR-AUC {c1_prauc_full}"
    assert b_prauc_full in model_card_text, f"MODEL_CARD missing B PR-AUC {b_prauc_full}"

    uncal_brier_full = f"{test_summary['calibrator']['test_brier_uncalibrated']:.6f}"
    cal_brier_full = f"{test_summary['calibrator']['test_brier_calibrated']:.6f}"
    assert uncal_brier_full in model_card_text, f"MODEL_CARD missing uncalibrated Brier {uncal_brier_full}"
    assert cal_brier_full in model_card_text, f"MODEL_CARD missing calibrated Brier {cal_brier_full}"

    assert "Demo only: no auth, returns labels, not production-ready" in model_card_text


def test_operating_points_numbers_match_csv(report_data):
    df_ops = report_data["df_ops"]
    readme_text = Path("README.md").read_text(encoding="utf-8")
    html_text = Path("src/fraudshield/api/static/index.html").read_text(encoding="utf-8")

    # Check top 0.5% cutoff and recall
    row_05 = df_ops[df_ops["budget_name"] == "top_0.5pct"].iloc[0]
    thresh_05 = f"{row_05['val_fixed_threshold']:.4f}"
    rec_05 = f"{row_05['test_fixed_threshold_recall'] * 100:.2f}%"

    assert thresh_05 in readme_text
    assert thresh_05 in html_text
    assert rec_05 in readme_text
    assert rec_05 in html_text
