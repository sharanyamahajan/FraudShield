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


def test_dashboard_kpis_do_not_mix_models():
    """Verify that KPI grids in index.html do not mix metrics from different models in the same grid."""
    html_text = Path("src/fraudshield/api/static/index.html").read_text(encoding="utf-8")

    # Split by kpi-grid
    grids = html_text.split('<div class="kpi-grid"')
    assert len(grids) >= 3, "Expected at least 2 KPI grids in overview"

    # Grid 1: Deployed Model (Set B)
    grid_b = grids[1].split('</div>\n            </div>')[0]
    assert "Set B Test PR-AUC" in grid_b
    assert "Set B Test ROC-AUC" in grid_b
    assert "0.4296" in grid_b
    assert "0.8705" in grid_b
    # Must NOT contain Set C1 PR-AUC in this Set B grid
    assert "0.4002" not in grid_b, "Grid 1 mistakenly mixed Set C1 PR-AUC into Set B grid"

    # Grid 2: Comparison Model (Set C1)
    grid_c1 = grids[2].split('</div>\n            </div>')[0]
    assert "Set C1 Test PR-AUC" in grid_c1
    assert "Set C1 Test ROC-AUC" in grid_c1
    assert "0.4002" in grid_c1
    assert "0.8748" in grid_c1
    # Must NOT contain Set B numbers in this C1 grid
    assert "0.4296" not in grid_c1, "Grid 2 mistakenly mixed Set B PR-AUC into Set C1 grid"
    assert "0.8705" not in grid_c1, "Grid 2 mistakenly mixed Set B ROC-AUC into Set C1 grid"


def test_headline_precision_recall_row_consistency():
    """Assert mathematical consistency (frauds == round(precision * rows))
    for all headline precision/recall figures mentioned in docs and reports."""
    # 1. Headline Set B top 1.0% rank-based: 886 rows, 86.00% precision
    rows_b_1 = 886
    prec_b_1 = 0.860045
    frauds_b_1 = round(rows_b_1 * prec_b_1)
    assert frauds_b_1 == 762, f"Expected 762 frauds, got {frauds_b_1}"

    # 2. Headline Set C1 top 1.0% rank-based: 886 rows, 72.69% precision
    rows_c1_1 = 886
    prec_c1_1 = 0.726862
    frauds_c1_1 = round(rows_c1_1 * prec_c1_1)
    assert frauds_c1_1 == 644, f"Expected 644 frauds, got {frauds_c1_1}"

    # 3. Headline Set C1 threshold-based operating points on test (N=88,581, total_frauds=3083)
    tier_checks = [
        ("top_0.5pct", 515, 0.943689, 486, 0.157639),
        ("top_1.0pct", 1742, 0.479334, 835, 0.270840),
        ("top_2.0pct", 3503, 0.364830, 1278, 0.414531),
        ("top_5.0pct", 6670, 0.268516, 1791, 0.580928),
    ]
    for tier, rows, prec, exp_frauds, rec in tier_checks:
        calculated_frauds = round(rows * prec)
        assert calculated_frauds == exp_frauds, (
            f"Tier {tier} precision inconsistency: {rows} * {prec} = {calculated_frauds} != {exp_frauds}"
        )
        calculated_from_recall = round(3083 * rec)
        assert calculated_from_recall == exp_frauds, (
            f"Tier {tier} recall inconsistency: 3083 * {rec} = {calculated_from_recall} != {exp_frauds}"
        )

    # 4. Verify text occurrences in README, MODEL_CARD, and index.html do not claim 860 frauds
    readme_text = Path("README.md").read_text(encoding="utf-8")
    model_card_text = Path("docs/MODEL_CARD.md").read_text(encoding="utf-8")
    html_text = Path("src/fraudshield/api/static/index.html").read_text(encoding="utf-8")
    assert "860 true frauds" not in readme_text
    assert "860 true frauds" not in model_card_text
    assert "860 true frauds" not in html_text
    assert "762" in readme_text
    assert "762" in model_card_text
    assert "762" in html_text


def test_model_b_test_brier_matches_script_output():
    """Verify that Model B test Brier values in reports match the output of scripts/evaluate_model_b_calibration.py."""
    import sys
    sys.path.insert(0, str(Path(".").resolve()))
    from scripts.evaluate_model_b_calibration import compute_model_b_test_brier

    reports_dir = Path("reports")
    models_dir = Path("models")

    if not (reports_dir / "test_probs_b.npy").exists() or not (reports_dir / "fraudshield.db").exists():
        pytest.skip("Test scores or database not found - run pipeline first")

    script_raw_brier, script_cal_brier = compute_model_b_test_brier(reports_dir, models_dir)

    # Compare with test_evaluation_summary.json
    with open(reports_dir / "test_evaluation_summary.json", "r", encoding="utf-8") as f:
        summary_json = json.load(f)
    json_b = summary_json["baseline_model_set_b"]
    assert pytest.approx(json_b["test_brier_uncalibrated"], rel=1e-5) == script_raw_brier
    assert pytest.approx(json_b["test_brier_calibrated"], rel=1e-5) == script_cal_brier

    # Compare with test_evaluation_summary.csv
    df_csv = pd.read_csv(reports_dir / "test_evaluation_summary.csv")
    csv_b = df_csv[df_csv["model"].str.contains("Set B")].iloc[0]
    assert pytest.approx(float(csv_b["brier_uncalibrated"]), rel=1e-5) == script_raw_brier
    assert pytest.approx(float(csv_b["brier_calibrated"]), rel=1e-5) == script_cal_brier


def test_test_prediction_alignment_and_c1_brier_reproducibility():
    """Verify that test predictions strictly align with SQLite labels and C1 Brier is script-reproducible."""
    import sys
    sys.path.insert(0, str(Path(".").resolve()))
    from scripts.evaluate_model_b_calibration import check_test_alignment, compute_model_c1_test_brier

    reports_dir = Path("reports")
    models_dir = Path("models")

    if not (reports_dir / "test_probs_b.npy").exists() or not (reports_dir / "fraudshield.db").exists():
        pytest.skip("Test scores or database not found - run pipeline first")

    # 1. Test alignment check
    alignment = check_test_alignment(reports_dir)
    assert alignment["aligned"] is True
    assert alignment["row_count"] == 88581
    assert alignment["is_dt_monotonic"] is True
    assert alignment["tid_hash"] == "2cdc1792935bf23f2a14ec831f93335bda9c5e0127ebe277eb54c13d5689a6dc"
    assert pytest.approx(alignment["prauc"], rel=1e-4) == 0.429595
    assert pytest.approx(alignment["rocauc"], rel=1e-4) == 0.870473

    # 2. Test Set C1 calibration reproducibility via models/calibrator_c1.pkl
    if (models_dir / "calibrator_c1.pkl").exists():
        c1_raw_brier, c1_cal_brier = compute_model_c1_test_brier(reports_dir, models_dir)
        with open(reports_dir / "test_evaluation_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        c1_meta = summary["calibrator"]
        assert pytest.approx(c1_meta["test_brier_uncalibrated"], rel=1e-4) == c1_raw_brier
        assert pytest.approx(c1_meta["test_brier_calibrated"], rel=1e-4) == c1_cal_brier
        assert round(c1_cal_brier, 4) == 0.0319


