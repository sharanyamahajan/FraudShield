"""Evaluate Model B Probability Calibration on the Held-Out Test Split from Saved Scores.

Strict Zero-Leakage Constraints:
- Reads ONLY saved test scores (reports/test_probs_b.npy) and saved labels from SQLite (reports/fraudshield.db).
- Does NOT access or read raw test transaction tables or raw/processed test parquet files.
- Applies the served Isotonic calibrator (models/calibrator_b.pkl fit on validation only).
- Updates reports/test_evaluation_summary.json, reports/test_evaluation_summary.csv, and reports/fraudshield.db.
- Labeled: 'post-hoc, test split already seen'.
"""

import json
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_model_b_test_brier(
    reports_dir: Path = Path("reports"),
    models_dir: Path = Path("models"),
) -> Tuple[float, float]:
    """Compute raw and calibrated Brier scores for Model B on test using saved scores and labels."""
    scores_path = reports_dir / "test_probs_b.npy"
    db_path = reports_dir / "fraudshield.db"
    calib_path = models_dir / "calibrator_b.pkl"

    if not scores_path.exists():
        raise FileNotFoundError(f"Saved test scores not found at {scores_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database with test labels not found at {db_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Model B calibrator not found at {calib_path}")

    # 1. Load saved scores
    probs_b = np.load(scores_path)
    probs_b_clipped = np.clip(probs_b, 1e-6, 1.0 - 1e-6)

    # 2. Load saved test labels from SQLite test_predictions table
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    y_test = np.array([r[0] for r in cur.execute("SELECT is_fraud FROM test_predictions ORDER BY rowid").fetchall()], dtype=np.int32)
    conn.close()

    assert len(probs_b) == len(y_test) == 88581, f"Expected 88,581 test rows, got {len(probs_b)} and {len(y_test)}"

    # 3. Compute raw Brier score
    raw_brier = float(brier_score_loss(y_test, probs_b_clipped))

    # 4. Load calibrator and compute calibrated test Brier score
    with open(calib_path, "rb") as f:
        calibrator = pickle.load(f)

    cal_probs = np.clip(calibrator.predict(probs_b_clipped), 0.0, 1.0)
    cal_brier = float(brier_score_loss(y_test, cal_probs))

    return raw_brier, cal_brier


def update_artifacts_with_brier(
    raw_brier: float,
    cal_brier: float,
    reports_dir: Path = Path("reports"),
):
    """Update JSON, CSV, and SQLite database with computed Model B Brier scores."""
    json_path = reports_dir / "test_evaluation_summary.json"
    csv_path = reports_dir / "test_evaluation_summary.csv"
    db_path = reports_dir / "fraudshield.db"

    # 1. Update JSON
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "baseline_model_set_b" in data:
            data["baseline_model_set_b"]["test_brier_uncalibrated"] = raw_brier
            data["baseline_model_set_b"]["test_brier_calibrated"] = cal_brier
            data["baseline_model_set_b"]["calibration_evaluation_note"] = "post-hoc, test split already seen"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Updated %s with Model B Brier scores (raw=%.6f, cal=%.6f)", json_path, raw_brier, cal_brier)

    # 2. Update CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        mask = df["model"].str.contains("Set B")
        if mask.any():
            df.loc[mask, "brier_uncalibrated"] = raw_brier
            df.loc[mask, "brier_calibrated"] = cal_brier
            df.to_csv(csv_path, index=False)
            logger.info("Updated %s with Model B Brier scores", csv_path)

    # 3. Update SQLite Database
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("UPDATE model_metadata SET brier_uncalibrated = ?, brier_calibrated = ?", (raw_brier, cal_brier))
        conn.commit()
        conn.close()
        logger.info("Updated %s model_metadata with Model B Brier scores", db_path)


def main():
    root = Path(__file__).resolve().parent.parent
    reports_dir = root / "reports"
    models_dir = root / "models"

    logger.info("Computing Model B test Brier scores from saved scores only...")
    raw_brier, cal_brier = compute_model_b_test_brier(reports_dir=reports_dir, models_dir=models_dir)

    logger.info("=================================================================")
    logger.info("Model B Test Split Calibration Results (Post-Hoc, Test Split Already Seen):")
    logger.info("  Raw (Uncalibrated) Brier Score: %.6f", raw_brier)
    logger.info("  Calibrated (Isotonic) Brier Score: %.6f", cal_brier)
    logger.info("  Calibration Delta: %+.6f", cal_brier - raw_brier)
    logger.info("=================================================================")

    update_artifacts_with_brier(raw_brier, cal_brier, reports_dir=reports_dir)


if __name__ == "__main__":
    main()
