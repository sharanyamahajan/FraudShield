"""Evaluate Model B and Set C1 Probability Calibration on the Held-Out Test Split from Saved Scores.

Strict Zero-Leakage Constraints:
- Reads ONLY saved test scores (reports/test_probs_b.npy, reports/fraudshield.db test_predictions) and saved labels.
- Does NOT access or read raw test transaction tables or raw/processed test parquet files.
- Applies the fitted Isotonic calibrators (models/calibrator_b.pkl and models/calibrator_c1.pkl fit on validation only).
- Verifies exact TransactionID order alignment and label consistency without leakage.
- Updates reports/test_evaluation_summary.json, reports/test_evaluation_summary.csv, and reports/fraudshield.db.
- Labeled: 'post-hoc, test split already seen'.
"""

import hashlib
import json
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_TEST_ROWS = 88581
EXPECTED_TID_SHA256 = "2cdc1792935bf23f2a14ec831f93335bda9c5e0127ebe277eb54c13d5689a6dc"
EXPECTED_MODEL_B_TEST_PRAUC = 0.429595
EXPECTED_MODEL_B_TEST_ROCAUC = 0.870473


def check_test_alignment(reports_dir: Path = Path("reports")) -> Dict[str, Any]:
    """Verify that test_probs_b.npy aligns row-for-row with test labels in SQLite test_predictions."""
    scores_path = reports_dir / "test_probs_b.npy"
    db_path = reports_dir / "fraudshield.db"

    if not scores_path.exists():
        raise FileNotFoundError(f"Saved test scores not found at {scores_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database with test predictions not found at {db_path}")

    probs_b = np.load(scores_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT transaction_id, transaction_dt, is_fraud FROM test_predictions ORDER BY rowid"
    ).fetchall()
    conn.close()

    if len(probs_b) != len(rows) or len(probs_b) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Row count mismatch: probs_b has {len(probs_b)} rows, SQLite has {len(rows)} rows (expected {EXPECTED_TEST_ROWS})"
        )

    tids = np.array([r[0] for r in rows], dtype=np.int64)
    tdts = np.array([r[1] for r in rows], dtype=np.int64)
    y_test = np.array([r[2] for r in rows], dtype=np.int32)

    # 1. Monotonicity of TransactionDT (chronological split)
    is_dt_monotonic = bool(np.all(np.diff(tdts) >= 0))
    if not is_dt_monotonic:
        raise ValueError("TransactionDT sequence in test_predictions is not monotonic non-decreasing")

    # 2. SHA-256 fingerprint of TransactionID order
    tid_hash = hashlib.sha256(tids.tobytes()).hexdigest()
    if tid_hash != EXPECTED_TID_SHA256:
        raise ValueError(
            f"TransactionID hash mismatch: got {tid_hash}, expected {EXPECTED_TID_SHA256}"
        )

    # 3. Label-to-score alignment confirmation via baseline ranking metrics
    prauc = float(average_precision_score(y_test, probs_b))
    rocauc = float(roc_auc_score(y_test, probs_b))

    if abs(prauc - EXPECTED_MODEL_B_TEST_PRAUC) > 1e-4:
        raise ValueError(
            f"Score alignment failure: PR-AUC is {prauc:.6f}, expected {EXPECTED_MODEL_B_TEST_PRAUC:.6f}"
        )
    if abs(rocauc - EXPECTED_MODEL_B_TEST_ROCAUC) > 1e-4:
        raise ValueError(
            f"Score alignment failure: ROC-AUC is {rocauc:.6f}, expected {EXPECTED_MODEL_B_TEST_ROCAUC:.6f}"
        )

    logger.info(
        "Alignment check PASSED: %d rows verified (TID hash: %s, PR-AUC: %.6f, ROC-AUC: %.6f)",
        len(probs_b),
        tid_hash[:16] + "...",
        prauc,
        rocauc,
    )
    return {
        "aligned": True,
        "row_count": len(probs_b),
        "tid_hash": tid_hash,
        "is_dt_monotonic": is_dt_monotonic,
        "prauc": prauc,
        "rocauc": rocauc,
    }


def compute_model_b_test_brier(
    reports_dir: Path = Path("reports"),
    models_dir: Path = Path("models"),
) -> Tuple[float, float]:
    """Compute raw and calibrated Brier scores for Model B on test using saved scores and labels."""
    check_test_alignment(reports_dir)

    scores_path = reports_dir / "test_probs_b.npy"
    db_path = reports_dir / "fraudshield.db"
    calib_path = models_dir / "calibrator_b.pkl"

    if not calib_path.exists():
        raise FileNotFoundError(f"Model B calibrator not found at {calib_path}")

    probs_b = np.load(scores_path)
    probs_b_clipped = np.clip(probs_b, 1e-6, 1.0 - 1e-6)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    y_test = np.array(
        [r[0] for r in cur.execute("SELECT is_fraud FROM test_predictions ORDER BY rowid").fetchall()],
        dtype=np.int32,
    )
    conn.close()

    raw_brier = float(brier_score_loss(y_test, probs_b_clipped))

    with open(calib_path, "rb") as f:
        calibrator = pickle.load(f)

    cal_probs = np.clip(calibrator.predict(probs_b_clipped), 0.0, 1.0)
    cal_brier = float(brier_score_loss(y_test, cal_probs))

    return raw_brier, cal_brier


def compute_model_c1_test_brier(
    reports_dir: Path = Path("reports"),
    models_dir: Path = Path("models"),
) -> Tuple[float, float]:
    """Compute raw and calibrated Brier scores for Set C1 on test from saved SQLite scores."""
    db_path = reports_dir / "fraudshield.db"
    calib_path = models_dir / "calibrator_c1.pkl"

    if not db_path.exists():
        raise FileNotFoundError(f"Database with test scores not found at {db_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Set C1 calibrator not found at {calib_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT raw_risk_score, is_fraud FROM test_predictions ORDER BY rowid"
    ).fetchall()
    conn.close()

    c1_scores = np.array([r[0] for r in rows], dtype=np.float32)
    y_test = np.array([r[1] for r in rows], dtype=np.int32)

    c1_scores_clipped = np.clip(c1_scores, 1e-6, 1.0 - 1e-6)
    raw_brier = float(brier_score_loss(y_test, c1_scores_clipped))

    with open(calib_path, "rb") as f:
        calibrator = pickle.load(f)

    cal_probs = np.clip(calibrator.predict(c1_scores_clipped), 0.0, 1.0)
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

    logger.info("Step 1: Running alignment and order verification...")
    alignment = check_test_alignment(reports_dir=reports_dir)

    logger.info("Step 2: Computing Model B test Brier scores from saved scores only...")
    raw_brier_b, cal_brier_b = compute_model_b_test_brier(reports_dir=reports_dir, models_dir=models_dir)

    logger.info("=================================================================")
    logger.info("Model B Test Split Calibration Results (Post-Hoc, Test Split Already Seen):")
    logger.info("  Raw (Uncalibrated) Brier Score: %.6f", raw_brier_b)
    logger.info("  Calibrated (Isotonic) Brier Score: %.6f", cal_brier_b)
    logger.info("  Calibration Delta: %+.6f", cal_brier_b - raw_brier_b)
    logger.info("=================================================================")

    if (models_dir / "calibrator_c1.pkl").exists():
        logger.info("Step 3: Reproducing Set C1 test Brier scores from saved scores...")
        raw_brier_c1, cal_brier_c1 = compute_model_c1_test_brier(reports_dir=reports_dir, models_dir=models_dir)
        logger.info("=================================================================")
        logger.info("Set C1 Test Split Calibration Results (Reproducible via calibrator_c1.pkl):")
        logger.info("  Raw (Uncalibrated) Brier Score: %.6f", raw_brier_c1)
        logger.info("  Calibrated (Isotonic) Brier Score: %.6f", cal_brier_c1)
        logger.info("  Calibration Delta: %+.6f", cal_brier_c1 - raw_brier_c1)
        logger.info("=================================================================")

    update_artifacts_with_brier(raw_brier_b, cal_brier_b, reports_dir=reports_dir)


if __name__ == "__main__":
    main()
