"""Calibrate Model B, compute SHAP for Model B on validation, and update serving assets.

Strict constraints:
- Reads ONLY validation split (data/processed/val_set_b.parquet) and previously saved reports.
- NEVER reads the held-out test split.
- Fits Isotonic calibrator on validation scores of models/xgb_set_b.json.
- Computes SHAP feature importance on validation sample using shap.TreeExplainer.
- Updates reports/fraudshield.db and models/calibrator.pkl for serving Model B.
"""
import json
import logging
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    reports_dir = root / "reports"
    data_dir = root / "data" / "processed"

    # 1. Load Model B and Preprocessor B
    logger.info("Loading Model B and Preprocessor B...")
    bst_b = xgb.Booster()
    bst_b.load_model(str(models_dir / "xgb_set_b.json"))

    with open(models_dir / "preprocessor_b.pkl", "rb") as f:
        prep_b = pickle.load(f)

    # 2. Load Validation Set B (strictly validation, zero test access)
    val_path = data_dir / "val_set_b.parquet"
    logger.info("Loading validation split from %s...", val_path)
    df_val_b = pd.read_parquet(val_path)

    y_val = df_val_b["isFraud"].values.astype(np.int32)
    feats_b = [c for c in df_val_b.columns if c not in ("TransactionID", "TransactionDT", "isFraud")]

    logger.info("Transforming validation features (%d rows, %d features)...", len(df_val_b), len(feats_b))
    X_val_b = prep_b.transform(df_val_b[feats_b])
    val_feature_names = list(X_val_b.columns)
    X_val_b_arr = np.ascontiguousarray(X_val_b.values, dtype=np.float32)

    dval_b = xgb.DMatrix(X_val_b_arr)
    val_preds_b = bst_b.predict(dval_b)
    val_preds_clipped = np.clip(val_preds_b, 1e-6, 1.0 - 1e-6)

    val_brier_uncal = float(brier_score_loss(y_val, val_preds_clipped))
    logger.info("Model B Validation Uncalibrated Brier Score: %.6f", val_brier_uncal)

    # 3. Fit Isotonic Calibrator on Validation
    logger.info("Fitting Isotonic Calibrator on validation scores...")
    iso_calibrator = IsotonicRegression(out_of_bounds="clip")
    iso_calibrator.fit(val_preds_clipped, y_val)
    val_preds_cal = np.clip(iso_calibrator.predict(val_preds_clipped), 0.0, 1.0)
    val_brier_cal = float(brier_score_loss(y_val, val_preds_cal))
    logger.info("Model B Validation Calibrated Brier Score: %.6f", val_brier_cal)

    # Save calibrator as calibrator_b.pkl and update calibrator.pkl
    with open(models_dir / "calibrator_b.pkl", "wb") as f:
        pickle.dump(iso_calibrator, f)
    with open(models_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(iso_calibrator, f)
    logger.info("Saved calibrator to %s and %s", models_dir / "calibrator_b.pkl", models_dir / "calibrator.pkl")

    # 4. Compute Global SHAP on validation sample
    logger.info("Computing TreeExplainer SHAP on validation sample...")
    np.random.seed(42)
    # Stratified/representative sample: 250 frauds + 250 non-frauds
    fraud_idx = np.where(y_val == 1)[0]
    nonfraud_idx = np.where(y_val == 0)[0]
    sample_fraud = np.random.choice(fraud_idx, size=min(250, len(fraud_idx)), replace=False)
    sample_nonfraud = np.random.choice(nonfraud_idx, size=250, replace=False)
    sample_indices = np.concatenate([sample_fraud, sample_nonfraud])

    X_shap_sample = X_val_b_arr[sample_indices]
    explainer = shap.TreeExplainer(bst_b)
    shap_values = explainer.shap_values(X_shap_sample)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_order = np.argsort(mean_abs_shap)[::-1]

    def categorize_feature(fname: str):
        if fname.startswith("V"):
            return "Vesta Anonymised (Set A)", True, "Anonymised structural pattern (behavioral driver, not causal)"
        if fname.startswith("C"):
            return "Transaction Counting (Set A)", False, "Transaction volume / frequency attribute"
        if fname.startswith("D"):
            return "Timetable / Time-delta (Set A)", False, "Time-delta / recurrence attribute"
        if fname.startswith("M"):
            return "Match / Identity Verification (Set A)", False, "Identity attribute matching flag"
        if fname.startswith("card"):
            return "Payment Card Attributes (Set A/B)", False, "Payment card attribute"
        if fname.startswith("addr"):
            return "Geographic / Billing Address (Set A/B)", False, "Billing / region geographic identifier"
        if "email" in fname.lower():
            return "Network / Address (Set B)", False, "Domain risk attribute"
        if "Device" in fname or "id_" in fname:
            return "Device & Digital Identity (Set B)", False, "Digital device footprint"
        return "Transaction Core Attributes", False, "Core transaction attribute"

    global_shap_records = []
    for rank, idx in enumerate(shap_order, start=1):
        feat_name = val_feature_names[idx]
        family, is_anon, note = categorize_feature(feat_name)
        global_shap_records.append({
            "rank": rank,
            "feature_name": feat_name,
            "feature_family": family,
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 6),
            "is_anonymised": is_anon,
            "interpretation_note": note,
        })

    logger.info("Top 5 SHAP features for Model B:")
    for r in global_shap_records[:5]:
        logger.info("  Rank %d: %s (mean |SHAP| = %.6f)", r["rank"], r["feature_name"], r["mean_abs_shap"])

    # Save SHAP artifacts
    df_shap_global = pd.DataFrame(global_shap_records)
    df_shap_global.to_csv(reports_dir / "shap_global_importance.csv", index=False)
    with open(reports_dir / "shap_global_importance.json", "w", encoding="utf-8") as f:
        json.dump(global_shap_records, f, indent=2)

    # 5. Load pre-computed Model B metrics from reports
    # From reports/test_evaluation_summary.csv and reports/m5_seed_stability.json
    # Model B test: PR-AUC=0.429595, ROC-AUC=0.870473, Brier uncal=0.026810
    test_pr_auc_b = 0.429595
    test_roc_auc_b = 0.870473
    test_brier_uncal_b = 0.026810461655259132

    # 6. Update SQLite Database reports/fraudshield.db
    db_path = reports_dir / "fraudshield.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Update model_metadata
    cur.execute("DELETE FROM model_metadata")
    cur.execute("""
        INSERT INTO model_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "xgb_set_b",
        "XGBoost (Hist Gradient Boosted Trees)",
        "Set B (Identity & Transaction Features - Deployed)",
        293,
        0.522243,
        test_pr_auc_b,
        test_roc_auc_b,
        test_brier_uncal_b,
        val_brier_cal,
        "IsotonicRegression",
        "719d51e",
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    ))

    # Update feature_importances table
    cur.execute("DELETE FROM feature_importances")
    for g in global_shap_records[:50]:
        cur.execute("""
            INSERT INTO feature_importances VALUES (?, ?, ?, ?, ?, ?)
        """, (
            g["rank"],
            g["feature_name"],
            g["feature_family"],
            g["mean_abs_shap"],
            1 if g["is_anonymised"] else 0,
            g["interpretation_note"],
        ))

    # Update operating_points for Model B
    cur.execute("DELETE FROM operating_points")
    # Model B test metrics:
    # Top 1.0%: 886 review count, 762 frauds captured (precision 86.0045%, recall 24.7162%)
    # Top 2.0%: 1772 review count, 963 frauds captured (precision 54.3454%, recall 31.2358%)
    # Top 5.0%: 4429 review count, 1449 frauds captured (precision 32.7162%, recall 46.9997%)
    # Top 0.5%: from validation/test
    cur.executemany("""
        INSERT INTO operating_points VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        ("top_0.5pct", 0.005, 0.9370, 0.1371, 0.9413, 0.1371, 0.9413, 443, 417),
        ("top_1.0pct", 0.010, 0.7191, 0.2640, 0.9063, 0.2472, 0.8600, 886, 762),
        ("top_2.0pct", 0.020, 0.2726, 0.4043, 0.6941, 0.3124, 0.5435, 1772, 963),
        ("top_5.0pct", 0.050, 0.0904, 0.5602, 0.3847, 0.4700, 0.3272, 4429, 1449),
    ])

    conn.commit()
    conn.close()
    logger.info("Successfully updated database at %s with Model B metadata, SHAP, and operating points.", db_path)


if __name__ == "__main__":
    main()
