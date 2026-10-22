"""FastAPI Application for FraudShield Risk Scoring and Explainability."""

import json
import logging
import pickle
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import xgboost as xgb

from fraudshield.api.schemas import (
    HealthResponse,
    ModelMetadataResponse,
    OperatingPointItem,
    FeatureImportanceItem,
    TransactionItem,
    PaginatedTransactionsResponse,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger("fraudshield.api")

# Global in-memory model and preprocessor state
model_state: Dict[str, Any] = {
    "model": None,
    "preprocessor": None,
    "calibrator": None,
    "feature_columns": None,
    "operating_thresholds": {},
    "db_path": Path("reports/fraudshield.db"),
    "reports_dir": Path("reports"),
}


def get_db_connection() -> sqlite3.Connection:
    """Create a connection to the SQLite database with row factory."""
    db_path = model_state["db_path"]
    if not db_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database file not found at {db_path}",
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, preprocessor, and calibration assets at startup."""
    models_dir = Path("models")
    model_file = models_dir / "xgb_set_c1.json"
    prep_file = models_dir / "preprocessor_c1.pkl"
    calib_file = models_dir / "calibrator.pkl"

    if model_file.exists():
        try:
            bst = xgb.Booster()
            bst.load_model(model_file)
            model_state["model"] = bst
            logger.info("Loaded XGBoost model from %s", model_file)
        except Exception as e:
            logger.warning("Failed to load XGBoost model: %s", e)

    if prep_file.exists():
        try:
            with open(prep_file, "rb") as f:
                model_state["preprocessor"] = pickle.load(f)
            logger.info("Loaded TreePreprocessor from %s", prep_file)
        except Exception as e:
            logger.warning("Failed to load preprocessor: %s", e)

    if calib_file.exists():
        try:
            with open(calib_file, "rb") as f:
                model_state["calibrator"] = pickle.load(f)
            logger.info("Loaded calibrator from %s", calib_file)
        except Exception as e:
            logger.warning("Failed to load calibrator: %s", e)

    # Load operating thresholds from SQLite or JSON
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        rows = cur.execute("SELECT budget_name, score_threshold FROM operating_points").fetchall()
        for r in rows:
            model_state["operating_thresholds"][r["budget_name"]] = float(r["score_threshold"])
        conn.close()
    except Exception as e:
        logger.warning("Could not pre-load operating thresholds: %s", e)
        model_state["operating_thresholds"] = {
            "top_0.5pct": 0.868444,
            "top_1.0pct": 0.613925,
            "top_2.0pct": 0.266463,
            "top_5.0pct": 0.115361,
        }

    yield

    model_state.clear()


app = FastAPI(
    title="FraudShield Risk Engine API",
    description="Production Fraud Risk Prioritization, Scoring & Explainability Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """System health and model readiness endpoint."""
    model_loaded = model_state.get("model") is not None
    db_connected = model_state["db_path"].exists()
    return HealthResponse(
        status="healthy" if (model_loaded and db_connected) else "degraded",
        model_loaded=model_loaded,
        db_connected=db_connected,
        version="1.0.0",
    )


@app.get("/api/metadata", response_model=ModelMetadataResponse)
async def get_model_metadata():
    """Retrieve model training, evaluation, and git provenance metadata."""
    conn = get_db_connection()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM model_metadata LIMIT 1").fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Model metadata not found in database")
    return ModelMetadataResponse(
        model_id=row["model_id"],
        model_name=row["model_name"],
        feature_set=row["feature_set"],
        best_iteration=row["best_iteration"],
        val_pr_auc=row["val_pr_auc"],
        test_pr_auc=row["test_pr_auc"],
        test_roc_auc=row["test_roc_auc"],
        brier_uncalibrated=row["brier_uncalibrated"],
        brier_calibrated=row["brier_calibrated"],
        calibrator_type=row["calibrator_type"],
        git_commit_hash=row["git_commit_hash"],
        evaluated_at=row["evaluated_at"],
    )


@app.get("/api/operating-points", response_model=List[OperatingPointItem])
async def get_operating_points():
    """Retrieve operational capacity review tiers and validation/test tradeoffs."""
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM operating_points ORDER BY budget_ratio ASC").fetchall()
    conn.close()
    return [
        OperatingPointItem(
            budget_name=r["budget_name"],
            budget_ratio=r["budget_ratio"],
            score_threshold=r["score_threshold"],
            val_recall=r["val_recall"],
            val_precision=r["val_precision"],
            test_recall=r["test_recall"],
            test_precision=r["test_precision"],
            test_review_count=r["test_review_count"],
            test_fraud_captured=r["test_fraud_captured"],
        )
        for r in rows
    ]


@app.get("/api/features/importance", response_model=List[FeatureImportanceItem])
async def get_feature_importances(limit: int = Query(50, ge=1, le=100)):
    """Retrieve global TreeExplainer SHAP feature importance rankings."""
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM feature_importances ORDER BY rank ASC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [
        FeatureImportanceItem(
            rank=r["rank"],
            feature_name=r["feature_name"],
            feature_family=r["feature_family"],
            mean_abs_shap=r["mean_abs_shap"],
            is_anonymised=bool(r["is_anonymised"]),
            interpretation_note=r["interpretation_note"],
        )
        for r in rows
    ]


@app.get("/api/transactions", response_model=PaginatedTransactionsResponse)
async def get_transactions(
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    tier: Optional[str] = Query(None, description="Filter by tier: top_0.5pct, top_1.0pct, top_2.0pct, top_5.0pct"),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    is_fraud: Optional[int] = Query(None, ge=0, le=1),
):
    """Query scored test transactions with flexible filtering and pagination."""
    conn = get_db_connection()
    cur = conn.cursor()

    conditions = []
    params = []

    if tier == "top_0.5pct":
        conditions.append("flag_top_0_5pct = 1")
    elif tier == "top_1.0pct":
        conditions.append("flag_top_1_0pct = 1")
    elif tier == "top_2.0pct":
        conditions.append("flag_top_2_0pct = 1")
    elif tier == "top_5.0pct":
        conditions.append("flag_top_5_0pct = 1")

    if min_score is not None:
        conditions.append("raw_risk_score >= ?")
        params.append(min_score)

    if is_fraud is not None:
        conditions.append("is_fraud = ?")
        params.append(is_fraud)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    count_query = f"SELECT COUNT(*) FROM test_predictions {where_clause}"
    total_count = cur.execute(count_query, params).fetchone()[0]

    select_query = f"""
        SELECT * FROM test_predictions {where_clause}
        ORDER BY raw_risk_score DESC
        LIMIT ? OFFSET ?
    """
    rows = cur.execute(select_query, params + [limit, offset]).fetchall()
    conn.close()

    results = []
    for r in rows:
        shap_drivers = json.loads(r["top_shap_drivers"]) if r["top_shap_drivers"] else None
        results.append(
            TransactionItem(
                transaction_id=r["transaction_id"],
                transaction_dt=r["transaction_dt"],
                transaction_amt=r["transaction_amt"],
                raw_risk_score=r["raw_risk_score"],
                calibrated_prob=r["calibrated_prob"],
                is_fraud=r["is_fraud"],
                flag_top_0_5pct=r["flag_top_0_5pct"],
                flag_top_1_0pct=r["flag_top_1_0pct"],
                flag_top_2_0pct=r["flag_top_2_0pct"],
                flag_top_5_0pct=r["flag_top_5_0pct"],
                top_shap_drivers=shap_drivers,
            )
        )

    return PaginatedTransactionsResponse(
        total_count=total_count,
        limit=limit,
        offset=offset,
        transactions=results,
    )


@app.get("/api/transactions/{transaction_id}", response_model=TransactionItem)
async def get_transaction_detail(transaction_id: int):
    """Retrieve detailed risk evaluation and local SHAP attributions for a single transaction."""
    conn = get_db_connection()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT * FROM test_predictions WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found in test predictions",
        )

    shap_drivers = json.loads(row["top_shap_drivers"]) if row["top_shap_drivers"] else None
    return TransactionItem(
        transaction_id=row["transaction_id"],
        transaction_dt=row["transaction_dt"],
        transaction_amt=row["transaction_amt"],
        raw_risk_score=row["raw_risk_score"],
        calibrated_prob=row["calibrated_prob"],
        is_fraud=row["is_fraud"],
        flag_top_0_5pct=row["flag_top_0_5pct"],
        flag_top_1_0pct=row["flag_top_1_0pct"],
        flag_top_2_0pct=row["flag_top_2_0pct"],
        flag_top_5_0pct=row["flag_top_5_0pct"],
        top_shap_drivers=shap_drivers,
    )


@app.post("/api/score", response_model=ScoreResponse)
async def score_transaction(request: ScoreRequest):
    """Real-time transaction risk scoring and operational review triage endpoint."""
    model = model_state.get("model")
    prep = model_state.get("preprocessor")
    calib = model_state.get("calibrator")

    if model is None or prep is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring engine model or preprocessor not loaded",
        )

    df_in = pd.DataFrame([request.features])
    try:
        X_proc = prep.transform(df_in)
        X_arr = np.ascontiguousarray(X_proc.values, dtype=np.float32)
        dmat = xgb.DMatrix(X_arr)
        raw_score = float(model.predict(dmat)[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature preprocessing or inference error: {str(e)}",
        )

    # Calibrate
    raw_clipped = float(np.clip(raw_score, 1e-6, 1.0 - 1e-6))
    if calib is not None:
        try:
            if hasattr(calib, "predict_proba"):
                calibrated = float(calib.predict_proba([[raw_clipped]])[0][1])
            else:
                calibrated = float(calib.predict([raw_clipped])[0])
        except Exception:
            calibrated = raw_clipped
    else:
        calibrated = raw_clipped

    thresholds = model_state["operating_thresholds"]
    flags = {
        "top_0.5pct_priority": raw_score >= thresholds.get("top_0.5pct", 0.868),
        "top_1.0pct_high_risk": raw_score >= thresholds.get("top_1.0pct", 0.613),
        "top_2.0pct_medium_risk": raw_score >= thresholds.get("top_2.0pct", 0.266),
        "top_5.0pct_elevated_risk": raw_score >= thresholds.get("top_5.0pct", 0.115),
    }

    if flags["top_0.5pct_priority"]:
        action = "Immediate Manual Review (Top 0.5% Priority Tier - 94.4% Precision)"
    elif flags["top_1.0pct_high_risk"]:
        action = "High Risk Review Tier (Top 1.0% Capacity)"
    elif flags["top_2.0pct_medium_risk"]:
        action = "Secondary Review Tier (Top 2.0% Capacity)"
    elif flags["top_5.0pct_elevated_risk"]:
        action = "Elevated Risk Monitoring (Top 5.0% Capacity)"
    else:
        action = "Standard Automated Processing"

    return ScoreResponse(
        raw_risk_score=round(raw_score, 6),
        calibrated_prob=round(calibrated, 6),
        flags=flags,
        recommended_action=action,
    )


@app.get("/api/experiments/rq2")
async def get_rq2_experiment():
    """Retrieve RQ2 temporal vs random split overstatement experiment results."""
    rq2_file = model_state["reports_dir"] / "m5_rq2_split_comparison.json"
    if not rq2_file.exists():
        raise HTTPException(status_code=404, detail="RQ2 results not found")
    with open(rq2_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/experiments/bootstrap")
async def get_bootstrap_experiment():
    """Retrieve paired bootstrap inference confidence intervals."""
    boot_file = model_state["reports_dir"] / "m5_paired_bootstrap.json"
    if not boot_file.exists():
        raise HTTPException(status_code=404, detail="Bootstrap results not found")
    with open(boot_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/experiments/backtest")
async def get_backtest_experiment():
    """Retrieve rolling-origin backtest records."""
    bt_file = model_state["reports_dir"] / "m5_rolling_backtest.json"
    if not bt_file.exists():
        raise HTTPException(status_code=404, detail="Backtest results not found")
    with open(bt_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/experiments/test-summary")
async def get_test_summary():
    """Retrieve final held-out test evaluation summary."""
    test_file = model_state["reports_dir"] / "test_evaluation_summary.json"
    if not test_file.exists():
        raise HTTPException(status_code=404, detail="Test evaluation summary not found")
    with open(test_file, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# DASHBOARD FRONTEND AT ROOT (/)
# =============================================================================
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the interactive HTML/JS risk review dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>FraudShield API Running</h1><p>Dashboard HTML not found.</p>")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
