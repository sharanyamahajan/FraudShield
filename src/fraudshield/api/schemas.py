"""Pydantic schemas for the FraudShield API."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
    version: str


class ModelMetadataResponse(BaseModel):
    model_id: str
    model_name: str
    feature_set: str
    best_iteration: Optional[int]
    val_pr_auc: Optional[float]
    test_pr_auc: Optional[float]
    test_roc_auc: Optional[float]
    brier_uncalibrated: Optional[float]
    brier_calibrated: Optional[float]
    calibrator_type: Optional[str]
    git_commit_hash: Optional[str]
    evaluated_at: Optional[str]


class OperatingPointItem(BaseModel):
    budget_name: str
    budget_ratio: float
    score_threshold: float
    val_recall: float
    val_precision: float
    test_recall: Optional[float]
    test_precision: Optional[float]
    test_review_count: Optional[int]
    test_fraud_captured: Optional[int]


class FeatureImportanceItem(BaseModel):
    rank: int
    feature_name: str
    feature_family: str
    mean_abs_shap: float
    is_anonymised: bool
    interpretation_note: str


class TransactionItem(BaseModel):
    transaction_id: int
    transaction_dt: int
    transaction_amt: float
    raw_risk_score: float
    calibrated_prob: float
    is_fraud: int
    flag_top_0_5pct: int
    flag_top_1_0pct: int
    flag_top_2_0pct: int
    flag_top_5_0pct: int
    top_shap_drivers: Optional[Dict[str, Any]] = None


class PaginatedTransactionsResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    transactions: List[TransactionItem]


class ScoreRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Dictionary of raw transaction features")


class ScoreResponse(BaseModel):
    raw_risk_score: float
    calibrated_prob: float
    flags: Dict[str, bool]
    recommended_action: str
