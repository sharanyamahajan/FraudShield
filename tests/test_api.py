"""Test suite for the FraudShield FastAPI application."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from fraudshield.api.main import app

DB_PATH = Path("reports/fraudshield.db")
MODEL_PATH = Path("models/xgb_set_b.json")
PREP_PATH = Path("models/preprocessor_b.pkl")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoints(client):
    r1 = client.get("/health")
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["status"] in ["healthy", "degraded"]
    assert "version" in data1

    r2 = client.get("/api/health")
    assert r2.status_code == 200
    assert r2.json() == data1


def test_metadata_endpoint(client):
    if not DB_PATH.exists():
        pytest.skip("reports/fraudshield.db not found (fresh clone without database) - run pipeline first")
    r = client.get("/api/metadata")
    assert r.status_code == 200
    data = r.json()
    assert "model_id" in data
    assert "val_pr_auc" in data
    assert "test_pr_auc" in data
    assert data["test_pr_auc"] is not None


def test_operating_points_endpoint(client):
    if not DB_PATH.exists():
        pytest.skip("reports/fraudshield.db not found (fresh clone without database) - run pipeline first")
    r = client.get("/api/operating-points")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 4
    tier_names = [d["budget_name"] for d in data]
    assert "top_0.5pct" in tier_names
    assert "top_1.0pct" in tier_names
    for item in data:
        assert 0.0 <= item["score_threshold"] <= 1.0
        assert 0.0 <= item["val_recall"] <= 1.0


def test_features_importance_endpoint(client):
    if not DB_PATH.exists():
        pytest.skip("reports/fraudshield.db not found (fresh clone without database) - run pipeline first")
    r = client.get("/api/features/importance?limit=10")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 10
    assert data[0]["rank"] == 1
    assert "feature_name" in data[0]
    assert "mean_abs_shap" in data[0]


def test_transactions_pagination_and_filter(client):
    if not DB_PATH.exists():
        pytest.skip("reports/fraudshield.db not found (fresh clone without database) - run pipeline first")
    # 1. Default pagination
    r = client.get("/api/transactions?limit=10&offset=0")
    assert r.status_code == 200
    data = r.json()
    assert data["total_count"] == 88581
    assert len(data["transactions"]) == 10

    # 2. Tier filtering
    r_tier = client.get("/api/transactions?tier=top_0.5pct&limit=5")
    assert r_tier.status_code == 200
    data_tier = r_tier.json()
    assert data_tier["total_count"] > 0
    for tx in data_tier["transactions"]:
        assert tx["flag_top_0_5pct"] == 1


def test_transaction_detail_and_not_found(client):
    if not DB_PATH.exists():
        pytest.skip("reports/fraudshield.db not found (fresh clone without database) - run pipeline first")
    # Get first transaction ID
    r_list = client.get("/api/transactions?limit=1")
    tx_id = r_list.json()["transactions"][0]["transaction_id"]

    # Valid ID
    r = client.get(f"/api/transactions/{tx_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["transaction_id"] == tx_id
    assert "calibrated_prob" in data

    # Invalid ID
    r_inv = client.get("/api/transactions/999999999")
    assert r_inv.status_code == 404


def test_score_endpoint(client):
    if not MODEL_PATH.exists() or not PREP_PATH.exists():
        pytest.skip("Model or preprocessor artifact not found in models/ - run pipeline first")
    # Construct a minimal feature payload using default features
    payload = {
        "features": {
            "TransactionAmt": 150.0,
            "ProductCD": "W",
            "card1": 10000,
            "card2": 500.0,
            "card3": 150.0,
            "card4": "visa",
            "card5": 226.0,
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "addr1": 300.0,
            "addr2": 87.0,
        }
    }
    r = client.post("/api/score", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "raw_risk_score" in data
    assert "calibrated_prob" in data
    assert "flags" in data
    assert "recommended_action" in data


def test_score_endpoint_missing_fields(client):
    if not MODEL_PATH.exists() or not PREP_PATH.exists():
        pytest.skip("Model or preprocessor artifact not found in models/ - run pipeline first")
    # Payload with almost all columns missing (only TransactionAmt and card1 provided)
    payload = {
        "features": {
            "TransactionAmt": 99.99,
            "card1": 5432,
        }
    }
    r = client.post("/api/score", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "raw_risk_score" in data
    assert "calibrated_prob" in data
    assert 0.0 <= data["calibrated_prob"] <= 1.0


def test_score_endpoint_unseen_categories(client):
    if not MODEL_PATH.exists() or not PREP_PATH.exists():
        pytest.skip("Model or preprocessor artifact not found in models/ - run pipeline first")
    # Payload with novel/unseen categorical values
    payload = {
        "features": {
            "TransactionAmt": 250.0,
            "ProductCD": "TOTALLY_UNSEEN_PRODUCT",
            "card4": "novel_crypto_network",
            "P_emaildomain": "unseen_domain_99.biz",
            "addr1": 999.0,
        }
    }
    r = client.post("/api/score", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "raw_risk_score" in data
    assert "calibrated_prob" in data
    assert 0.0 <= data["calibrated_prob"] <= 1.0


def test_experiments_endpoints(client):
    r_rq2 = client.get("/api/experiments/rq2")
    assert r_rq2.status_code == 200
    data_rq2 = r_rq2.json()
    assert "temporal_split" in data_rq2
    assert "random_split_optimistic_leaky" in data_rq2

    r_boot = client.get("/api/experiments/bootstrap")
    assert r_boot.status_code == 200
    assert "c1_minus_b" in r_boot.json()

    r_bt = client.get("/api/experiments/backtest")
    assert r_bt.status_code == 200
    assert "folds" in r_bt.json()

    r_test = client.get("/api/experiments/test-summary")
    assert r_test.status_code == 200
    assert "primary_model_set_c1" in r_test.json()


def test_dashboard_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "FraudShield" in r.text
    assert "text/html" in r.headers["content-type"]
