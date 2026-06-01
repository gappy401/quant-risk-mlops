"""API tests. We inject a small trained pipeline into the holder so the
endpoints are tested without depending on a populated MLflow registry."""
from fastapi.testclient import TestClient

from quant_risk.data.generate import generate_pandas
from quant_risk.data.load import to_labeled
from quant_risk.features.transforms import build_features
from quant_risk.models.train import build_pipeline
from quant_risk.schema import TARGET
from quant_risk.serving import api

VALID = {
    "loan_amnt": 12000, "term": "36 months", "int_rate": 13.5, "installment": 407.0,
    "grade": "C", "emp_length": "5 years", "home_ownership": "RENT", "annual_inc": 65000,
    "purpose": "debt_consolidation", "dti": 18.2, "delinq_2yrs": 0, "open_acc": 9,
    "revol_util": 42.0, "fico_range_low": 690, "fico_range_high": 694,
}


def _trained_pipeline():
    raw = to_labeled(generate_pandas(3_000, seed=11))
    return build_pipeline(seed=11, max_iter=1000).fit(build_features(raw), raw[TARGET])


def test_health():
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"


def test_score_with_injected_model():
    with TestClient(api.app) as c:
        api.holder.model = _trained_pipeline()
        api.holder.version = "test"
        r = c.post("/score", json=VALID)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["pd"] <= 1.0
        assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH"}


def test_bad_payload_rejected():
    with TestClient(api.app) as c:
        assert c.post("/score", json={**VALID, "int_rate": 999}).status_code == 422
        assert c.post("/score", json={**VALID, "surprise_field": 1}).status_code == 422