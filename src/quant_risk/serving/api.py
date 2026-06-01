"""FastAPI scoring service: loads the registered model, scores single + batch."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from quant_risk.config import load_config
from quant_risk.features.transforms import build_features
from quant_risk.schema import SCHEMA_VERSION
from quant_risk.serving.schema import BatchScoreRequest, LoanApplication, ScoreResponse

log = logging.getLogger("quant_risk.api")
_METRICS = {"score_requests_total": 0, "score_errors_total": 0}


def _band(p: float) -> str:
    return "LOW" if p < 0.10 else "MEDIUM" if p < 0.30 else "HIGH"


class ModelHolder:
    def __init__(self):
        self.model = None
        self.version = "unloaded"

    def load(self):
        cfg = load_config()
        import mlflow
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        uri = os.getenv("MODEL_URI", f"models:/{cfg.model_name}/latest")
        try:
            self.model = mlflow.sklearn.load_model(uri)
            self.version = uri
        except Exception as e:  # noqa: BLE001
            log.error("could not load model from %s: %s", uri, e)
            self.model, self.version = None, "unloaded"


holder = ModelHolder()


@asynccontextmanager
async def lifespan(_: FastAPI):
    holder.load()
    yield


app = FastAPI(title="Credit PD Scoring", version=SCHEMA_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if holder.model is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "model_version": holder.version}


def _score(records: list[dict]) -> list[float]:
    if holder.model is None:
        raise HTTPException(503, "model not loaded")
    feats = build_features(pd.DataFrame(records))
    return holder.model.predict_proba(feats)[:, 1].tolist()


@app.post("/score", response_model=ScoreResponse)
def score(app_in: LoanApplication):
    _METRICS["score_requests_total"] += 1
    try:
        p = _score([app_in.to_record()])[0]
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        _METRICS["score_errors_total"] += 1
        raise HTTPException(500, f"scoring failed: {e}") from e
    return ScoreResponse(pd=round(p, 6), risk_band=_band(p),
                         model_version=holder.version, schema_version=SCHEMA_VERSION)


@app.post("/score/batch")
def score_batch(req: BatchScoreRequest):
    _METRICS["score_requests_total"] += len(req.applications)
    scores = _score([a.to_record() for a in req.applications])
    return {"scores": [{"pd": round(s, 6), "risk_band": _band(s)} for s in scores],
            "model_version": holder.version, "schema_version": SCHEMA_VERSION}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [f"{k} {v}" for k, v in _METRICS.items()]
    lines.append(f'model_loaded{{version="{holder.version}"}} {int(holder.model is not None)}')
    return "\n".join(lines) + "\n"