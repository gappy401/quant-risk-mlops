"""Batch prediction (offline / scheduled scoring).

The second serving mode: the FastAPI app scores one request on demand; this
scores a whole batch on a schedule -- the "nightly prediction system". Same
model from the registry, same shared build_features, so batch and online
scoring can't diverge either.

Scoring is pandas here (right for the current scale). For a huge population,
the score step becomes a Spark job applying the model across partitions -- the
features come from the same build_features via mapInPandas.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from quant_risk.config import load_config
from quant_risk.data.load import load_raw
from quant_risk.features.transforms import build_features


def _band(p: float) -> str:
    return "LOW" if p < 0.10 else "MEDIUM" if p < 0.30 else "HIGH"


def batch_predict(n: int = 20_000, seed: int = 7, raw_dir: str = "data/raw",
                  out: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    model = mlflow.sklearn.load_model(os.getenv("MODEL_URI", f"models:/{cfg.model_name}/latest"))

    # in prod: the day's new applications (read from S3); here: synthetic fallback
    raw = load_raw(raw_dir=raw_dir, n_synthetic=n, seed=seed)
    scores = model.predict_proba(build_features(raw))[:, 1]

    ts = pd.Timestamp.now(tz="UTC")
    results = pd.DataFrame({
        "id": raw["id"].to_numpy(),
        "scored_at": ts,
        "pd": scores.round(6),
        "risk_band": [_band(p) for p in scores],
    })
    out = out or str(cfg.paths.gold / "predictions" / f"scores_{ts:%Y%m%d}.parquet")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out, index=False)
    print(f"scored {len(results):,} rows -> {out} | bands: {results['risk_band'].value_counts().to_dict()}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    batch_predict(n=args.n, out=args.out)