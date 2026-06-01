"""Training pipeline.

Deliberately a plain, untuned model -- the value is the production pattern
around it: reproducible (seeded + data-hashed), versioned (MLflow registry),
gated (won't register below the AUC floor), and it persists a drift *reference
baseline* so monitoring has something to compare against later.
"""
from __future__ import annotations

import hashlib
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from quant_risk.config import load_config
from quant_risk.data.load import load_raw, to_labeled
from quant_risk.features.transforms import build_features
from quant_risk.models.evaluate import evaluate
from quant_risk.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SCHEMA_VERSION, TARGET


def _hash_frame(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:12]


def build_pipeline(seed: int, max_iter: int) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=max_iter, random_state=seed))])


def train(n_synthetic: int | None = None) -> dict:
    cfg = load_config()
    cfg.paths.make()
    n = n_synthetic or cfg.n_synthetic

    raw = to_labeled(load_raw(n_synthetic=n, seed=cfg.seed))
    X, y = build_features(raw), raw[TARGET]
    data_hash = _hash_frame(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.model.test_size, random_state=cfg.seed, stratify=y)
    pipe = build_pipeline(cfg.seed, cfg.model.max_iter).fit(X_tr, y_tr)
    metrics = evaluate(y_te, pipe.predict_proba(X_te)[:, 1])
    passed = metrics["auc"] >= cfg.model.auc_gate

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({"model": "logreg", "schema_version": SCHEMA_VERSION,
                           "data_hash": data_hash, "n_rows": len(X), "seed": cfg.seed,
                           "source": raw.attrs.get("source", "unknown")})
        mlflow.log_metrics(metrics)
        sig = infer_signature(X_te, pipe.predict_proba(X_te)[:, 1])
        mlflow.sklearn.log_model(pipe, name="model", signature=sig,
                                 registered_model_name=cfg.model_name if passed else None)
        ref_path = cfg.paths.gold / "reference.parquet"
        X_tr.assign(**{TARGET: y_tr.values}).to_parquet(ref_path, index=False)
        mlflow.log_artifact(str(ref_path))
        json.dump(metrics, open("metrics.json", "w"))
        print(f"run={run.info.run_id} metrics={metrics} "
              f"gate(auc>={cfg.model.auc_gate})={'PASS' if passed else 'FAIL'}")
    return {"run_id": run.info.run_id, "metrics": metrics, "passed": passed}


if __name__ == "__main__":
    train()