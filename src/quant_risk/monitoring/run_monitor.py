"""Drift-monitoring job entrypoint.

Loads the training-time reference baseline (written by models/train.py) and a
current batch, computes drift, optionally writes an Evidently HTML report, and
dispatches alerts. This is what Airflow runs on a schedule and what CI can run
as a smoke check.
"""
from __future__ import annotations

import json

import pandas as pd

from quant_risk.config import load_config
from quant_risk.data.load import load_raw, to_labeled
from quant_risk.features.transforms import build_features
from quant_risk.monitoring.alerts import dispatch
from quant_risk.monitoring.drift import build_evidently_report, detect_drift
from quant_risk.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def run(current_drift: float = 0.0, n_current: int = 20_000) -> dict:
    cfg = load_config()
    ref_path = cfg.paths.gold / "reference.parquet"
    if not ref_path.exists():
        raise FileNotFoundError(f"no reference baseline at {ref_path}; train a model first")

    reference = pd.read_parquet(ref_path)
    feat_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    cur_raw = to_labeled(load_raw(n_synthetic=n_current, seed=999, drift=current_drift))
    current = build_features(cur_raw)

    report = detect_drift(
        reference[feat_cols], current[feat_cols], feat_cols,
        ref_target=reference[TARGET] if TARGET in reference else None,
        cur_target=cur_raw[TARGET])

    out_dir = cfg.paths.gold / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "drift_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    build_evidently_report(reference[feat_cols], current[feat_cols], str(out_dir / "drift.html"))

    sent = dispatch(report)
    print(json.dumps({"drift_detected": report.drift_detected,
                      "alerting_features": report.alerting_features, "dispatched": sent}, indent=2))
    return report.to_dict()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=20_000)
    args = ap.parse_args()
    run(current_drift=args.drift, n_current=args.n)