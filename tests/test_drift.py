"""Drift tests: PSI ~0 for identical data, escalates to an alert under drift."""
from quant_risk.data.generate import generate_pandas
from quant_risk.features.transforms import build_features
from quant_risk.monitoring.drift import PSI_ALERT, detect_drift
from quant_risk.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_no_drift_when_same_distribution():
    a = build_features(generate_pandas(8_000, seed=1))
    b = build_features(generate_pandas(8_000, seed=2))   # same DGP, different sample
    rpt = detect_drift(a, b, FEATS)
    assert not rpt.drift_detected
    assert max(f.psi for f in rpt.features) < PSI_ALERT


def test_alert_under_severe_drift():
    a = build_features(generate_pandas(8_000, seed=1, drift=0.0))
    b = build_features(generate_pandas(8_000, seed=2, drift=1.5))
    rpt = detect_drift(a, b, FEATS)
    assert rpt.drift_detected
    assert len(rpt.alerting_features) > 0