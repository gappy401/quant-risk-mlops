"""Drift detection.

The metrics credit-risk teams actually use on scorecards:
  * PSI (Population Stability Index) -- feature and score drift.
        PSI < 0.10            stable
        0.10 <= PSI < 0.25    moderate shift -> WATCH
        PSI >= 0.25           material shift  -> ALERT
  * KS (two-sample Kolmogorov-Smirnov) -- distributional shift for numerics.

detect_drift() returns a structured, JSON-serializable report consumed by the
monitoring job and alerts.py. Evidently is OPTIONAL (build_evidently_report);
the alerting logic does not depend on it, so this runs with zero heavy deps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WATCH = 0.10
PSI_ALERT = 0.25


def _psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """PSI between a reference and current distribution.
    Numeric: quantile bins fixed from the reference (edges don't move).
    Categorical: per-category shares."""
    eps = 1e-6
    if pd.api.types.is_numeric_dtype(ref):
        edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        ref_pct = np.histogram(ref.dropna(), edges)[0] / max(len(ref.dropna()), 1)
        cur_pct = np.histogram(cur.dropna(), edges)[0] / max(len(cur.dropna()), 1)
    else:
        cats = pd.Index(ref.dropna().unique()).union(cur.dropna().unique())
        ref_pct = ref.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
        cur_pct = cur.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _status(psi: float) -> str:
    if psi >= PSI_ALERT:
        return "ALERT"
    if psi >= PSI_WATCH:
        return "WATCH"
    return "OK"


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_stat: float | None
    ks_pvalue: float | None
    status: str


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    features: list[FeatureDrift] = field(default_factory=list)
    prediction_psi: float | None = None
    target_rate_reference: float | None = None
    target_rate_current: float | None = None

    @property
    def drift_detected(self) -> bool:
        return any(f.status == "ALERT" for f in self.features) or (
            self.prediction_psi is not None and self.prediction_psi >= PSI_ALERT)

    @property
    def alerting_features(self) -> list[str]:
        return [f.feature for f in self.features if f.status == "ALERT"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["drift_detected"] = self.drift_detected
        d["alerting_features"] = self.alerting_features
        return d


def detect_drift(reference, current, features, ref_scores=None, cur_scores=None,
                 ref_target=None, cur_target=None) -> DriftReport:
    report = DriftReport(n_reference=len(reference), n_current=len(current))
    for feat in features:
        if feat not in reference or feat not in current:
            continue
        psi = _psi(reference[feat], current[feat])
        ks_stat = ks_p = None
        if pd.api.types.is_numeric_dtype(reference[feat]):
            res = ks_2samp(reference[feat].dropna(), current[feat].dropna())
            ks_stat, ks_p = float(res.statistic), float(res.pvalue)
        report.features.append(FeatureDrift(feat, round(psi, 4), ks_stat, ks_p, _status(psi)))
    if ref_scores is not None and cur_scores is not None:
        report.prediction_psi = round(_psi(ref_scores, cur_scores), 4)
    if ref_target is not None and cur_target is not None:
        report.target_rate_reference = round(float(ref_target.mean()), 4)
        report.target_rate_current = round(float(cur_target.mean()), 4)
    return report


def build_evidently_report(reference, current, out_html):
    """Optional richer HTML report. No-op (returns None) if Evidently absent."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return None
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=reference, current_data=current).save_html(out_html)
    return out_html