"""Credit-risk evaluation metrics: AUC, Gini, KS.

Standard and minimal -- accuracy tuning is out of scope. These exist mainly so
CI has an objective *quality gate* to decide whether to promote a model.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def ks_statistic(y_true, y_score) -> float:
    """KS = max gap between the cumulative good/bad score distributions."""
    order = np.argsort(y_score)
    y = np.asarray(y_true)[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def evaluate(y_true, y_score) -> dict[str, float]:
    auc = float(roc_auc_score(y_true, y_score))
    return {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4),
            "ks": round(ks_statistic(y_true, y_score), 4)}