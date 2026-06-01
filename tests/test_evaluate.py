"""Metric tests: sanity-check AUC/KS on easy cases."""
import numpy as np

from quant_risk.models.evaluate import evaluate


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(y, s)
    assert m["auc"] == 1.0 and m["gini"] == 1.0


def test_random_scores_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    s = rng.uniform(size=5000)
    assert abs(evaluate(y, s)["auc"] - 0.5) < 0.05