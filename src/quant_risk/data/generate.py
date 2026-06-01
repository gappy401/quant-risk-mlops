"""Synthetic, schema-faithful Lending Club loan book.

Real LC data (dropped into data/raw/ later) is the source of record; this
generator is the fallback that lets us (a) run with zero setup/auth, (b) scale
to arbitrary size, and (c) *manufacture drift on demand* to test monitoring.

The label is built from a KNOWN logit of the risk drivers, so there is genuine
signal to learn -- but model accuracy is explicitly not the goal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_risk.schema import (
    EMP_LENGTH_VALUES, GRADE_VALUES, HOME_VALUES, PURPOSE_VALUES, TERM_VALUES,
)


def generate_pandas(n: int, base_rate: float = 0.15, seed: int = 42, drift: float = 0.0) -> pd.DataFrame:
    """Generate `n` raw loans. `drift` in [0,1] shifts the covariate
    distributions (worse macro conditions) to simulate data drift."""
    rng = np.random.default_rng(seed)

    annual_inc = np.clip(rng.lognormal(11.0 - 0.15 * drift, 0.5, n), 8_000, 5_000_000).round(2)
    loan_amnt = np.clip(rng.lognormal(9.4 + 0.1 * drift, 0.55, n), 500, 40_000).round(2)
    dti = np.clip(rng.normal(18 + 6 * drift, 9, n), 0, 100).round(2)
    revol_util = np.clip(rng.normal(45 + 15 * drift, 22, n), 0, 200).round(1)
    int_rate = np.clip(rng.normal(13 + 3 * drift, 4.5, n), 5, 30).round(2)
    delinq_2yrs = rng.poisson(0.3 + 0.4 * drift, n).astype("int64")
    open_acc = np.clip(rng.poisson(11, n), 1, 60).astype("int64")
    fico_low = np.clip(rng.normal(695 - 15 * drift, 35, n), 660, 845).astype(int)
    fico_low = (fico_low // 5) * 5
    term = rng.choice(TERM_VALUES, n, p=[0.72, 0.28])
    grade = rng.choice(GRADE_VALUES, n, p=[0.16, 0.29, 0.27, 0.15, 0.08, 0.035, 0.015])
    home = rng.choice(HOME_VALUES[:4], n, p=[0.40, 0.49, 0.10, 0.01])
    purpose = rng.choice(PURPOSE_VALUES, n,
                         p=[0.50, 0.22, 0.06, 0.04, 0.015, 0.03, 0.02, 0.012, 0.008, 0.005, 0.09])
    emp_idx = np.clip(rng.geometric(0.18, n) - 1, 0, len(EMP_LENGTH_VALUES) - 1)
    emp_length = np.array(EMP_LENGTH_VALUES)[emp_idx].astype(object)
    emp_length[rng.uniform(size=n) < 0.05] = None   # realistic missingness

    installment = (loan_amnt * (int_rate / 1200) /
                   (1 - (1 + int_rate / 1200) ** -np.where(term == "36 months", 36, 60))).round(2)

    # known default-generating process (a logit of the risk drivers)
    z = (-2.7 + 0.06 * dti + 0.012 * revol_util + 0.45 * delinq_2yrs
         + 0.10 * int_rate - 1.5e-6 * annual_inc + 2.0e-5 * loan_amnt
         - 0.004 * fico_low + np.where(home == "RENT", 0.2, 0.0)
         + rng.normal(0, 0.4, n))                       # irreducible noise
    p = 1 / (1 + np.exp(-z))
    shift = np.log(base_rate / (1 - base_rate)) - np.log(p.mean() / (1 - p.mean()))
    p = 1 / (1 + np.exp(-(z + shift)))                  # nudge realized rate -> base_rate
    default = rng.uniform(size=n) < p
    loan_status = np.where(default, "Charged Off", "Fully Paid")

    return pd.DataFrame({
        "id": np.arange(n, dtype=np.int64),
        "issue_d": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), "D"),
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "installment": installment,
        "grade": grade, "emp_length": emp_length, "home_ownership": home,
        "annual_inc": annual_inc, "purpose": purpose, "dti": dti,
        "delinq_2yrs": delinq_2yrs, "open_acc": open_acc, "revol_util": revol_util,
        "fico_range_low": fico_low, "fico_range_high": fico_low + 4, "loan_status": loan_status,
    })


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--out", default="data/raw/loans.parquet")
    args = ap.parse_args()
    frame = generate_pandas(args.n, drift=args.drift)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"wrote {len(frame):,} rows -> {args.out} "
          f"(default rate={(frame.loan_status=='Charged Off').mean():.3f})")