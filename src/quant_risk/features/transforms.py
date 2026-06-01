"""Feature engineering -- the GOLD layer.

Imported by BOTH training and the serving API, so a record scored online goes
through the EXACT same transforms as the training rows. That eliminates
train/serve skew -- the most common silent cause of production model failure.

Pure pandas, vectorized, stateless, no I/O -> trivially unit-testable and
reusable inside a Spark mapInPandas for the distributed path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_risk.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

_EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw (contract-valid) frame -> modeling frame with NUMERIC+CATEGORICAL features."""
    out = pd.DataFrame(index=df.index)

    for col in ["loan_amnt", "int_rate", "installment", "annual_inc",
                "dti", "delinq_2yrs", "open_acc"]:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    out["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce").fillna(0.0)
    out["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    out["emp_length_years"] = df["emp_length"].map(_EMP_MAP).fillna(0).astype(float)

    inc = out["annual_inc"].replace(0, np.nan)              # guard divide-by-zero
    out["installment_to_income"] = (out["installment"] * 12 / inc).fillna(0.0)
    out["loan_to_income"] = (out["loan_amnt"] / inc).fillna(0.0)

    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].astype("object").fillna("MISSING")

    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].fillna(0.0)   # final safety: no NaN to the model
    return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def features_from_record(record: dict) -> pd.DataFrame:
    """Single online record (already schema-checked by Pydantic) -> 1-row feature frame."""
    return build_features(pd.DataFrame([record]))