"""The data contract: a single, versioned source of truth for valid data.

Everything (pipeline, model, API, tests) imports from here so they cannot
silently drift apart. Bump SCHEMA_VERSION when the shape changes.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

SCHEMA_VERSION = "1.1.0"
# 1.0.0 initial Lending Club subset
# 1.1.0 added fico_range_low/high; revol_util explicitly nullable

# ---- categorical domains: the allowed values, declared once ----------------
TERM_VALUES = ["36 months", "60 months"]
GRADE_VALUES = list("ABCDEFG")
HOME_VALUES = ["RENT", "MORTGAGE", "OWN", "OTHER", "NONE", "ANY"]
PURPOSE_VALUES = [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house", "other",
]
EMP_LENGTH_VALUES = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
    "6 years", "7 years", "8 years", "9 years", "10+ years",
]
# Lending Club loan_status strings -> which mean "defaulted"
DEFAULT_STATUSES = {"Charged Off", "Default", "Late (31-120 days)"}
PAID_STATUSES = {"Fully Paid"}

# ---- the RAW contract: what an ingested loan row must look like -------------
RAW_SCHEMA = DataFrameSchema(
    {
        "id": Column(pa.Int64, unique=True),
        "issue_d": Column(pa.DateTime, nullable=False),
        "loan_amnt": Column(float, Check.in_range(500, 40_000)),
        "term": Column(str, Check.isin(TERM_VALUES)),
        "int_rate": Column(float, Check.in_range(0, 35)),
        "installment": Column(float, Check.gt(0)),
        "grade": Column(str, Check.isin(GRADE_VALUES)),
        "emp_length": Column(str, Check.isin(EMP_LENGTH_VALUES), nullable=True),
        "home_ownership": Column(str, Check.isin(HOME_VALUES)),
        "annual_inc": Column(float, Check.in_range(0, 5_000_000)),
        "purpose": Column(str, Check.isin(PURPOSE_VALUES)),
        "dti": Column(float, Check.in_range(0, 100), nullable=True),
        "delinq_2yrs": Column(pa.Int64, Check.ge(0)),
        "open_acc": Column(pa.Int64, Check.ge(0)),
        "revol_util": Column(float, Check.in_range(0, 200), nullable=True),
        "fico_range_low": Column(pa.Int64, Check.in_range(300, 850)),
        "fico_range_high": Column(pa.Int64, Check.in_range(300, 850)),
        "loan_status": Column(str),
    },
    strict="filter",   # drop unexpected columns instead of failing (additive-safe)
    coerce=True,        # cast to the declared dtype where possible
    name=f"lending_club_raw_v{SCHEMA_VERSION}",
)

# ---- the MODELING contract: the columns the model expects ------------------
# Note: some of these (fico, installment_to_income, ...) don't exist in raw data;
# they're engineered later in features/transforms.py. This list is the contract
# between feature-engineering and the model.
NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "open_acc", "revol_util", "fico",
    "installment_to_income", "loan_to_income", "emp_length_years",
]
CATEGORICAL_FEATURES = ["term", "grade", "home_ownership", "purpose"]
TARGET = "default"


def coerce_to_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Validate + coerce a raw frame. Fails closed on any violation."""
    return RAW_SCHEMA.validate(df, lazy=True)


def derive_target(loan_status: pd.Series) -> pd.Series:
    """Map Lending Club loan_status -> binary default label.
    Loans that are neither clearly paid nor defaulted (e.g. 'Current') become
    NA so they can be dropped — you can't label a loan that isn't finished."""
    out = pd.Series(pd.NA, index=loan_status.index, dtype="Int64")
    out[loan_status.isin(DEFAULT_STATUSES)] = 1
    out[loan_status.isin(PAID_STATUSES)] = 0
    return out