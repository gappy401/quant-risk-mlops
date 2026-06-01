"""Data-contract tests: accept valid data, reject violations, tolerate additive changes."""
import pandas as pd
import pandera.errors
import pytest

from quant_risk import schema
from quant_risk.data.generate import generate_pandas


def test_generated_data_satisfies_contract():
    df = generate_pandas(2_000, seed=7)
    validated = schema.coerce_to_contract(df)
    assert len(validated) == len(df)
    assert set(schema.RAW_SCHEMA.columns).issubset(validated.columns)


def test_out_of_range_value_is_rejected():
    df = generate_pandas(500, seed=7)
    df.loc[0, "int_rate"] = 99.0          # contract allows 0..35
    with pytest.raises(pandera.errors.SchemaErrors):
        schema.coerce_to_contract(df)


def test_additive_extra_column_is_dropped_not_fatal():
    df = generate_pandas(500, seed=7)
    df["brand_new_lc_field"] = 1          # upstream added a column
    validated = schema.coerce_to_contract(df)
    assert "brand_new_lc_field" not in validated.columns


def test_target_derivation():
    s = pd.Series(["Charged Off", "Fully Paid", "Current"])
    out = schema.derive_target(s)
    assert out.tolist()[:2] == [1, 0]
    assert pd.isna(out.iloc[2])