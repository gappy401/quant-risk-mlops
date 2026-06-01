"""Feature tests -- the most important guarantee is train/serve parity."""
import pandas as pd

from quant_risk.data.generate import generate_pandas
from quant_risk.features.transforms import build_features, features_from_record
from quant_risk.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def test_feature_columns_and_no_nans():
    feats = build_features(generate_pandas(1_000, seed=3))
    assert list(feats.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert feats[NUMERIC_FEATURES].isna().sum().sum() == 0


def test_engineered_ratios():
    df = generate_pandas(200, seed=3)
    feats = build_features(df)
    expected = (df["loan_amnt"] / df["annual_inc"].replace(0, float("nan"))).fillna(0.0)
    assert (feats["loan_to_income"] - expected).abs().max() < 1e-9


def test_train_serve_parity():
    """One record scored online must equal the same row in a batch transform."""
    df = generate_pandas(50, seed=3)
    batch = build_features(df).reset_index(drop=True)
    record = df.iloc[0].to_dict()
    single = features_from_record(record).reset_index(drop=True)
    pd.testing.assert_frame_equal(single[batch.columns], batch.iloc[[0]].reset_index(drop=True))