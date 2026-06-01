"""Ingestion: real-data-first, synthetic-fallback, contract-validated.

Order of preference:
  1. data/raw/*.csv     -> a real Lending Club export you downloaded
  2. data/raw/*.parquet -> a cached / synthetic book
  3. generate on the fly (CI, scale tests, drift simulation)

Whatever the source, the frame is coerced through RAW_SCHEMA before it's
allowed downstream. Bad data fails here, loudly -- not three steps later.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from quant_risk import schema
from quant_risk.data.generate import generate_pandas

_KEEP = list(schema.RAW_SCHEMA.columns.keys())


def _read_real(raw_dir: Path) -> pd.DataFrame | None:
    csvs = glob.glob(str(raw_dir / "*.csv"))
    if csvs:
        df = pd.concat((pd.read_csv(f, skiprows=1, low_memory=False) for f in csvs), ignore_index=True)
        if "int_rate" in df and df["int_rate"].dtype == object:
            df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)
        if "revol_util" in df and df["revol_util"].dtype == object:
            df["revol_util"] = df["revol_util"].str.rstrip("%").astype(float)
        if "issue_d" in df:
            df["issue_d"] = pd.to_datetime(df["issue_d"], format="mixed", errors="coerce")
        if "id" not in df:
            df["id"] = range(len(df))
        return df
    parquets = glob.glob(str(raw_dir / "*.parquet"))
    if parquets:
        return pd.concat((pd.read_parquet(f) for f in parquets), ignore_index=True)
    return None


def load_raw(raw_dir="data/raw", n_synthetic=50_000, drift=0.0, seed=42) -> pd.DataFrame:
    """Return a contract-valid raw frame from the best available source."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = _read_real(raw_dir)
    source = "real"
    if df is None:
        df = generate_pandas(n_synthetic, seed=seed, drift=drift)
        source = f"synthetic(n={n_synthetic}, drift={drift})"

    df = df[[c for c in _KEEP if c in df.columns]].copy()   # keep only contract columns
    df = schema.coerce_to_contract(df)                      # the gate: validate + coerce
    df.attrs["source"] = source
    return df


def to_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the binary target and drop rows that aren't finished (no outcome)."""
    df = df.copy()
    df[schema.TARGET] = schema.derive_target(df["loan_status"])
    return df.dropna(subset=[schema.TARGET]).astype({schema.TARGET: "int64"})