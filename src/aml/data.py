"""Load and validate data.

Two entry points:
- load_split("train_sample" | "val_sample" | "val_full" | "test_full" | "tail"):
  the feature files produced by notebooks/02_features.ipynb.
- load_raw_transactions(path): any raw transactions CSV, mapped to the standard
  column names via configs/schema.yaml (this is what makes the pipeline reusable
  on other datasets).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aml.config import DATA_DIR, load_schema

PROCESSED = DATA_DIR / "processed"
SPLITS = ["train_sample", "val_sample", "val_full", "test_full", "tail"]
META_COLS = ["row_id", "timestamp", "split", "label", "pattern", "weight"]
CATEGORICAL = ["payment_format", "payment_currency"]


class DataValidationError(ValueError):
    pass


def feature_list(processed_dir: Path = PROCESSED) -> list[str]:
    return json.loads((processed_dir / "feature_list.json").read_text())


def validate(df: pd.DataFrame, features: list[str], name: str = "data") -> None:
    """Fail loudly if the file is not what the models expect."""
    missing = [c for c in ["row_id", "timestamp", "label"] + features if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing columns {missing}")
    if not set(np.unique(df["label"])) <= {0, 1}:
        raise DataValidationError(f"{name}: label must be 0/1")
    nulls = df[features].isna().sum()
    if nulls.any():
        raise DataValidationError(f"{name}: nulls in {nulls[nulls > 0].to_dict()}")
    if df["row_id"].duplicated().any():
        raise DataValidationError(f"{name}: duplicate row_id")
    if not df["timestamp"].is_monotonic_increasing:
        raise DataValidationError(f"{name}: not sorted by time")
    if "weight" in df.columns and (df["weight"] <= 0).any():
        raise DataValidationError(f"{name}: non-positive weights")


def load_split(name: str, processed_dir: Path = PROCESSED, columns: list[str] | None = None) -> pd.DataFrame:
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; choose from {SPLITS}")
    features = feature_list(processed_dir)
    df = pd.read_parquet(processed_dir / f"{name}.parquet", columns=columns)
    if columns is None:
        validate(df, features, name)
    return df


def align_categories(frames: list[pd.DataFrame], reference: pd.DataFrame) -> list[pd.DataFrame]:
    """Give categorical columns the SAME category list as the training frame.
    (LightGBM maps categories by integer code, so mismatched lists = silent bugs.)
    Unseen categories become NaN, which the models treat as missing."""
    out = []
    for f in frames:
        f = f.copy()
        for c in CATEGORICAL:
            if c in f.columns:
                cats = reference[c].astype("category").cat.categories
                vals = f[c].astype(str)
                f[c] = pd.Categorical(vals.where(vals.isin(cats)), categories=cats)
        out.append(f)
    return out


def xyw(df: pd.DataFrame, features: list[str]):
    """Split a frame into model inputs X, labels y and sample weights w."""
    X = df[features]
    y = df["label"].to_numpy(dtype=np.int8)
    w = df["weight"].to_numpy(dtype=np.float64) if "weight" in df.columns else np.ones(len(df))
    return X, y, w


def load_raw_transactions(path: str | Path, schema: dict | None = None) -> pd.DataFrame:
    """Read any transactions CSV and rename its columns to the standard names."""
    schema = schema or load_schema()
    mapping = {v: k for k, v in schema["columns"].items() if v is not None}
    df = pd.read_csv(path, dtype={c: "string" for c in [schema["columns"]["from_account"],
                                                         schema["columns"]["to_account"]]})
    absent = [c for c in mapping if c not in df.columns]
    if absent:
        raise DataValidationError(f"columns {absent} from schema.yaml not found in {path}")
    df = df.rename(columns=mapping)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=schema.get("timestamp_format")).astype("datetime64[ns]")
    if "label" not in df.columns:
        df["label"] = 0
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)
