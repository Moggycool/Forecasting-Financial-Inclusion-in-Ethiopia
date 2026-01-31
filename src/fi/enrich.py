""" Functions to enrich the unified dataset with new records."""
from __future__ import annotations

import datetime as dt
import pandas as pd
import yaml


def load_new_records_yaml(path: str) -> pd.DataFrame:
    """Load enrichment records from YAML (top-level: records: [..])."""
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(
            "YAML must contain a top-level key `records` as a list")
    return pd.DataFrame.from_records(records)


def ensure_record_ids(df: pd.DataFrame, prefix: str = "ENR") -> pd.DataFrame:
    """Assign record_ids if missing."""
    df = df.copy()
    if "record_id" not in df.columns:
        df["record_id"] = None

    missing = df["record_id"].isna() | (
        df["record_id"].astype(str).str.strip() == "")
    if missing.any():
        today = dt.date.today().strftime("%Y%m%d")
        k = 1
        for i in df.index[missing]:
            df.loc[i, "record_id"] = f"{prefix}_{today}_{k:04d}"
            k += 1
    return df


def append_records(unified: pd.DataFrame, new_records: pd.DataFrame) -> pd.DataFrame:
    """Append new records, aligning to unified schema columns."""
    new_records = new_records.copy()
    for c in unified.columns:
        if c not in new_records.columns:
            new_records[c] = None
    return pd.concat([unified, new_records[unified.columns]], ignore_index=True)
