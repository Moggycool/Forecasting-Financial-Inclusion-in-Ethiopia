"""Data I/O utilities for the Financial Indicators project."""
from __future__ import annotations

import pandas as pd

REQUIRED_EVENTS_COLS = [
    "record_id",
    "category",
    "indicator",
    "indicator_code",
    "observation_date",
    "source_name",
    "confidence",
]

# In your raw CSV, impact_magnitude is numeric (pp) and impact_estimate is optional numeric.
REQUIRED_LINKS_COLS = [
    "record_id",
    "parent_id",
    "pillar",
    "related_indicator",
    "indicator_code",
    "impact_direction",
    "impact_magnitude",
    "lag_months",
    "evidence_basis",
    "confidence",
]


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    """Ensure that the DataFrame contains the required columns."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def coerce_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Coerce a column to datetime, returning a new DataFrame."""
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def coerce_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Coerce a column to numeric, returning a new DataFrame."""
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
