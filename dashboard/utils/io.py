""" Utility functions for I/O operations in the dashboard. """
from __future__ import annotations

import pandas as pd
from pathlib import Path


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file from the given path."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_year_col(df: pd.DataFrame) -> str | None:
    """Heuristically detect the year column in the dataframe."""
    candidates = ["year", "Year", "YYYY", "survey_year", "obs_year", "time_year", "date_year"]
    col = _pick_first_existing(df, candidates)
    if col:
        return col

    for c in df.columns:
        if "year" in c.lower():
            return c
    return None


def _detect_indicator_col(df: pd.DataFrame) -> str | None:
    """Heuristically detect the indicator code column in the dataframe."""
    candidates = [
        "indicator_code", "Indicator Code", "indicator", "indicator_id", "series_code",
        "kpi_code", "code", "indicator_name_code"
    ]
    col = _pick_first_existing(df, candidates)
    if col:
        return col

    for c in df.columns:
        cl = c.lower()
        if "indicator" in cl and ("code" in cl or "id" in cl):
            return c
    return None


def _detect_value_col(df: pd.DataFrame) -> str | None:
    """Heuristically detect the value column in the dataframe."""
    candidates = [
        "value", "Value", "obs_value", "observation_value", "indicator_value", "val",
        "numeric_value", "amount", "count", "rate", "percent", "pct", "pp"
    ]
    col = _pick_first_existing(df, candidates)
    if col:
        return col

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    bad = {"year", "id", "index"}
    for c in numeric_cols:
        if c.lower() not in bad and "id" not in c.lower() and "code" not in c.lower():
            return c
    return None


def load_forecast_table(repo_root: Path) -> pd.DataFrame:
    """Load the forecast table from Task 4 outputs."""
    path = repo_root / "outputs" / "task_4" / "forecast_table_task4.csv"
    df = _read_csv(path)

    required = [
        "indicator_code", "indicator_label", "scenario", "year",
        "trend_pred_pp", "event_effect_pp", "pred_pp", "lo_pp", "hi_pp", "se_pp"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"forecast_table_task4.csv missing columns: {missing}")

    df["year"] = df["year"].astype(int)
    df["scenario"] = df["scenario"].astype(str)

    return df.sort_values(["indicator_code", "scenario", "year"]).reset_index(drop=True)


def load_event_effects(repo_root: Path) -> pd.DataFrame:
    """Load the event effects tidy table from Task 3 outputs."""
    path = repo_root / "outputs" / "task_3" / "event_effects_tidy.csv"
    df = _read_csv(path)
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    return df


def load_top_contributors(repo_root: Path) -> pd.DataFrame:
    """Load the top event contributors for 2027 from Task 4 outputs."""
    path = repo_root / "outputs" / "task_4" / "top_event_contributors_2027.csv"
    df = _read_csv(path)

    required = [
        "indicator_code", "indicator_label", "contributors_from_indicator",
        "event_record_id", "event_name", "effect_pp"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"top_event_contributors_2027.csv missing columns: {missing}")

    return df.sort_values(["indicator_code", "effect_pp"], ascending=[True, False]).reset_index(drop=True)

def load_enriched_observations(repo_root: Path) -> pd.DataFrame:
    """
    Loads unified enriched dataset and returns standardized long-format observations with:
      year (int), indicator_code (str), value (float)
    plus extra columns for filtering when available:
      record_type, value_type, pillar, unit, gender, location, region, source_type, confidence, etc.
    """
    path = repo_root / "data" / "processed" / "ethiopia_fi_unified_data__enriched.csv"
    raw = _read_csv(path)

    required = ["record_type", "indicator_code", "value_numeric"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"enriched csv missing required columns: {missing}")

    # --- Normalize key string columns early (pre-filter) ---
    # record_type normalization is crucial (avoids dropping rows due to case/whitespace)
    raw["record_type"] = raw["record_type"].astype(str).str.strip().str.lower()

    # Normalize indicator_code to avoid trailing spaces / mixed case issues
    raw["indicator_code"] = raw["indicator_code"].astype(str).str.strip()

    # Normalize slicer fields if present (so UI selections match reliably)
    for c in ["gender", "location", "region", "unit", "value_type", "pillar", "indicator"]:
        if c in raw.columns:
            raw[c] = raw[c].astype(str).str.strip()

    # --- Derive year ---
    if "fiscal_year" in raw.columns:
        # Handles numeric years; FY strings will become NaN (kept out unless observation_date exists)
        yr = pd.to_numeric(raw["fiscal_year"], errors="coerce")
    else:
        yr = pd.Series([pd.NA] * len(raw), index=raw.index)

    if ("observation_date" in raw.columns) and yr.isna().all():
        # Fallback if fiscal_year is entirely non-numeric / missing
        yr = pd.to_datetime(raw["observation_date"], errors="coerce").dt.year
    elif "observation_date" in raw.columns:
        # Fill missing year values from observation_date where possible
        yr2 = pd.to_datetime(raw["observation_date"], errors="coerce").dt.year
        yr = yr.fillna(yr2)

    raw["year"] = yr
    raw["value"] = pd.to_numeric(raw["value_numeric"], errors="coerce")

    # --- Keep observations only ---
    df = raw[raw["record_type"].eq("observation")].copy()

    # Keep filterable metadata if present
    keep_cols = [
        "year", "indicator_code", "value",
        "record_type", "value_type",
        "indicator", "pillar", "unit", "gender", "location", "region",
        "source_name", "source_type", "confidence"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df = df.dropna(subset=["year", "indicator_code", "value"]).copy()

    # Standard types
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    df["indicator_code"] = df["indicator_code"].astype(str).str.strip()

    # Normalize commonly used slicers to stable forms
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    if "location" in df.columns:
        df["location"] = df["location"].astype(str).str.strip().str.lower()
    if "unit" in df.columns:
        df["unit"] = df["unit"].astype(str).str.strip()
    if "value_type" in df.columns:
        df["value_type"] = df["value_type"].astype(str).str.strip().str.lower()

    return df.sort_values(["indicator_code", "year"]).reset_index(drop=True)