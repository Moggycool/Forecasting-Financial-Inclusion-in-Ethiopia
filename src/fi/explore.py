""" Exploratory functions for dataframes in the FI schema."""
from __future__ import annotations

import pandas as pd


def counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Get value counts for a given column, treating NaNs as '(empty)'."""
    return (
        df[col]
        .fillna("")
        .astype(str)
        .replace({"": "(empty)"})
        .value_counts(dropna=False)
        .rename_axis(col)
        .reset_index(name="n")
    )


def temporal_range(df: pd.DataFrame, date_col: str = "observation_date") -> dict:
    """Get temporal range information for a given date column."""
    s = pd.to_datetime(df[date_col], errors="coerce")
    return {
        "min_date": None if s.min() is pd.NaT else str(s.min().date()),
        "max_date": None if s.max() is pd.NaT else str(s.max().date()),
        "n_parsed": int(s.notna().sum()),
        "n_total": int(len(s)),
    }


def indicator_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Get coverage information for indicators in the dataframe."""
    obs = df[df["record_type"].astype(
        str).str.lower().eq("observation")].copy()
    obs["observation_date"] = pd.to_datetime(
        obs["observation_date"], errors="coerce")
    g = obs.groupby(["indicator_code", "indicator"], dropna=False)
    out = g.agg(
        n_obs=("value_numeric", "count"),
        min_date=("observation_date", "min"),
        max_date=("observation_date", "max"),
    ).reset_index()
    out["min_date"] = pd.to_datetime(
        out["min_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["max_date"] = pd.to_datetime(
        out["max_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.sort_values(["n_obs", "indicator_code"], ascending=[False, True])


def list_events(df: pd.DataFrame) -> pd.DataFrame:
    """List all event records in the dataframe."""
    ev = df[df["record_type"].astype(str).str.lower().eq("event")].copy()
    ev["observation_date"] = pd.to_datetime(
        ev["observation_date"], errors="coerce")
    cols = [
        c for c in
        ["record_id", "category", "indicator", "indicator_code",
            "observation_date", "source_name", "confidence"]
        if c in ev.columns
    ]
    return ev[cols].sort_values("observation_date")


def list_impact_links(df: pd.DataFrame) -> pd.DataFrame:
    """List all impact link records in the dataframe."""
    il = df[df["record_type"].astype(str).str.lower().eq("impact_link")].copy()
    cols = [
        c for c in
        [
            "record_id", "parent_id", "pillar", "related_indicator", "indicator_code",
            "impact_direction", "impact_magnitude", "lag_months", "evidence_basis",
            "confidence",
        ]
        if c in il.columns
    ]
    return il[cols].copy()
