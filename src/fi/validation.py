""" Validation functions for dataframes in the FI schema."""
from __future__ import annotations

import pandas as pd


REQUIRED_MIN_COLUMNS = [
    "record_id", "record_type", "pillar", "category",
    "indicator", "indicator_code", "value_numeric",
    "observation_date", "source_name", "source_url",
    "source_type", "confidence",
    "parent_id",
    "original_text",
    "collected_by", "collection_date", "notes",
]


def assert_min_schema(df: pd.DataFrame) -> None:
    """Assert that the dataframe has the required minimum columns."""
    missing = [c for c in REQUIRED_MIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def invalid_events_with_pillar(df: pd.DataFrame) -> pd.DataFrame:
    """Events must have empty pillar; return offending rows."""
    m = (
        df["record_type"].astype(str).str.lower().eq("event")
        & df["pillar"].notna()
        & (df["pillar"].astype(str).str.strip() != "")
    )
    return df.loc[m].copy()


def invalid_impact_links_missing_parent(df: pd.DataFrame) -> pd.DataFrame:
    """impact_link rows must have parent_id; return offending rows."""
    m = (
        df["record_type"].astype(str).str.lower().eq("impact_link")
        & (df["parent_id"].isna() | (df["parent_id"].astype(str).str.strip() == ""))
    )
    return df.loc[m].copy()


def invalid_impact_links_unresolved_parent(df: pd.DataFrame) -> pd.DataFrame:
    """parent_id must match an event record_id; return unresolved links."""
    events = set(
        df.loc[df["record_type"].astype(
            str).str.lower().eq("event"), "record_id"]
        .astype(str)
        .tolist()
    )
    links = df.loc[df["record_type"].astype(
        str).str.lower().eq("impact_link")].copy()
    links["parent_id"] = links["parent_id"].astype(str)
    bad = links.loc[~links["parent_id"].isin(events)].copy()
    return bad
def duplicate_record_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with duplicate record_id values."""
    rid = df["record_id"].astype(str)
    return df.loc[rid.duplicated(keep=False)].copy()