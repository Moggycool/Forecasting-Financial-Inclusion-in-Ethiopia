"""Utilities for processing impact links and related events."""
from __future__ import annotations

import numpy as np
import pandas as pd


def standardize_direction(x) -> int:
    """
    Map textual direction -> sign.
    Accepts: increase/positive/+ -> +1, decrease/negative/- -> -1
    """
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in ["increase", "positive", "+", "up", "pos"]:
        return 1
    if s in ["decrease", "negative", "-", "down", "neg"]:
        return -1
    if s.startswith("-"):
        return -1
    return 1


def to_float(x) -> float:
    """Coerce input to float, stripping non-numeric characters if needed."""
    if pd.isna(x) or x == "":
        return np.nan
    try:
        return float(x)
    except Exception:
        s = "".join(ch for ch in str(x) if ch.isdigit() or ch in [".", "-"])
        return float(s) if s else np.nan


def join_links_events(links_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Join impact links with events on parent_id -> record_id."""
    return links_df.merge(
        events_df,
        left_on="parent_id",
        right_on="record_id",
        how="left",
        suffixes=("_link", "_event"),
    )


def build_impact_links_summary(joined: pd.DataFrame) -> pd.DataFrame:
    """Returns a tidy link-level table with signed magnitude in pp-space."""
    out = pd.DataFrame(
        {
            "link_record_id": joined.get("record_id_link", joined.get("record_id")),
            "event_record_id": joined.get("record_id_event"),
            "event_name": joined.get("indicator_event", joined.get("indicator")),
            "event_category": joined.get("category_event", joined.get("category")),
            "event_date": joined.get("event_date", joined.get("observation_date_event", joined.get("observation_date"))),
            "event_source_name": joined.get("source_name_event", joined.get("source_name")),
            "event_confidence": joined.get("confidence_event"),
            "pillar": joined.get("pillar_link", joined.get("pillar")),
            "related_indicator": joined.get("related_indicator"),
            # NOTE: raw data uses related_indicator as the actual code; indicator_code is often blank
            "indicator_code": joined.get("indicator_code"),
            "impact_direction": joined.get("impact_direction"),
            "direction_sign": joined.get("impact_direction").apply(standardize_direction),
            # NOTE: raw data often stores numeric magnitude in impact_estimate; impact_magnitude can be 'high/medium/low'
            "impact_magnitude": joined.get("impact_magnitude").apply(to_float),
            "impact_estimate": joined.get("impact_estimate").apply(to_float),
            "lag_months": joined.get("lag_months").apply(to_float),
            "evidence_basis": joined.get("evidence_basis"),
            "confidence_link": joined.get("confidence_link", joined.get("confidence")),
        }
    )

    # Fill indicator_code from related_indicator when missing/blank
    if "indicator_code" in out.columns:
        ind = out["indicator_code"].astype("string").fillna("").str.strip()
        rel = out["related_indicator"].astype("string").fillna("").str.strip()
        out["indicator_code"] = ind.mask(ind.eq(""), rel)

    # Choose best available numeric magnitude
    mag = out["impact_estimate"].where(out["impact_estimate"].notna(), out["impact_magnitude"])
    out["impact_magnitude_pp"] = mag * out["direction_sign"]

    # Keep output schema consistent with the rest of the project
    out = out.drop(columns=["impact_estimate"], errors="ignore")
    return out