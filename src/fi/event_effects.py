"""Functions for computing event effects on financial indicators."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

FINDEX_YEAR_GRID: list[int] = [2011, 2014, 2017, 2021, 2024]


def effect_step(t_year: int, event_year: int | float, lag_years: float, mag_pp: float) -> float:
    """Step effect: full impact after lag, zero before."""
    if pd.isna(event_year) or pd.isna(mag_pp):
        return 0.0
    start = float(event_year) + lag_years
    return float(mag_pp) if (t_year >= start) else 0.0


def effect_ramp(
    t_year: int,
    event_year: int | float,
    lag_years: float,
    ramp_years: float,
    mag_pp: float,
) -> float:
    """Ramp effect: linear increase from zero to full impact over ramp_years after lag."""
    if pd.isna(event_year) or pd.isna(mag_pp):
        return 0.0
    start = float(event_year) + lag_years
    if t_year < start:
        return 0.0
    if ramp_years <= 0:
        return float(mag_pp)
    progress = (t_year - start) / ramp_years
    progress = max(0.0, min(1.0, float(progress)))
    return float(mag_pp) * progress


def compute_effect_series(
    link_row: pd.Series,
    years: Sequence[int] | None = None,
    shape: str = "ramp",
    ramp_years: float = 3.0,
) -> pd.Series:
    """Compute effect series for a single impact link row."""
    years_list = list(years) if years is not None else FINDEX_YEAR_GRID
    year_index = pd.Index(years_list, dtype="int64")

    # Pylance-safe: always feed to_datetime a list[str]
    raw = link_row.get("event_date")

    # Treat None/NaN as missing
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        raw_str = ""
    else:
        raw_str = str(raw)

    event_ts = pd.to_datetime([raw_str], errors="coerce")[0]
    event_year: int | float = float("nan") if pd.isna(event_ts) else int(event_ts.year)

    lag_months = link_row.get("lag_months", 0.0)
    lag_years = (float(lag_months) / 12.0) if pd.notna(lag_months) else 0.0
    mag_pp = float(link_row.get("impact_magnitude_pp", np.nan))

    vals: list[float] = []
    for y in years_list:
        if shape == "step":
            vals.append(effect_step(int(y), event_year, lag_years, mag_pp))
        else:
            vals.append(effect_ramp(int(y), event_year, lag_years, ramp_years, mag_pp))

    return pd.Series(vals, index=year_index, dtype="float64")


def effects_tidy(
    df_summary: pd.DataFrame,
    indicators: Sequence[str] | None,
    years: Sequence[int] | None = None,
    default_shape: str = "ramp",
    default_ramp_years: float = 3.0,
) -> pd.DataFrame:
    """Compute tidy effects DataFrame from summary links DataFrame."""
    years_list = list(years) if years is not None else FINDEX_YEAR_GRID

    rows: list[dict[str, object]] = []
    for _, r in df_summary.iterrows():
        if indicators is not None and (r.get("indicator_code") not in set(indicators)):
            continue

        s = compute_effect_series(r, years=years_list, shape=default_shape, ramp_years=default_ramp_years)

        # Iterate over the known int year grid (avoids Hashable typing from Series.items()).
        for y in years_list:
            v = float(s.loc[y])
            rows.append(
                {
                    "event_record_id": r.get("event_record_id"),
                    "event_name": r.get("event_name"),
                    "indicator_code": r.get("indicator_code"),
                    "year": int(y),
                    "effect_pp": v,
                    "shape": default_shape,
                    "lag_months": r.get("lag_months"),
                }
            )
    return pd.DataFrame(rows)


def sum_effects_over_events(effects_df: pd.DataFrame) -> pd.DataFrame:
    """Sum effects over events to get total effect per indicator per year."""
    if effects_df.empty:
        return effects_df

    grouped = effects_df.groupby(["year", "indicator_code"], as_index=False)[["effect_pp"]].sum()
    out = grouped.rename(columns={"effect_pp": "total_effect_pp"})
    return out