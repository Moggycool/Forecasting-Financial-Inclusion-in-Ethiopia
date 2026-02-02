"""Functions for computing event effects on financial indicators."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

FINDEX_YEAR_GRID: list[int] = [2011, 2014, 2017, 2021, 2024]


def _to_timestamp(x) -> pd.Timestamp:
    """Coerce a single value to Timestamp (NaT if invalid)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    return pd.to_datetime(str(x), errors="coerce")


def _month_index(ts: pd.Timestamp) -> float:
    """Continuous month index for month-precision arithmetic."""
    return float(ts.year) * 12.0 + float(ts.month - 1)


def _year_month_index(year: int) -> float:
    """Month index for a Findex snapshot year, assuming snapshot at year-end (Dec).

    This choice makes a 12-month lag from May-2021 start effectively from May-2022,
    and by Dec-2024 you realize ~31/36 of a 3-year ramp.
    If you prefer mid-year snapshots, change month here.
    """
    return float(year) * 12.0 + 11.0  # Dec


def effect_step_month_aware(t_year: int, event_ts: pd.Timestamp, lag_months: float, mag_pp: float) -> float:
    """Step effect with month-aware lag."""
    if pd.isna(event_ts) or pd.isna(mag_pp):
        return 0.0
    start_m = _month_index(event_ts) + float(lag_months)
    t_m = _year_month_index(int(t_year))
    return float(mag_pp) if (t_m >= start_m) else 0.0


def effect_ramp_month_aware(
    t_year: int,
    event_ts: pd.Timestamp,
    lag_months: float,
    ramp_months: float,
    mag_pp: float,
) -> float:
    """Ramp effect with month-aware lag and ramp duration."""
    if pd.isna(event_ts) or pd.isna(mag_pp):
        return 0.0

    start_m = _month_index(event_ts) + float(lag_months)
    t_m = _year_month_index(int(t_year))

    if t_m < start_m:
        return 0.0
    if ramp_months <= 0:
        return float(mag_pp)

    progress = (t_m - start_m) / float(ramp_months)
    progress = max(0.0, min(1.0, float(progress)))
    return float(mag_pp) * progress


def compute_effect_series(
    link_row: pd.Series,
    years: Sequence[int] | None = None,
    shape: str | None = None,
    ramp_years: float | None = None,
) -> pd.Series:
    """Compute effect series for a single impact link row.

    Notes
    -----
    - Uses month-aware lag and ramp so 12-month lag behaves correctly.
    - Allows per-link overrides if columns exist:
        - effect_shape / shape
        - ramp_years
    """
    years_list = list(years) if years is not None else FINDEX_YEAR_GRID
    year_index = pd.Index(years_list, dtype="int64")

    # Event date -> timestamp
    raw_date = link_row.get("event_date")
    event_ts = _to_timestamp(raw_date)

    # Lag
    lag_months = link_row.get("lag_months", 0.0)
    lag_months = float(lag_months) if pd.notna(lag_months) else 0.0

    # Magnitude (pp)
    mag_pp = link_row.get("impact_magnitude_pp", np.nan)
    mag_pp = float(mag_pp) if pd.notna(mag_pp) else np.nan

    # Shape: prefer per-row if available
    row_shape = link_row.get("effect_shape", link_row.get("shape", None))
    use_shape = (str(row_shape).strip().lower() if row_shape is not None and str(row_shape).strip() else None) or (
        str(shape).strip().lower() if shape is not None else "ramp"
    )

    # Ramp duration: prefer per-row if available
    row_ramp_years = link_row.get("ramp_years", None)
    use_ramp_years = row_ramp_years if pd.notna(row_ramp_years) else ramp_years
    use_ramp_years = float(use_ramp_years) if use_ramp_years is not None else 3.0
    ramp_months = 12.0 * use_ramp_years

    vals: list[float] = []
    for y in years_list:
        if use_shape == "step":
            vals.append(effect_step_month_aware(int(y), event_ts, lag_months, mag_pp))
        else:
            vals.append(effect_ramp_month_aware(int(y), event_ts, lag_months, ramp_months, mag_pp))

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
    ind_set = set(indicators) if indicators is not None else None

    rows: list[dict[str, object]] = []
    for _, r in df_summary.iterrows():
        ind = r.get("indicator_code")
        if ind_set is not None and (ind not in ind_set):
            continue

        s = compute_effect_series(
            r,
            years=years_list,
            shape=default_shape,
            ramp_years=default_ramp_years,
        )

        for y in years_list:
            rows.append(
                {
                    "event_record_id": r.get("event_record_id"),
                    "event_name": r.get("event_name"),
                    "indicator_code": ind,
                    "year": int(y),
                    "effect_pp": float(s.loc[int(y)]),
                    "shape": r.get("effect_shape", default_shape),
                    "ramp_years": r.get("ramp_years", default_ramp_years),
                    "lag_months": r.get("lag_months"),
                    # useful for debugging
                    "impact_magnitude_pp": r.get("impact_magnitude_pp"),
                    "event_date": r.get("event_date"),
                    "link_record_id": r.get("link_record_id", r.get("record_id")),
                }
            )

    return pd.DataFrame(rows)


def sum_effects_over_events(effects_df: pd.DataFrame) -> pd.DataFrame:
    """Sum effects over events to get total effect per indicator per year."""
    if effects_df is None or effects_df.empty:
        return effects_df if effects_df is not None else pd.DataFrame()

    grouped = effects_df.groupby(["year", "indicator_code"], as_index=False)[["effect_pp"]].sum()
    out = grouped.rename(columns={"effect_pp": "total_effect_pp"})
    return out
