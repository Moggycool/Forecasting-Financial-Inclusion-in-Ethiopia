"""Functions for computing event effects on financial inclusion indicators.

Implements month-aware lag and (by default) linear ramp diffusion across an annual year grid.

Key ideas
---------
- Each impact link has a total magnitude in percentage points (pp): `impact_magnitude_pp`.
- Each link can have a lag in months (`lag_months`).
- Effects can be represented as:
    - step: 0 until lag is complete, then full magnitude
    - ramp: 0 until lag is complete, then linearly ramps to full magnitude over `ramp_years`

This module is intentionally deterministic and audit-friendly.

Update note (Task 4 horizon support)
-----------------------------------
Task 4 requires annual event effects for forecast years (e.g., 2025–2027).
Earlier versions often generated tidy effects only on the FINDEX observation grid
(2011, 2014, 2017, 2021, 2024). That causes Task 4 joins to return 0 event impacts
for 2025–2027 even if events have `event_date` in 2025.

This version adds utilities to extend the year grid safely and makes it easy for
Task 3 scripts to generate tidy effects including the forecast horizon.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Historical FINDEX-like grid used in earlier tasks
FINDEX_YEAR_GRID: list[int] = [2011, 2014, 2017, 2021, 2024]

# Default forecast horizon used in Task 4 (override as needed)
DEFAULT_FORECAST_YEARS: list[int] = [2025, 2026, 2027]


def _to_timestamp(x) -> pd.Timestamp:
    """Coerce a single value to Timestamp (NaT if invalid)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    return pd.to_datetime(str(x), errors="coerce")


def _month_index(ts: pd.Timestamp) -> float:
    """Continuous month index for month-precision arithmetic."""
    return float(ts.year) * 12.0 + float(ts.month - 1)


def _year_month_index(year: int, snapshot_month: int = 12) -> float:
    """Month index for a snapshot year (month-aware arithmetic).

    Parameters
    ----------
    year:
        Snapshot year.
    snapshot_month:
        Month of within-year snapshot (default=12 / Dec).

    Notes
    -----
    Using Dec makes a 12-month lag from May-2021 start effectively from May-2022,
    and by Dec-2024 you realize ~31/36 of a 3-year ramp (if ramp_years=3).
    """
    m = int(snapshot_month)
    if not (1 <= m <= 12):
        raise ValueError("snapshot_month must be in 1..12")
    return float(year) * 12.0 + float(m - 1)


def normalize_years(
    years: Sequence[int] | None,
    *,
    base_grid: Sequence[int] = FINDEX_YEAR_GRID,
    ensure_years: Sequence[int] | None = None,
) -> list[int]:
    """Normalize/validate a year grid and optionally ensure extra years are present.

    This function is the core patch for the Task 3 → Task 4 handoff:
    include ensure_years=[2025, 2026, 2027] (or any horizon) so tidy outputs contain forecast years.
    """
    if years is None:
        years_list = [int(y) for y in base_grid]
    else:
        years_list = [int(y) for y in years]

    if ensure_years is not None:
        years_list = sorted(set(years_list).union({int(y) for y in ensure_years}))

    years_list = sorted(set(years_list))
    if len(years_list) == 0:
        raise ValueError("years grid is empty")

    return years_list


def effect_step_month_aware(
    t_year: int,
    event_ts: pd.Timestamp,
    lag_months: float,
    mag_pp: float,
    snapshot_month: int = 12,
) -> float:
    """Step effect with month-aware lag."""
    if pd.isna(event_ts) or pd.isna(mag_pp):
        return 0.0
    start_m = _month_index(event_ts) + float(lag_months)
    t_m = _year_month_index(int(t_year), snapshot_month=snapshot_month)
    return float(mag_pp) if (t_m >= start_m) else 0.0


def effect_ramp_month_aware(
    t_year: int,
    event_ts: pd.Timestamp,
    lag_months: float,
    ramp_months: float,
    mag_pp: float,
    snapshot_month: int = 12,
) -> float:
    """Ramp effect with month-aware lag and ramp duration."""
    if pd.isna(event_ts) or pd.isna(mag_pp):
        return 0.0

    start_m = _month_index(event_ts) + float(lag_months)
    t_m = _year_month_index(int(t_year), snapshot_month=snapshot_month)

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
    snapshot_month: int = 12,
) -> pd.Series:
    """Compute effect series for a single impact link row.

    Required columns (expected)
    ---------------------------
    - event_date
    - lag_months
    - impact_magnitude_pp

    Optional per-link overrides
    ---------------------------
    - effect_shape / shape: "ramp" (default) or "step"
    - ramp_years: float
    """
    years_list = normalize_years(years)
    year_index = pd.Index(years_list, dtype="int64")

    # Event date -> timestamp
    event_ts = _to_timestamp(link_row.get("event_date"))

    # Lag
    lag_m = link_row.get("lag_months", 0.0)
    lag_m = float(lag_m) if pd.notna(lag_m) else 0.0

    # Magnitude (pp)
    mag_pp = link_row.get("impact_magnitude_pp", np.nan)
    mag_pp = float(mag_pp) if pd.notna(mag_pp) else np.nan

    # Shape resolution: prefer per-row if available
    row_shape = link_row.get("effect_shape", link_row.get("shape", None))
    use_shape = (
        (str(row_shape).strip().lower() if row_shape is not None and str(row_shape).strip() else None)
        or (str(shape).strip().lower() if shape is not None and str(shape).strip() else "ramp")
    )
    if use_shape not in {"ramp", "step"}:
        use_shape = "ramp"  # defensive fallback

    # Ramp duration: prefer per-row if available
    row_ramp_years = link_row.get("ramp_years", None)
    use_ramp_years = row_ramp_years if pd.notna(row_ramp_years) else ramp_years
    use_ramp_years = float(use_ramp_years) if use_ramp_years is not None else 3.0
    ramp_months = 12.0 * use_ramp_years

    vals: list[float] = []
    for y in years_list:
        if use_shape == "step":
            vals.append(effect_step_month_aware(int(y), event_ts, lag_m, mag_pp, snapshot_month=snapshot_month))
        else:
            vals.append(
                effect_ramp_month_aware(
                    int(y),
                    event_ts,
                    lag_m,
                    ramp_months,
                    mag_pp,
                    snapshot_month=snapshot_month,
                )
            )

    return pd.Series(vals, index=year_index, dtype="float64")


def effects_tidy(
    df_summary: pd.DataFrame,
    indicators: Sequence[str] | None,
    years: Sequence[int] | None = None,
    default_shape: str = "ramp",
    default_ramp_years: float = 3.0,
    snapshot_month: int = 12,
    *,
    ensure_forecast_years: bool = True,
    forecast_years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Compute tidy effects DataFrame from summary links DataFrame.

    Output columns (export-friendly)
    --------------------------------
    event_record_id,event_name,indicator_code,year,effect_pp,shape,ramp_years,lag_months,
    impact_magnitude_pp,event_date,link_record_id

    Parameters
    ----------
    ensure_forecast_years:
        If True, ensures forecast_years (default: DEFAULT_FORECAST_YEARS) are included in the year grid.
        This is the recommended Task 3 patch so Task 4 can join effects for 2025–2027.
    """
    if df_summary is None or df_summary.empty:
        return pd.DataFrame(
            columns=[
                "event_record_id",
                "event_name",
                "indicator_code",
                "year",
                "effect_pp",
                "shape",
                "ramp_years",
                "lag_months",
                "impact_magnitude_pp",
                "event_date",
                "link_record_id",
            ]
        )

    ind_set = set(indicators) if indicators is not None else None

    ensure_years = None
    if ensure_forecast_years:
        ensure_years = list(forecast_years) if forecast_years is not None else DEFAULT_FORECAST_YEARS

    years_list = normalize_years(years, ensure_years=ensure_years)

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
            snapshot_month=snapshot_month,
        )

        # Write resolved values (avoid NaN leakage for audit stability)
        row_shape = r.get("effect_shape", r.get("shape", None))
        out_shape = (
            str(row_shape).strip().lower()
            if row_shape is not None and str(row_shape).strip()
            else str(default_shape).strip().lower()
        )
        row_ramp = r.get("ramp_years", None)
        out_ramp = float(row_ramp) if pd.notna(row_ramp) else float(default_ramp_years)

        for y in years_list:
            rows.append(
                {
                    "event_record_id": r.get("event_record_id"),
                    "event_name": r.get("event_name"),
                    "indicator_code": ind,
                    "year": int(y),
                    "effect_pp": float(s.loc[int(y)]),
                    "shape": out_shape,
                    "ramp_years": out_ramp,
                    "lag_months": r.get("lag_months"),
                    "impact_magnitude_pp": r.get("impact_magnitude_pp"),
                    "event_date": r.get("event_date"),
                    "link_record_id": r.get("link_record_id", r.get("record_id")),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["indicator_code", "event_record_id", "year"]).reset_index(drop=True)
    return out


def sum_effects_over_events(effects_df: pd.DataFrame) -> pd.DataFrame:
    """Sum effects over events to get total effect per indicator per year."""
    if effects_df is None or effects_df.empty:
        return effects_df if effects_df is not None else pd.DataFrame()

    grouped = effects_df.groupby(["year", "indicator_code"], as_index=False)[["effect_pp"]].sum()
    return grouped.rename(columns={"effect_pp": "total_effect_pp"})
