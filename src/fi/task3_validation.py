"""Functions for validating observed data against predicted impacts.

This module is defensive by design because the "observations" input may sometimes be a
non-observations summary table (e.g., temporal ranges). In those cases, functions will
return NaN rather than raising, so notebooks can skip validation gracefully.

Key behaviors:
- `get_observed(...)` returns NaN if required columns are missing.
- `validate_telebirr_mm(...)` is parameterized by `target_indicator` and reports which
  indicators Telebirr links actually target (`telebirr_link_targets`), plus the count
  of all Telebirr links found (`n_telebirr_links_all_targets`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _year_from_date(s) -> float:
    """Extract year from a date-like value, returning NaN if invalid."""
    dt = pd.to_datetime(s, errors="coerce")
    if isinstance(dt, pd.Timestamp):
        return float(dt.year) if not pd.isna(dt) else np.nan

    ser = pd.Series(dt).dropna()
    if ser.empty:
        return np.nan
    return float(pd.Timestamp(ser.iloc[0]).year)


def _pick_date_col(df: pd.DataFrame) -> str | None:
    """Pick a plausible date column for deriving year."""
    for c in ["observation_date", "date", "period_end", "period_start", "event_date"]:
        if c in df.columns:
            return c
    return None


def _pick_value_col(df: pd.DataFrame) -> str | None:
    """Pick a plausible numeric value column."""
    for c in ["value_numeric", "value", "observed_value"]:
        if c in df.columns:
            return c
    return None


def _normalize_year_column(df: pd.DataFrame) -> pd.Series:
    """Return a nullable Int64 year Series extracted from df['year'] without pd.to_numeric (Pylance-friendly)."""
    s = df["year"].astype("string")
    extracted = s.str.extract(r"(\d{4})", expand=False)
    return extracted.astype("Int64")


def _derive_year_from_dates(dates: pd.Series) -> pd.Series:
    """Return a nullable Int64 year Series from date-like values, with NaT -> <NA>."""
    dts = pd.to_datetime(dates, errors="coerce")
    years = pd.Series(pd.DatetimeIndex(dts).year, index=dates.index)
    years = years.where(pd.notna(dts), pd.NA).astype("Int64")
    return years


def get_observed(
    df_obs: pd.DataFrame,
    indicator_code: str,
    year: int,
    gender: str = "all",
    location: str = "national",
) -> float:
    """Get observed value for given indicator_code and year, optionally filtering by gender and location.

    Works with observations tables that may have:
    - `year` directly, or a date column such as `observation_date`/`period_end`
    - `indicator_code` or only `related_indicator`
    - different numeric value columns (prefers `value_numeric`)
    """
    if df_obs is None or df_obs.empty:
        return np.nan

    df = df_obs.copy()

    # Ensure indicator_code exists
    if "indicator_code" not in df.columns and "related_indicator" in df.columns:
        df["indicator_code"] = df["related_indicator"]

    if "indicator_code" not in df.columns:
        # Not an observations table
        return np.nan

    # Ensure year exists
    if "year" in df.columns:
        df["year"] = _normalize_year_column(df)
    else:
        date_col = _pick_date_col(df)
        if date_col is None:
            return np.nan
        df["year"] = _derive_year_from_dates(pd.Series(df[date_col], index=df.index))

    value_col = _pick_value_col(df)
    if value_col is None:
        return np.nan

    sub = df[(df["indicator_code"] == indicator_code) & (df["year"] == int(year))]

    if "gender" in sub.columns:
        sub = sub[sub["gender"].fillna("all") == gender]
    if "location" in sub.columns:
        sub = sub[sub["location"].fillna("national") == location]

    if sub.empty:
        return np.nan

    vals = pd.to_numeric(pd.Series(sub[value_col], index=sub.index), errors="coerce")
    return float(vals.mean())


def validate_telebirr_mm(
    df_obs: pd.DataFrame,
    df_links_summary: pd.DataFrame,
    year_a: int = 2021,
    year_b: int = 2024,
    target_indicator: str = "ACC_MM_ACCOUNT",
    event_regex: str = "telebirr",
) -> pd.DataFrame:
    """Validate observed changes against predicted impacts for a Telebirr-related event."""
    obs_a = get_observed(df_obs, target_indicator, year_a)
    obs_b = get_observed(df_obs, target_indicator, year_b)
    obs_delta = obs_b - obs_a

    if df_links_summary is None or df_links_summary.empty or "event_name" not in df_links_summary.columns:
        tele_all = pd.DataFrame()
    else:
        mask_evt = df_links_summary["event_name"].astype(str).str.contains(event_regex, case=False, na=False)
        tele_all = df_links_summary[mask_evt].copy()

    tele_targets = (
        tele_all["indicator_code"].astype(str).str.strip().replace({"": np.nan}).dropna().unique().tolist()
        if (not tele_all.empty and "indicator_code" in tele_all.columns)
        else []
    )
    tele_targets_str = ", ".join(sorted(tele_targets)) if tele_targets else ""

    if tele_all.empty or "indicator_code" not in tele_all.columns:
        tele = pd.DataFrame()
    else:
        mask_ind = tele_all["indicator_code"].astype(str).str.strip().str.upper().eq(target_indicator.upper())
        tele = tele_all[mask_ind].copy()

    if not tele.empty and "impact_magnitude_pp" in tele.columns:
        mags = pd.to_numeric(pd.Series(tele["impact_magnitude_pp"], index=tele.index), errors="coerce")
        pred_tele_saturated = float(mags.sum())
    else:
        pred_tele_saturated = np.nan

    return pd.DataFrame(
        [
            {
                "indicator_code": target_indicator,
                "obs_year_a": year_a,
                "obs_year_b": year_b,
                "obs_a": obs_a,
                "obs_b": obs_b,
                "obs_delta_pp": obs_delta,
                "pred_telebirr_saturated_pp": pred_tele_saturated,
                "residual_pp": (obs_delta - pred_tele_saturated) if pd.notna(pred_tele_saturated) else np.nan,
                "n_telebirr_links": int(len(tele)),
                "telebirr_link_targets": tele_targets_str,
                "n_telebirr_links_all_targets": int(len(tele_all)),
            }
        ]
    )