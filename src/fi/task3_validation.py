"""Functions for validating observed data against predicted impacts.

Defensive by design: if inputs are not true observations/effects tables (e.g., temporal-range
summaries), functions return NaN rather than raising.

Key behaviors:
- `get_observed(...)` returns NaN if required columns are missing.
- `validate_telebirr_mm(...)` reports Telebirr link diagnostics and computes residuals.

Task 3 rubric alignment:
- `validate_telebirr_mm(...)` can consume `df_event_effects_tidy` (month-aware).
  It prefers realized Telebirr effect for year_b, falling back to saturated link sums.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def _normalize_year_series_to_int64(year_like: pd.Series) -> pd.Series:
    """Nullable Int64 year Series extracted from mixed types (safe for Pylance)."""
    s = year_like.astype("string")
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
    - `year` directly, OR `fiscal_year`, OR a date column (observation_date/period_end/etc.)
    - `indicator_code` or `related_indicator`
    - different numeric value columns (prefers value_numeric)
    """
    if df_obs is None or df_obs.empty:
        return np.nan

    df = df_obs.copy()

    # If this is a unified table, optionally narrow to observation rows (defensive)
    if "record_type" in df.columns:
        rt = df["record_type"].astype("string").str.lower().fillna("")
        obs_mask = rt.eq("observation")
        if obs_mask.any():
            df = df[obs_mask].copy()

    # Ensure indicator_code exists
    if "indicator_code" not in df.columns and "related_indicator" in df.columns:
        df["indicator_code"] = df["related_indicator"]

    if "indicator_code" not in df.columns:
        return np.nan

    # Ensure year exists (year -> fiscal_year -> derive from date)
    if "year" in df.columns:
        df["year"] = _normalize_year_series_to_int64(pd.Series(df["year"], index=df.index))
    elif "fiscal_year" in df.columns:
        df["year"] = _normalize_year_series_to_int64(pd.Series(df["fiscal_year"], index=df.index))
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
    vals = vals.dropna()
    if vals.empty:
        return np.nan
    return float(vals.mean())


def _pick_effect_col(df_event_effects_tidy: pd.DataFrame) -> str | None:
    """Pick a plausible realized effect column from tidy effects."""
    for c in ["effect_pp", "effect", "realized_effect_pp", "pred_effect_pp"]:
        if c in df_event_effects_tidy.columns:
            return c
    return None


def _normalize_tidy_effects(df_event_effects_tidy: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized columns for matching (event_id, indicator_code, year)."""
    df = df_event_effects_tidy.copy()

    # Event id aliases (your file uses event_record_id)
    if "event_id" not in df.columns and "event_record_id" in df.columns:
        df["event_id"] = df["event_record_id"]
    if "event_id" not in df.columns and "event_code" in df.columns:
        df["event_id"] = df["event_code"]

    # Indicator aliases
    if "indicator_code" not in df.columns and "indicator" in df.columns:
        df["indicator_code"] = df["indicator"]

    # Year normalization
    if "year" in df.columns:
        df["year"] = _normalize_year_series_to_int64(pd.Series(df["year"], index=df.index))
    else:
        date_col = _pick_date_col(df)
        if date_col is None:
            df["year"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
        else:
            df["year"] = _derive_year_from_dates(pd.Series(df[date_col], index=df.index))

    return df


def _compute_pred_telebirr_realized(
    df_event_effects_tidy: pd.DataFrame | None,
    telebirr_event_id: str,
    target_indicator: str,
    year_b: int,
    event_regex: str = "telebirr",
) -> tuple[float, str]:
    """Return (realized_effect_pp, match_mode).

    match_mode:
      - 'by_event_id' if matched on telebirr_event_id
      - 'by_event_name' if event_id match failed but event_name regex matched
      - 'missing' if not found / not parseable
    """
    if df_event_effects_tidy is None or df_event_effects_tidy.empty:
        return (np.nan, "missing")

    df = _normalize_tidy_effects(df_event_effects_tidy)

    if "event_id" not in df.columns or "indicator_code" not in df.columns:
        return (np.nan, "missing")

    effect_col = _pick_effect_col(df)
    if effect_col is None:
        return (np.nan, "missing")

    # 1) Match by event id (preferred)
    sub = df[
        (df["event_id"].astype(str).str.strip() == str(telebirr_event_id).strip())
        & (df["indicator_code"].astype(str).str.strip().str.upper() == target_indicator.strip().upper())
        & (df["year"] == int(year_b))
    ]
    if not sub.empty:
        vals = pd.to_numeric(pd.Series(sub[effect_col], index=sub.index), errors="coerce").dropna()
        if not vals.empty:
            return (float(vals.mean()), "by_event_id")

    # 2) Optional fallback: match by event_name regex (helps if ids differ)
    if "event_name" in df.columns:
        mask_evt = df["event_name"].astype(str).str.contains(event_regex, case=False, na=False)
        sub2 = df[
            mask_evt
            & (df["indicator_code"].astype(str).str.strip().str.upper() == target_indicator.strip().upper())
            & (df["year"] == int(year_b))
        ]
        if not sub2.empty:
            vals = pd.to_numeric(pd.Series(sub2[effect_col], index=sub2.index), errors="coerce").dropna()
            if not vals.empty:
                return (float(vals.mean()), "by_event_name")

    return (np.nan, "missing")


def validate_telebirr_mm(
    df_obs: pd.DataFrame,
    df_links_summary: pd.DataFrame,
    df_event_effects_tidy: pd.DataFrame | None = None,
    telebirr_event_id: str = "EVT_0001",
    year_a: int = 2021,
    year_b: int = 2024,
    target_indicator: str = "ACC_MM_ACCOUNT",
    event_regex: str = "telebirr",
) -> pd.DataFrame:
    """Validate observed changes against predicted impacts for a Telebirr-related event.

    Preference order for prediction:
    1) realized effect from df_event_effects_tidy (month-aware)
    2) saturated sum from df_links_summary (impact_magnitude_pp)
    """
    obs_a = get_observed(df_obs, target_indicator, year_a)
    obs_b = get_observed(df_obs, target_indicator, year_b)
    obs_delta = obs_b - obs_a

    # --- All Telebirr links (any indicator) for diagnostics ---
    if df_links_summary is None or df_links_summary.empty or "event_name" not in df_links_summary.columns:
        tele_all = pd.DataFrame()
    else:
        mask_evt = df_links_summary["event_name"].astype(str).str.contains(event_regex, case=False, na=False)
        tele_all = df_links_summary[mask_evt].copy()

    tele_targets = (
        tele_all["indicator_code"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .unique()
        .tolist()
        if (not tele_all.empty and "indicator_code" in tele_all.columns)
        else []
    )
    tele_targets_str = ", ".join(sorted(tele_targets)) if tele_targets else ""

    # Telebirr links targeting the requested indicator
    if tele_all.empty or "indicator_code" not in tele_all.columns:
        tele = pd.DataFrame()
    else:
        mask_ind = tele_all["indicator_code"].astype(str).str.strip().str.upper().eq(target_indicator.upper())
        tele = tele_all[mask_ind].copy()

    # --- Saturated prediction (fallback) ---
    if not tele.empty and "impact_magnitude_pp" in tele.columns:
        mags = pd.to_numeric(pd.Series(tele["impact_magnitude_pp"], index=tele.index), errors="coerce").dropna()
        pred_tele_saturated = float(mags.sum()) if not mags.empty else np.nan
    else:
        pred_tele_saturated = np.nan

    # --- Realized prediction (preferred) ---
    pred_tele_realized, realized_match_mode = _compute_pred_telebirr_realized(
        df_event_effects_tidy=df_event_effects_tidy,
        telebirr_event_id=telebirr_event_id,
        target_indicator=target_indicator,
        year_b=year_b,
        event_regex=event_regex,
    )

    if pd.notna(pred_tele_realized):
        pred_used = pred_tele_realized
        pred_source = "tidy_realized"
    else:
        pred_used = pred_tele_saturated
        pred_source = "saturated_sum" if pd.notna(pred_tele_saturated) else "missing"

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
                "pred_telebirr_realized_pp": pred_tele_realized,
                "pred_used_pp": pred_used,
                "pred_source": pred_source,
                "tidy_realized_match_mode": realized_match_mode,
                "residual_pp": (obs_delta - pred_used) if pd.notna(pred_used) else np.nan,
                "n_telebirr_links": int(len(tele)),
                "telebirr_link_targets": tele_targets_str,
                "n_telebirr_links_all_targets": int(len(tele_all)),
                "telebirr_event_id": telebirr_event_id,
            }
        ]
    )
