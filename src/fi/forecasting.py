"""
Task 4 forecasting utilities.

Targets (Task 4):
- ACC_OWNERSHIP: % of adults with an account (FI or mobile money)
- USG_ACTIVE_RATE: % proxy for active usage of digital financial services

Given sparse points, we implement:
1) OLS trend forecast when >= 2 points
2) Fallback flat carry-forward baseline when only 1 point exists
3) Event-augmented forecast = baseline + annual event effects (Task 3 event_effects_tidy.csv)
4) Scenario analysis via scaling trend and event impacts

Uncertainty:
- Trend uncertainty from approximate OLS prediction intervals
- Flat-baseline uncertainty: configurable sd widened with horizon
- Event uncertainty: sd = |event_effect| * event_sd_frac
- Scenario extra_sd_pp

Special handling (Usage proxy):
- event_effects_tidy has no direct rows for USG_ACTIVE_RATE.
- We map USG_P2P_COUNT annual effects into USG_ACTIVE_RATE pp effects via:
    USG_ACTIVE_RATE_effect_pp = linkage_beta * USG_P2P_COUNT_effect_pp
  This is explicit, auditable, and can be sensitivity-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

TargetCode = Literal["ACC_OWNERSHIP", "USG_ACTIVE_RATE"]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    event_multiplier: float
    trend_multiplier: float = 1.0
    extra_sd_pp: float = 0.0


DEFAULT_SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec("pessimistic", event_multiplier=0.50, trend_multiplier=0.90, extra_sd_pp=1.0),
    ScenarioSpec("base",        event_multiplier=1.00, trend_multiplier=1.00, extra_sd_pp=0.0),
    ScenarioSpec("optimistic",  event_multiplier=1.25, trend_multiplier=1.05, extra_sd_pp=0.5),
]


def _clamp_pct(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 100.0)


def prepare_findex_series(
    df_obs: pd.DataFrame,
    indicator_code: str,
    gender: str = "all",
    location: str = "national",
) -> pd.DataFrame:
    df = df_obs.copy()

    required = {"indicator_code", "year", "value_numeric"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Observations missing required columns: {sorted(missing)}")

    df = df[df["indicator_code"].astype(str) == indicator_code]

    if "gender" in df.columns and gender is not None:
        df = df[df["gender"].astype(str) == gender]
    if "location" in df.columns and location is not None:
        df = df[df["location"].astype(str) == location]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value_pct"] = pd.to_numeric(df["value_numeric"], errors="coerce")

    out = (
        df[["year", "value_pct"]]
        .dropna()
        .groupby("year", as_index=False)["value_pct"].mean()
        .sort_values("year")
    )

    if out.empty:
        raise ValueError(
            f"No observations found for indicator={indicator_code}, gender={gender}, location={location}."
        )
    return out


def ols_trend_forecast(series: pd.DataFrame, years_forecast: Sequence[int], alpha: float = 0.05) -> pd.DataFrame:
    y = series["value_pct"].to_numpy(dtype=float)
    x = series["year"].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError("Need at least 2 historical points for trend regression.")

    x_mean = x.mean()
    Sxx = np.sum((x - x_mean) ** 2)
    if Sxx == 0:
        raise ValueError("Degenerate year values.")

    b1 = np.sum((x - x_mean) * (y - y.mean())) / Sxx
    b0 = y.mean() - b1 * x_mean

    y_hat = b0 + b1 * x
    resid = y - y_hat
    dof = max(len(x) - 2, 1)
    s2 = np.sum(resid**2) / dof
    s = float(np.sqrt(s2))

    try:
        from scipy.stats import t  # type: ignore
        tcrit = float(t.ppf(1 - alpha / 2, df=dof))
    except Exception:
        tcrit = 2.0

    rows = []
    for yr in years_forecast:
        yr = int(yr)
        y_pred = b0 + b1 * yr
        se_pred = s * np.sqrt(1.0 + (1.0 / len(x)) + ((yr - x_mean) ** 2) / Sxx)
        lo = y_pred - tcrit * se_pred
        hi = y_pred + tcrit * se_pred
        rows.append(
            dict(
                year=yr,
                trend_pred=float(_clamp_pct(np.array([y_pred]))[0]),
                trend_se=float(se_pred),
                trend_lo=float(_clamp_pct(np.array([lo]))[0]),
                trend_hi=float(_clamp_pct(np.array([hi]))[0]),
            )
        )
    return pd.DataFrame(rows)


def flat_baseline_forecast(
    series: pd.DataFrame,
    years_forecast: Sequence[int],
    alpha: float = 0.05,
    base_sd_pp: float = 4.0,
    horizon_sd_pp: float = 1.5,
) -> pd.DataFrame:
    if series.empty:
        raise ValueError("Series is empty.")

    last = series.sort_values("year").iloc[-1]
    last_year = int(last["year"])
    last_val = float(last["value_pct"])

    try:
        from scipy.stats import norm  # type: ignore
        z = float(norm.ppf(1 - alpha / 2))
    except Exception:
        z = 1.96

    rows = []
    for yr in years_forecast:
        yr = int(yr)
        h = max(yr - last_year, 1)
        se = float(base_sd_pp + horizon_sd_pp * np.sqrt(h))
        lo = last_val - z * se
        hi = last_val + z * se
        rows.append(
            dict(
                year=yr,
                trend_pred=float(_clamp_pct(np.array([last_val]))[0]),
                trend_se=se,
                trend_lo=float(_clamp_pct(np.array([lo]))[0]),
                trend_hi=float(_clamp_pct(np.array([hi]))[0]),
            )
        )
    return pd.DataFrame(rows)


def baseline_forecast_auto(
    series: pd.DataFrame,
    years_forecast: Sequence[int],
    alpha: float = 0.05,
    flat_base_sd_pp: float = 4.0,
    flat_horizon_sd_pp: float = 1.5,
) -> pd.DataFrame:
    if len(series) >= 2:
        return ols_trend_forecast(series, years_forecast, alpha=alpha)
    return flat_baseline_forecast(
        series, years_forecast, alpha=alpha, base_sd_pp=flat_base_sd_pp, horizon_sd_pp=flat_horizon_sd_pp
    )


def event_effects_by_year(
    df_event_effects_tidy: pd.DataFrame,
    indicator_code: str,
    years: Sequence[int],
    effect_col: str = "effect_pp",
) -> pd.DataFrame:
    df = df_event_effects_tidy.copy()
    df = df[df["indicator_code"].astype(str) == indicator_code]

    if "year" not in df.columns:
        raise ValueError("event_effects_tidy must contain column 'year'.")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[effect_col] = pd.to_numeric(df[effect_col], errors="coerce")

    agg = df.groupby("year", as_index=False)[effect_col].sum().rename(columns={effect_col: "event_effect_pp"})

    frame = pd.DataFrame({"year": [int(y) for y in years]})
    out = frame.merge(agg, on="year", how="left").fillna({"event_effect_pp": 0.0})
    return out


def event_effects_for_target(
    df_event_effects_tidy: pd.DataFrame,
    target_indicator_code: str,
    years: Sequence[int],
    *,
    usage_proxy_source_code: str = "USG_P2P_COUNT",
    usage_proxy_beta: float = 0.15,
) -> pd.DataFrame:
    """
    Get annual event effects for a target indicator.

    If target is USG_ACTIVE_RATE and there are no direct effects, we map
    effects from `usage_proxy_source_code` using a simple linear coefficient beta.

    Returns columns: year, event_effect_pp
    """
    direct = event_effects_by_year(df_event_effects_tidy, indicator_code=target_indicator_code, years=years)

    if target_indicator_code != "USG_ACTIVE_RATE":
        return direct

    # If direct effects exist and are non-zero, keep them (future-proof)
    if float(direct["event_effect_pp"].abs().sum()) > 0:
        return direct

    # Map from source usage series (typically count-based) into pp changes
    src = event_effects_by_year(df_event_effects_tidy, indicator_code=usage_proxy_source_code, years=years)
    mapped = src.copy()
    mapped["event_effect_pp"] = mapped["event_effect_pp"].astype(float) * float(usage_proxy_beta)
    return mapped


def forecast_event_augmented(
    trend_df: pd.DataFrame,
    event_df: pd.DataFrame,
    scenario: ScenarioSpec,
    alpha: float = 0.05,
    event_sd_frac: float = 0.35,
) -> pd.DataFrame:
    df = trend_df.merge(event_df, on="year", how="left")
    df["event_effect_pp"] = df["event_effect_pp"].fillna(0.0)

    trend_s = df["trend_pred"].to_numpy(dtype=float) * scenario.trend_multiplier
    event_s = df["event_effect_pp"].to_numpy(dtype=float) * scenario.event_multiplier
    pred = trend_s + event_s

    trend_se = df["trend_se"].to_numpy(dtype=float)
    event_sd = np.abs(event_s) * event_sd_frac
    total_se = np.sqrt(trend_se**2 + event_sd**2 + scenario.extra_sd_pp**2)

    try:
        from scipy.stats import t  # type: ignore
        tcrit = float(t.ppf(1 - alpha / 2, df=3))
    except Exception:
        tcrit = 2.0

    lo = pred - tcrit * total_se
    hi = pred + tcrit * total_se

    return pd.DataFrame(
        {
            "scenario": scenario.name,
            "year": df["year"].astype(int),
            "trend_pred_pp": _clamp_pct(trend_s),
            "event_effect_pp": event_s.astype(float),
            "pred_pp": _clamp_pct(pred),
            "lo_pp": _clamp_pct(lo),
            "hi_pp": _clamp_pct(hi),
            "se_pp": total_se.astype(float),
        }
    )


def top_event_contributors(
    df_event_effects_tidy: pd.DataFrame,
    indicator_code: str,
    year: int,
    k: int = 5,
    effect_col: str = "effect_pp",
) -> pd.DataFrame:
    df = df_event_effects_tidy.copy()
    df = df[(df["indicator_code"].astype(str) == indicator_code) & (pd.to_numeric(df["year"], errors="coerce") == year)]

    if df.empty:
        return pd.DataFrame(columns=["event_record_id", "event_name", "effect_pp"])

    df[effect_col] = pd.to_numeric(df[effect_col], errors="coerce")

    out = (
        df.groupby(["event_record_id", "event_name"], as_index=False)[effect_col]
        .sum()
        .rename(columns={effect_col: "effect_pp"})
    )
    out["abs_effect_pp"] = out["effect_pp"].abs()
    out = out.sort_values("abs_effect_pp", ascending=False).head(k)
    return out.drop(columns=["abs_effect_pp"])
