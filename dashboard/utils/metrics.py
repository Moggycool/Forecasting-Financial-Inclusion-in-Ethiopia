""" Metric utility functions for calculating latest values, deltas, and growth rates. """
from __future__ import annotations
import pandas as pd


def latest_and_delta(series_by_year: pd.Series) -> tuple[float | None, float | None]:
    """
    series_by_year: index=year, values=numeric
    Returns (latest_value, delta_vs_prev_year)
    """
    series_by_year = series_by_year.dropna()
    if series_by_year.empty:
        return None, None

    series_by_year = series_by_year.sort_index()
    latest_year = series_by_year.index.max()
    latest_val = float(series_by_year.loc[latest_year])

    prev_year = int(latest_year) - 1
    if prev_year in series_by_year.index:
        delta = float(latest_val - float(series_by_year.loc[prev_year]))
    else:
        delta = None
    return latest_val, delta


def yoy_growth(df: pd.DataFrame, year_col: str, value_col: str) -> pd.DataFrame:
    """Returns yoy growth (%), by any grouping already present in df."""
    out = df.sort_values(year_col).copy()
    out["yoy_growth_pct"] = out[value_col].pct_change() * 100.0
    return out


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Computes num / den, returning NaN where den is zero or missing."""
    den = den.replace(0, pd.NA)
    return num / den
