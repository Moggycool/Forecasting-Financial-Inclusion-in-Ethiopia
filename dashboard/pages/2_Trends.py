""" Trends page for interactive time-series exploration.
Assumes a long-format table with columns: year, indicator_code, and a numeric value column."""
import streamlit as st
from pathlib import Path
import pandas as pd

from dashboard.utils.io import load_enriched_observations
from dashboard.utils.charts import multi_line

ROOT = Path(__file__).resolve().parents[2]

st.title("Trends")
enriched = load_enriched_observations(ROOT)

st.caption(
    "Interactive time-series exploration from `ethiopia_fi_unified_data__enriched.csv`. "
    "This page assumes a long-format table with columns: `year`, `indicator_code`, and a numeric value column."
)

value_col_candidates = [c for c in ["value", "obs_value", "indicator_value", "val"] if c in enriched.columns]

if "year" not in enriched.columns or "indicator_code" not in enriched.columns or not value_col_candidates:
    st.error(
        "Missing required columns in enriched dataset. Need `year`, `indicator_code`, and one of "
        "[value, obs_value, indicator_value, val]."
    )
    st.stop()

vcol = value_col_candidates[0]
df = enriched.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)

indicator_list = sorted(df["indicator_code"].dropna().unique().tolist())

left, right = st.columns([2, 1])
with left:
    selected = st.multiselect(
        "Select indicators to compare (channel comparison view)",
        indicator_list,
        default=[c for c in ["USG_P2P_COUNT", "USG_ATM_COUNT"] if c in indicator_list][:2],
    )
with right:
    min_y, max_y = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider("Year range", min_y, max_y, (max(min_y, max_y - 10), max_y))

f = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
if selected:
    f = f[f["indicator_code"].isin(selected)]

# Aggregate if duplicates exist
plot_df = f.groupby(["year", "indicator_code"], as_index=False)[vcol].mean()

st.plotly_chart(
    multi_line(plot_df, x="year", y=vcol, color="indicator_code", title="Indicator trends (selected)"),
    use_container_width=True
)

st.download_button(
    "Download filtered trends CSV",
    data=plot_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_trends.csv",
    mime="text/csv",
)
