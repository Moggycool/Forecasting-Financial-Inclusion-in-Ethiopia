""" Trends page for interactive time-series exploration. """
import streamlit as st
import pandas as pd

import sys
from pathlib import Path

# Ensure repo root is importable so `import dashboard...` works under `streamlit run`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.io import load_enriched_observations
from dashboard.utils.charts import multi_line

st.title("Trends")
enriched = load_enriched_observations(ROOT)

st.caption(
    "Interactive time-series exploration from `ethiopia_fi_unified_data__enriched.csv` "
    "(observations only). Uses standardized columns: `year`, `indicator_code`, `value`."
)

required = {"year", "indicator_code", "value"}
if not required.issubset(enriched.columns):
    st.error(f"Missing required columns in enriched dataset. Need {sorted(required)}.")
    st.stop()

df = enriched.dropna(subset=["year", "indicator_code", "value"]).copy()
df["year"] = df["year"].astype(int)

# Optional slicers if present
filters = {}
filter_cols = [c for c in ["unit", "value_type", "location", "gender", "pillar"] if c in df.columns]
if filter_cols:
    with st.expander("Filters"):
        cols = st.columns(min(3, len(filter_cols)))
        for i, c in enumerate(filter_cols):
            with cols[i % len(cols)]:
                opts = ["(all)"] + sorted([x for x in df[c].dropna().unique().tolist()])
                sel = st.selectbox(f"{c}", opts, index=0, key=f"trend_filter_{c}")
                if sel != "(all)":
                    filters[c] = sel

for c, sel in filters.items():
    df = df[df[c] == sel]

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

plot_df = f.groupby(["year", "indicator_code"], as_index=False)["value"].mean()

st.plotly_chart(
    multi_line(plot_df, x="year", y="value", color="indicator_code", title="Indicator trends (selected)"),
    use_container_width=True,
)

st.download_button(
    "Download filtered trends CSV",
    data=plot_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_trends.csv",
    mime="text/csv",
)
