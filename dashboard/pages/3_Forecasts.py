""" Forecasts page for visualizing forecasted indicators with confidence intervals, milestones, and top event contributors. """
import streamlit as st
import pandas as pd

import sys
from pathlib import Path

# Ensure repo root is importable so `import dashboard...` works under `streamlit run`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.io import load_forecast_table, load_top_contributors
from dashboard.utils.charts import line_with_ci, bar_sorted

st.title("Forecasts (2025–2027)")
fc = load_forecast_table(ROOT)
contributors = load_top_contributors(ROOT)

fc["year"] = fc["year"].astype(int)

indicator_codes = sorted(fc["indicator_code"].unique().tolist())

left, right = st.columns([2, 1])
with left:
    default_idx = indicator_codes.index("ACC_OWNERSHIP") if "ACC_OWNERSHIP" in indicator_codes else 0
    indicator = st.selectbox("Indicator", indicator_codes, index=default_idx)
with right:
    model = st.selectbox(
        "Model selection",
        ["Baseline + Event Effects (Task 4)"],
        index=0
    )

st.caption("Confidence intervals come from `lo_pp` / `hi_pp` in your Task 4 output.")

plot_df = fc[fc["indicator_code"] == indicator].copy()

# Validate required columns for plotting
need_cols = {"year", "pred_pp", "lo_pp", "hi_pp", "scenario"}
missing = [c for c in need_cols if c not in plot_df.columns]
if missing:
    st.error(f"Forecast table missing columns needed for chart: {missing}")
    st.stop()

fig = line_with_ci(
    plot_df,
    x="year",
    y="pred_pp",
    lo="lo_pp",
    hi="hi_pp",
    color="scenario",
    title=f"{indicator} — Forecast with confidence intervals"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Projected milestones")
target = st.number_input("Milestone target (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

milestones = []
for scn, g in plot_df.groupby("scenario"):
    g = g.sort_values("year")
    hit = g[g["pred_pp"] >= target]
    hit_year = int(hit["year"].iloc[0]) if not hit.empty else None
    final_year = int(g["year"].max())
    milestones.append({
        "scenario": scn,
        "first_year_reaching_target": hit_year,
        "pred_pp_final": float(g[g["year"] == final_year]["pred_pp"].iloc[0]),
        "event_effect_pp_final": float(g[g["year"] == final_year]["event_effect_pp"].iloc[0]) if "event_effect_pp" in g.columns else None,
        "trend_pred_pp_final": float(g[g["year"] == final_year]["trend_pred_pp"].iloc[0]) if "trend_pred_pp" in g.columns else None,
        "final_year": final_year,
    })

milestones_df = pd.DataFrame(milestones).sort_values("scenario")
st.dataframe(milestones_df, use_container_width=True)

st.download_button(
    "Download forecast (filtered) CSV",
    data=plot_df.to_csv(index=False).encode("utf-8"),
    file_name=f"forecast_{indicator}.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Top event contributors (2027)")

c = contributors[contributors["indicator_code"] == indicator].copy()
if c.empty:
    st.info("No contributor rows found for this indicator in top_event_contributors_2027.csv")
else:
    st.plotly_chart(
        bar_sorted(c, x="event_name", y="effect_pp", title="Top contributors — effect_pp (as stored in contributors table)"),
        use_container_width=True
    )
    st.download_button(
        "Download contributors CSV",
        data=c.to_csv(index=False).encode("utf-8"),
        file_name=f"contributors_2027_{indicator}.csv",
        mime="text/csv",
    )

st.caption(
    "Note: For USG_ACTIVE_RATE, the contributor table may represent source-indicator effects "
    "(e.g., USG_P2P_COUNT). If you want, we can add a 'mapped_effect_pp' column to reconcile with beta=0.15."
)
