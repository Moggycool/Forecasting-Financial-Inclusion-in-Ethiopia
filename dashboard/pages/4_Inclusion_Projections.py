""" Inclusion Projections page for visualizing ACC_OWNERSHIP projections and progress toward targets. """
import streamlit as st
from pathlib import Path
import pandas as pd

from dashboard.utils.io import load_forecast_table
from dashboard.utils.charts import target_progress

ROOT = Path(__file__).resolve().parents[2]

st.title("Inclusion Projections")

fc = load_forecast_table(ROOT)

st.caption(
    "This page treats ACC_OWNERSHIP as the primary inclusion proxy (account ownership). "
    "It visualizes progress toward the 60% target under each scenario."
)

scenario = st.selectbox("Scenario", ["pessimistic", "base", "optimistic"], index=1)
target = st.slider("Target (%)", min_value=40, max_value=90, value=60, step=1)

acc = fc[(fc["indicator_code"] == "ACC_OWNERSHIP") & (fc["scenario"] == scenario)].copy()
if acc.empty:
    st.error("ACC_OWNERSHIP not found in forecast_table_task4.csv")
    st.stop()

fig = target_progress(
    acc, x="year", y="pred_pp", target=float(target),
    title=f"ACC_OWNERSHIP projection — {scenario} scenario"
)
st.plotly_chart(fig, use_container_width=True)

# Progress bar to target by final year
final_year = int(acc["year"].max())
final_val = float(acc.loc[acc["year"] == final_year, "pred_pp"].iloc[0])
progress = min(max(final_val / float(target), 0.0), 1.0)

st.subheader("Progress toward target")
st.write(f"Projected {final_year} inclusion: **{final_val:.2f}%** vs target **{target}%**")
st.progress(progress)

# Consortium key questions
st.subheader("Key questions (auto-answered)")

first_hit = acc[acc["pred_pp"] >= float(target)]
hit_year = int(first_hit["year"].iloc[0]) if not first_hit.empty else None

q1 = "Yes" if hit_year is not None else "No"
st.write(f"1) Will Ethiopia reach **{target}%** account ownership by {final_year} under **{scenario}**? **{q1}**")
st.write(f"2) If yes, first year reaching target: **{hit_year if hit_year is not None else 'Not reached in horizon'}**")
st.write(
    "3) What is the 2027 decomposition (Trend vs Events)? "
    f"Trend={float(acc.loc[acc['year']==final_year,'trend_pred_pp'].iloc[0]):.2f} pp, "
    f"Events={float(acc.loc[acc['year']==final_year,'event_effect_pp'].iloc[0]):.2f} pp, "
    f"Total={final_val:.2f} pp"
)

st.download_button(
    "Download inclusion projection CSV",
    data=acc.to_csv(index=False).encode("utf-8"),
    file_name=f"inclusion_projection_{scenario}.csv",
    mime="text/csv",
)