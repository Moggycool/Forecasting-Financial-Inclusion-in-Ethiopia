""" Overview page showing key KPIs, P2P/ATM ratio, and growth highlights. """
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable so `import dashboard...` works under `streamlit run`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from dashboard.utils.io import (
    load_forecast_table,
    load_enriched_observations,
)
from dashboard.utils.metrics import latest_and_delta, safe_ratio
from dashboard.utils.charts import multi_line, bar_sorted

st.title("Overview")

fc = load_forecast_table(ROOT)
enriched = load_enriched_observations(ROOT)


st.subheader("Key KPI summary (from Task 4 forecasts)")
scenario = st.selectbox("Scenario", ["pessimistic", "base", "optimistic"], index=1)

kpi_codes = ["ACC_OWNERSHIP", "USG_ACTIVE_RATE"]
cols = st.columns(2)

for i, code in enumerate(kpi_codes):
    g = fc[(fc["indicator_code"] == code) & (fc["scenario"] == scenario)].set_index("year")["pred_pp"]
    latest, delta = latest_and_delta(g)
    label = fc.loc[fc["indicator_code"] == code, "indicator_label"].iloc[0]
    with cols[i]:
        st.metric(
            label=f"{label} — {scenario} (latest)",
            value="N/A" if latest is None else f"{latest:.2f} pp",
            delta=None if delta is None else f"{delta:+.2f} pp vs prior year",
        )

st.divider()
st.subheader("P2P / ATM Crossover Ratio (from enriched dataset)")

st.caption(
    "This requires the enriched dataset to include time-series observations for P2P and ATM. "
    "If your column names differ, tell me what they are and I’ll map them."
)

# Heuristic: look for columns
# Expected long-ish schema: year, indicator_code, value (or obs_value)
value_col_candidates = [c for c in ["value", "obs_value", "indicator_value", "val"] if c in enriched.columns]
if "year" not in enriched.columns or "indicator_code" not in enriched.columns or not value_col_candidates:
    st.warning(
        "Cannot compute P2P/ATM ratio because enriched dataset lacks columns: "
        "`year`, `indicator_code`, and one of [value, obs_value, indicator_value, val]."
    )
else:
    vcol = value_col_candidates[0]

    p2p_code = st.text_input("P2P indicator_code", value="USG_P2P_COUNT")
    atm_code = st.text_input("ATM indicator_code", value="USG_ATM_COUNT")

    obs = enriched.dropna(subset=["year"]).copy()
    obs = obs[obs["indicator_code"].isin([p2p_code, atm_code])]
    pivot = obs.pivot_table(index="year", columns="indicator_code", values=vcol, aggfunc="mean").sort_index()

    if p2p_code in pivot.columns and atm_code in pivot.columns:
        ratio = safe_ratio(pivot[p2p_code], pivot[atm_code]).rename("P2P_ATM_ratio").reset_index()
        st.plotly_chart(
            multi_line(ratio, x="year", y="P2P_ATM_ratio", color=None, title="P2P/ATM Crossover Ratio over time"),
            use_container_width=True,
        )
        st.download_button(
            "Download P2P/ATM ratio CSV",
            data=ratio.to_csv(index=False).encode("utf-8"),
            file_name="p2p_atm_ratio.csv",
            mime="text/csv",
        )
    else:
        st.info("No overlapping P2P/ATM series found for those indicator codes in the enriched dataset.")

st.divider()
st.subheader("Growth highlights (Top YoY changes by indicator)")

if "year" in enriched.columns and "indicator_code" in enriched.columns and value_col_candidates:
    vcol = value_col_candidates[0]
    df = enriched.dropna(subset=["year"]).copy()
    df = df.sort_values(["indicator_code", "year"])
    df["yoy_pp"] = df.groupby("indicator_code")[vcol].diff()

    latest_year = int(df["year"].dropna().max()) if not df["year"].dropna().empty else None
    if latest_year is not None:
        latest = df[df["year"] == latest_year].dropna(subset=["yoy_pp"]).copy()
        top_n = st.slider("Top N indicators", 5, 30, 10)
        top = latest.groupby("indicator_code", as_index=False)["yoy_pp"].mean().sort_values("yoy_pp", ascending=False).head(top_n)

        st.plotly_chart(bar_sorted(top, x="indicator_code", y="yoy_pp", title=f"Top YoY changes (pp) — year {latest_year}"), use_container_width=True)
        st.download_button(
            "Download growth highlights CSV",
            data=top.to_csv(index=False).encode("utf-8"),
            file_name="growth_highlights.csv",
            mime="text/csv",
        )
else:
    st.warning("Growth highlights unavailable: enriched dataset missing year/indicator/value columns.")
