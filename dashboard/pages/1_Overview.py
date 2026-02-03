""" Overview page showing key KPIs, P2P/ATM ratio, and growth highlights. """
from __future__ import annotations
import streamlit as st
import pandas as pd

import sys
from pathlib import Path

# Ensure repo root is importable so `import dashboard...` works under `streamlit run`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.io import load_forecast_table, load_enriched_observations
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
    "This requires the enriched dataset to include time-series observations for both indicators "
    "in overlapping years. Use filters below if your dataset contains multiple slices (location/gender)."
)

required_cols = {"year", "indicator_code", "value"}
if not required_cols.issubset(set(enriched.columns)):
    st.warning(f"Cannot compute P2P/ATM ratio because enriched dataset lacks required columns: {sorted(required_cols)}")
else:
    p2p_code = st.text_input("P2P indicator_code", value="USG_P2P_COUNT")
    atm_code = st.text_input("ATM indicator_code", value="USG_ATM_COUNT")

    # Optional slicers (only if columns exist)
    f = enriched.dropna(subset=["year", "indicator_code", "value"]).copy()
    f["year"] = f["year"].astype(int)

    slicer_cols = []
    left, right = st.columns(2)

    if "location" in f.columns:
        with left:
            locs = ["(all)"] + sorted([x for x in f["location"].dropna().unique().tolist()])
            loc_sel = st.selectbox("Location filter", locs, index=0)
        if loc_sel != "(all)":
            f = f[f["location"] == loc_sel]
        slicer_cols.append("location")

    if "gender" in f.columns:
        with right:
            gens = ["(all)"] + sorted([x for x in f["gender"].dropna().unique().tolist()])
            gen_sel = st.selectbox("Gender filter", gens, index=0)
        if gen_sel != "(all)":
            f = f[f["gender"] == gen_sel]
        slicer_cols.append("gender")

    obs = f[f["indicator_code"].isin([p2p_code, atm_code])].copy()

    # Diagnostics: existence
    present_codes = set(obs["indicator_code"].unique().tolist())
    if p2p_code not in present_codes:
        st.warning(f"P2P code not found in filtered observations: {p2p_code}")
    if atm_code not in present_codes:
        st.warning(f"ATM code not found in filtered observations: {atm_code}")

    pivot = (
        obs.pivot_table(index="year", columns="indicator_code", values="value", aggfunc="mean")
        .sort_index()
    )

    if p2p_code in pivot.columns and atm_code in pivot.columns:
        overlap_years = pivot[[p2p_code, atm_code]].dropna().shape[0]
        if overlap_years == 0:
            st.info(
                "Both series exist, but have **zero overlapping years** after filtering. "
                "Try changing the location/gender filters or indicator codes."
            )
        else:
            st.caption(f"Overlapping years available: {overlap_years}")
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

if not required_cols.issubset(set(enriched.columns)):
    st.warning("Growth highlights unavailable: enriched dataset missing year/indicator/value columns.")
else:
    df = enriched.dropna(subset=["year", "indicator_code", "value"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["indicator_code", "year"])

    # Restrict to percentage-type indicators to make YoY interpretable as pp.
    # Primary rule: unit == '%' OR value_type contains 'percent'
    is_pct = pd.Series(True, index=df.index)

    if "unit" in df.columns:
        is_pct = is_pct & (df["unit"].astype(str).str.strip() == "%")

    if "value_type" in df.columns:
        vt = df["value_type"].astype(str).str.lower()
        # if unit is missing/unreliable, allow value_type to drive filtering
        is_pct = is_pct | vt.str.contains("percent")

    pct_df = df[is_pct].copy()

    if pct_df.empty:
        st.info(
            "No percentage-type observation series found for Growth Highlights "
            "(expected unit='%' and/or value_type containing 'percent'). "
            "So YoY 'pp' highlights are not shown."
        )
    else:
        pct_df["yoy_pp"] = pct_df.groupby("indicator_code")["value"].diff()

        latest_year = int(pct_df["year"].max())
        latest = pct_df[pct_df["year"] == latest_year].dropna(subset=["yoy_pp"]).copy()

        if latest.empty:
            st.info(f"No YoY deltas available for year {latest_year} (may be single-year series).")
        else:
            top_n = st.slider("Top N indicators", 5, 30, 10)
            top = (
                latest.groupby("indicator_code", as_index=False)["yoy_pp"]
                .mean()
                .sort_values("yoy_pp", ascending=False)
                .head(top_n)
            )

            st.plotly_chart(
                bar_sorted(top, x="indicator_code", y="yoy_pp", title=f"Top YoY changes (pp) — year {latest_year}"),
                use_container_width=True,
            )
            st.download_button(
                "Download growth highlights CSV",
                data=top.to_csv(index=False).encode("utf-8"),
                file_name="growth_highlights.csv",
                mime="text/csv",
            )

            with st.expander("How this is computed"):
                st.write(
                    "YoY changes here are computed **only for percentage-type indicators** "
                    "(unit='%' and/or value_type contains 'percent') so the result is interpretable as **percentage points (pp)**."
                )
