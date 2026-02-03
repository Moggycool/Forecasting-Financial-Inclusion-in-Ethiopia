""" Main dashboard app for Ethiopia Financial Inclusion. """
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Ethiopia Financial Inclusion Dashboard",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]

st.title("Ethiopia Financial Inclusion — Interactive Dashboard")
st.caption(
    "Explore trends, event impacts, and 2025–2027 scenario forecasts for key financial inclusion KPIs."
)

with st.expander("How to use this dashboard", expanded=True):
    st.markdown(
        """
**Navigation:** Use the left sidebar to move between pages (Overview, Trends, Forecasts, Inclusion Projections).

**Data sources (local CSVs):**
- Task 4 forecasts: `outputs/task_4/forecast_table_task4.csv`
- Task 3 event effects: `outputs/task_3/event_effects_tidy.csv`
- Top contributors: `outputs/task_4/top_event_contributors_2027.csv`
- Unified enriched dataset (observations): `data/processed/ethiopia_fi_unified_data__enriched.csv`

If a chart looks empty, it usually means the expected columns/indicator codes are not present in your enriched dataset.
        """
    )

st.info(
    "Open a page from the sidebar. If you get a file/column error, tell me the exact traceback "
    "and the relevant CSV column names, and I’ll patch the loader to match your schema."
)
