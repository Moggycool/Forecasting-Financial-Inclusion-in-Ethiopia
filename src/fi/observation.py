from pathlib import Path
import pandas as pd

root = Path(r"D:\Python\Week10\Forecasting-Financial-Inclusion-in-Ethiopia")
src = root / "data" / "raw" / "ethiopia_fi_unified_data.csv"
out = root / "data" / "processed" / "eda_enriched" / "observations.csv"
out.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)

# Keep only observation rows
if "record_type" in df.columns:
    df = df[df["record_type"].astype(str).str.lower().eq("observation")].copy()

# Build the minimal observations table expected by get_observed(...)
obs = pd.DataFrame(
    {
        "record_id": df.get("record_id"),
        "indicator_code": df.get("indicator_code").fillna(df.get("related_indicator")),
        # Prefer fiscal_year (it’s populated in your data), fallback to parsed year from observation_date
        "year": df.get("fiscal_year"),
        "value_numeric": pd.to_numeric(df.get("value_numeric"), errors="coerce"),
        "gender": df.get("gender", "all"),
        "location": df.get("location", "national"),
        "observation_date": df.get("observation_date"),
        "source_name": df.get("source_name"),
        "confidence": df.get("confidence"),
    }
)

# If fiscal_year is missing for some rows, derive year from observation_date
missing_year = obs["year"].isna()
if missing_year.any() and "observation_date" in obs.columns:
    obs.loc[missing_year, "year"] = pd.to_datetime(
        obs.loc[missing_year, "observation_date"], errors="coerce"
    ).dt.year

# Normalize year to integer where possible
obs["year"] = (
    obs["year"].astype("string").str.extract(r"(\d{4})", expand=False).astype("Int64")
)

# Drop rows that still don’t have the essentials
obs = obs.dropna(subset=["indicator_code", "year", "value_numeric"]).copy()

obs.to_csv(out, index=False)
print(f"Wrote {len(obs)} rows -> {out}")
print(obs.head(10))