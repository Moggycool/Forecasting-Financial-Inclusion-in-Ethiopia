""" Script to run exploratory data analysis (EDA) on unified FI dataset. """
from __future__ import annotations

# --- add root to sys.path ---
import sys
from pathlib import Path
import argparse
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# --- end block ---

from src.fi.validation import assert_min_schema
from src.fi.explore import counts, temporal_range, indicator_coverage, list_events, list_impact_links
from src.fi.io import load_csv, save_csv


DEFAULT_IN = Path("data/raw/ethiopia_fi_unified_data.csv")
DEFAULT_OUT_DIR = Path("data/processed/eda")


def _ensure_dir(d: Path) -> None:
    """Ensure directory exists."""
    d.mkdir(parents=True, exist_ok=True)


def _save_kv_as_csv(d: Path, name: str, kv: dict) -> None:
    """Save a key-value dict as a one-row CSV."""
    df = pd.DataFrame([kv])
    save_csv(df, str(d / f"{name}.csv"))


def _norm_series(s: pd.Series) -> pd.Series:
    """Normalize a string series for robust filtering."""
    return s.fillna("").astype(str).str.strip().str.lower()


def _subset_by_record_type(df: pd.DataFrame, rt: str) -> pd.DataFrame:
    """Return subset filtered by record_type (case/space-insensitive)."""
    rts = _norm_series(df["record_type"])
    return df.loc[rts.eq(rt)].copy()


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(description="Run EDA tables for unified FI dataset.")
    ap.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(DEFAULT_IN),
        help="Input unified CSV (raw or enriched)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory to write EDA outputs",
    )
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    print(f"[load] {in_path}")
    df = load_csv(str(in_path))
    assert_min_schema(df)

    # 1) Counts
    print("[eda] counts: record_type, pillar, category")
    save_csv(counts(df, "record_type"), str(out_dir / "counts__record_type.csv"))
    save_csv(counts(df, "pillar"), str(out_dir / "counts__pillar.csv"))
    if "category" in df.columns:
        save_csv(counts(df, "category"), str(out_dir / "counts__category.csv"))

    # 2) Temporal range (overall + by record_type)
    print("[eda] temporal range (overall + by record_type)")
    _save_kv_as_csv(out_dir, "temporal_range", temporal_range(df, "observation_date"))

    df_obs = _subset_by_record_type(df, "observation")
    df_tgt = _subset_by_record_type(df, "target")
    df_evt = _subset_by_record_type(df, "event")

    if len(df_obs) > 0:
        _save_kv_as_csv(
            out_dir,
            "temporal_range__observations",
            temporal_range(df_obs, "observation_date"),
        )
    else:
        print("[eda] note: no observation records found for temporal_range__observations")

    if len(df_tgt) > 0:
        _save_kv_as_csv(
            out_dir,
            "temporal_range__targets",
            temporal_range(df_tgt, "observation_date"),
        )
    else:
        print("[eda] note: no target records found for temporal_range__targets")

    if len(df_evt) > 0:
        _save_kv_as_csv(
            out_dir,
            "temporal_range__events",
            temporal_range(df_evt, "observation_date"),
        )
    else:
        print("[eda] note: no event records found for temporal_range__events")

    # 3) Indicator coverage (observations only)
    print("[eda] indicator coverage")
    cov = indicator_coverage(df)
    save_csv(cov, str(out_dir / "indicator_coverage.csv"))

    # 4) Events and impact links tables
    print("[eda] events list")
    save_csv(list_events(df), str(out_dir / "events.csv"))

    print("[eda] impact_links list")
    save_csv(list_impact_links(df), str(out_dir / "impact_links.csv"))

    print(f"[done] wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
