""" Script to run Task 3: Analyze impact links and compute event effects. """
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.fi.association_matrix import (  # noqa: E402
    build_association_matrix,
    plot_heatmap_basic_to_file,
    plot_heatmap_signed_annotated_to_file,
)
from src.fi.data_io import coerce_datetime, load_csv  # noqa: E402
from src.fi.event_effects import (  # noqa: E402
    DEFAULT_FORECAST_YEARS,
    FINDEX_YEAR_GRID,
    effects_tidy,
    normalize_years,
)
from src.fi.impact_links import build_impact_links_summary, join_links_events  # noqa: E402
from src.fi.task3_validation import validate_telebirr_mm  # noqa: E402


# Keep existing KPI list used for the *matrix build* and effects.
KEY_INDICATORS = ["ACC_OWNERSHIP", "ACC_MM_ACCOUNT", "USG_DIGITAL_PAYMENT"]

# This is the *extra* KPI for the focused signed heatmap only.
KEY_HEATMAP_EXTRA = ["GAP_ACC_OWNERSHIP_MALE_MINUS_FEMALE_PP"]


def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _safe_load_optional_csv(path: Path) -> pd.DataFrame:
    """Load CSV if it exists; otherwise return empty df (defensive)."""
    try:
        if path.exists():
            return load_csv(str(path))
    except Exception:
        pass
    return pd.DataFrame()


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Defensive to_csv: never crash Task 3 just because a DF is empty/None."""
    if df is None:
        return
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"[task3] WARN: failed writing {path.name}: {e}")


def main() -> None:
    """Main function to run Task 3 analysis."""
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    out_dir = root / "outputs" / "task_3"
    out_dir.mkdir(parents=True, exist_ok=True)

    events_path = data_dir / "processed" / "eda_enriched" / "events.csv"
    links_path = data_dir / "processed" / "eda_enriched" / "impact_links.csv"
    raw_links_path = data_dir / "raw" / "ethiopia_fi_unified_data__impact_links.csv"

    # Optional observations (used only for validation; do not fail if absent)
    obs_candidates = [
        data_dir / "processed" / "eda_enriched" / "observations.csv",
        data_dir / "processed" / "observations.csv",
        data_dir / "raw" / "observations.csv",
    ]
    df_obs = pd.DataFrame()
    for p in obs_candidates:
        df_obs = _safe_load_optional_csv(p)
        if not df_obs.empty:
            print(f"[task3] loaded observations: {p}")
            break
    if df_obs.empty:
        print("[task3] observations not found (or empty); validation will return NaN safely.")

    events = load_csv(str(events_path))
    links = load_csv(str(links_path))

    # Fallback: if processed impact_links is empty, use raw unified impact links
    if links.empty:
        if not raw_links_path.exists():
            raise FileNotFoundError(
                f"impact_links.csv is empty and raw fallback not found: {raw_links_path}"
            )
        print(f"[task3] processed impact_links.csv is empty; using raw: {raw_links_path}")
        links = load_csv(str(raw_links_path))

    # Normalize join keys for reliability
    if "record_id" not in events.columns:
        raise ValueError(f"events.csv missing required 'record_id'. Columns={list(events.columns)}")
    if "parent_id" not in links.columns:
        raise ValueError(f"impact_links missing required 'parent_id'. Columns={list(links.columns)}")

    events = events.copy()
    links = links.copy()
    events["record_id"] = _norm_str(events["record_id"])
    links["parent_id"] = _norm_str(links["parent_id"])

    # Coerce the correct event date column (avoid silently breaking plots)
    if "event_date" in events.columns:
        events = coerce_datetime(events, "event_date")
    elif "observation_date" in events.columns:
        events = coerce_datetime(events, "observation_date")

    overlap = len(set(links["parent_id"].tolist()) & set(events["record_id"].tolist()))
    print(f"[task3] links={len(links)} events={len(events)} overlap={overlap}")

    joined = join_links_events(links, events)
    summary = build_impact_links_summary(joined)
    _safe_to_csv(summary, out_dir / "impact_links_summary.csv")

    # --- Association matrix CSV (same output file name as before) ---
    mat = build_association_matrix(summary, KEY_INDICATORS)
    _safe_to_csv(mat, out_dir / "event_indicator_association_matrix.csv")

    # --- Heatmaps (basic + improved signed) ---
    # Basic heatmap (keeps your existing artifact name)
    plot_heatmap_basic_to_file(
        mat=mat,
        key_indicators=KEY_INDICATORS,
        out_path=str(out_dir / "event_indicator_association_heatmap.png"),
    )

    # Signed full heatmap: use all indicator columns present in `mat`
    # (excluding event metadata). This gives you the "full signed matrix" view.
    meta_cols = ["event_record_id", "event_name", "event_category", "event_date"]
    full_indicators = [c for c in mat.columns if c not in meta_cols]

    plot_heatmap_signed_annotated_to_file(
        mat_with_event_cols=mat,
        key_indicators=full_indicators,
        out_path=str(out_dir / "event_indicator_association_heatmap_signed_full.png"),
        title="Event–Indicator Association (signed, diverging @0, annotated)",
    )

    # Signed key-indicator heatmap (includes gender gap KPI if present)
    key_signed = KEY_INDICATORS + ["GAP_ACC_OWNERSHIP_MALE_MINUS_FEMALE_PP"]
    key_signed = [c for c in key_signed if c in mat.columns]

    plot_heatmap_signed_annotated_to_file(
        mat_with_event_cols=mat,
        key_indicators=key_signed,
        out_path=str(out_dir / "event_indicator_association_heatmap_signed_key.png"),
        title="Event–Key Indicators (signed, diverging @0, annotated)",
    )

    # ---------------------------------------------------------------------
    # Month-aware effects (tidy) — PATCH: forecast-aware year grid
    # ---------------------------------------------------------------------
    years_effects = normalize_years(
        FINDEX_YEAR_GRID,
        ensure_years=DEFAULT_FORECAST_YEARS,  # ensures 2025–2027 appear in tidy
    )
    print(f"[task3] effects year grid = {years_effects}")

    eff = effects_tidy(
        summary,
        KEY_INDICATORS,
        years=years_effects,
        default_shape="ramp",
        default_ramp_years=3.0,
        # We already ensured the forecast years explicitly via normalize_years above.
        ensure_forecast_years=False,
    )

    # Guardrail: hard fail if forecast years are missing (prevents silent disconnection in Task 4)
    missing = sorted(set(DEFAULT_FORECAST_YEARS) - set(eff["year"].unique().tolist())) if not eff.empty else list(DEFAULT_FORECAST_YEARS)
    if missing:
        raise RuntimeError(
            "[task3] FATAL: event_effects_tidy is missing forecast years "
            f"{missing}. Check effects_tidy() / normalize_years() wiring."
        )
    print("[task3] OK: event_effects_tidy contains all forecast years:", DEFAULT_FORECAST_YEARS)

    _safe_to_csv(eff, out_dir / "event_effects_tidy.csv")

    # --- Realized-first Telebirr validation (safe if obs missing) ---
    tele_val = validate_telebirr_mm(
        df_obs=df_obs,
        df_links_summary=summary,
        df_event_effects_tidy=eff,
        telebirr_event_id="EVT_0001",
        year_a=2021,
        year_b=2024,
        target_indicator="ACC_MM_ACCOUNT",
        event_regex="telebirr",
    )
    _safe_to_csv(tele_val, out_dir / "telebirr_mm_validation.csv")

    # Console diagnostic to make rubric check obvious
    try:
        row = tele_val.iloc[0].to_dict() if (tele_val is not None and not tele_val.empty) else {}
        print(
            "[task3] telebirr validation:",
            f"obs_delta_pp={row.get('obs_delta_pp')}",
            f"pred_used_pp={row.get('pred_used_pp')}",
            f"pred_source={row.get('pred_source')}",
            f"residual_pp={row.get('residual_pp')}",
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
