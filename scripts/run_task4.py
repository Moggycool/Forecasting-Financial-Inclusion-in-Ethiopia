""" Script to run Task 4: Forecasting Access and Usage (2025–2027). """
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.fi.data_io import load_csv  # noqa: E402
from src.fi.forecasting import (  # noqa: E402
    DEFAULT_SCENARIOS,
    baseline_forecast_auto,
    event_effects_for_target,
    forecast_event_augmented,
    prepare_findex_series,
    top_event_contributors,
)

TARGETS = {
    "ACC_OWNERSHIP": "Account ownership (Access)",
    "USG_ACTIVE_RATE": "Active usage rate (Usage proxy)",
}
YEARS_FCST = [2025, 2026, 2027]

# Proxy linkage: map USG_P2P_COUNT event effects into USG_ACTIVE_RATE pp effects
USAGE_PROXY_SOURCE = "USG_P2P_COUNT"
USAGE_PROXY_BETA = 0.15  # 0.10–0.25 reasonable for sensitivity; 0.0 disables linkage


def main() -> None:
    root = _ROOT
    out_dir = root / "outputs" / "task_4"
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_path = root / "data" / "processed" / "eda_enriched" / "observations.csv"
    eff_path = root / "outputs" / "task_3" / "event_effects_tidy.csv"

    if not obs_path.exists():
        raise FileNotFoundError(f"[task4] Missing observations: {obs_path}")
    if not eff_path.exists():
        raise FileNotFoundError(f"[task4] Missing Task 3 effects: {eff_path}")

    df_obs = load_csv(str(obs_path))
    df_eff = load_csv(str(eff_path))

    all_fc = []
    all_top = []

    for code, label in TARGETS.items():
        series = prepare_findex_series(df_obs, indicator_code=code, gender="all", location="national")

        # robust baseline (OLS if possible; else flat carry-forward)
        trend = baseline_forecast_auto(
            series,
            YEARS_FCST,
            alpha=0.05,
            flat_base_sd_pp=4.0,
            flat_horizon_sd_pp=1.5,
        )

        ev = event_effects_for_target(
            df_eff,
            target_indicator_code=code,
            years=YEARS_FCST,
            usage_proxy_source_code=USAGE_PROXY_SOURCE,
            usage_proxy_beta=USAGE_PROXY_BETA,
        )

        for scen in DEFAULT_SCENARIOS:
            fc = forecast_event_augmented(trend, ev, scenario=scen)
            fc.insert(0, "indicator_code", code)
            fc.insert(1, "indicator_label", label)
            all_fc.append(fc)

        # For usage proxy, show contributors from the source series as the driver set
        if code == "USG_ACTIVE_RATE":
            top_code = USAGE_PROXY_SOURCE
        else:
            top_code = code

        top = top_event_contributors(df_eff, indicator_code=top_code, year=2027, k=5)
        top.insert(0, "indicator_code", code)
        top.insert(1, "indicator_label", label)
        top.insert(2, "contributors_from_indicator", top_code)
        all_top.append(top)

    forecast_table = pd.concat(all_fc, ignore_index=True)
    forecast_table.to_csv(out_dir / "forecast_table_task4.csv", index=False)

    top_table = pd.concat(all_top, ignore_index=True)
    top_table.to_csv(out_dir / "top_event_contributors_2027.csv", index=False)

    # Scenario plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (code, label) in zip(axes, TARGETS.items()):
        sub = forecast_table[forecast_table["indicator_code"] == code]
        for scen in ["pessimistic", "base", "optimistic"]:
            ssub = sub[sub["scenario"] == scen].sort_values("year")
            ax.plot(ssub["year"], ssub["pred_pp"], label=scen)
            ax.fill_between(ssub["year"], ssub["lo_pp"], ssub["hi_pp"], alpha=0.15)

        ax.set_title(label)
        ax.set_xlabel("Year")
        ax.set_ylabel("% of adults (Usage is a proxy rate)")
        ax.grid(True, alpha=0.3)

        if code == "USG_ACTIVE_RATE":
            ax.text(
                0.02, 0.02,
                f"Event linkage: {USAGE_PROXY_SOURCE} → {code}\nβ={USAGE_PROXY_BETA:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
            )

    axes[0].legend(loc="best")
    fig.suptitle("Task 4 Forecasts (2025–2027): scenarios with uncertainty bands")
    fig.tight_layout()
    fig.savefig(out_dir / "scenario_plot_access_usage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[task4] wrote: {out_dir / 'forecast_table_task4.csv'}")
    print(f"[task4] wrote: {out_dir / 'scenario_plot_access_usage.png'}")
    print(f"[task4] wrote: {out_dir / 'top_event_contributors_2027.csv'}")


if __name__ == "__main__":
    main()
