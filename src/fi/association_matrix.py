"""Functions for building and plotting event–indicator association matrices.

This module:
- builds an event × indicator matrix using `impact_magnitude_pp` (summed per event)
- renders a heatmap to file (headless; uses Matplotlib Agg backend)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EVENT_INDEX_COLS: list[str] = ["event_record_id", "event_name", "event_category", "event_date"]


def build_association_matrix(df_summary: pd.DataFrame, indicators: Sequence[str]) -> pd.DataFrame:
    """Build an event–indicator association matrix from an impact-links summary DataFrame.

    Parameters
    ----------
    df_summary:
        Expected to include:
        - event_record_id, event_name, event_category, event_date
        - indicator_code
        - impact_magnitude_pp
    indicators:
        Indicator codes to include as columns.

    Returns
    -------
    pd.DataFrame
        One row per event, with indicator columns filled (missing indicators filled with 0.0).
    """
    indicators_list = list(indicators)

    # If df_summary is missing expected structure, return an empty-but-well-formed frame
    required = set(EVENT_INDEX_COLS + ["indicator_code", "impact_magnitude_pp"])
    if df_summary is None or df_summary.empty or not required.issubset(df_summary.columns):
        cols = EVENT_INDEX_COLS + indicators_list
        return pd.DataFrame(columns=cols)

    df = df_summary.loc[df_summary["indicator_code"].isin(indicators_list)].copy()

    if df.empty:
        cols = EVENT_INDEX_COLS + indicators_list
        return pd.DataFrame(columns=cols)

    mat = (
        df.pivot_table(
            index=EVENT_INDEX_COLS,
            columns="indicator_code",
            values="impact_magnitude_pp",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )

    # Ensure all requested indicator columns exist (even if absent in the data)
    for c in indicators_list:
        if c not in mat.columns:
            mat[c] = 0.0

    # Stabilize column order: event index cols first, then indicators in the provided order
    return mat[EVENT_INDEX_COLS + indicators_list]


def plot_heatmap(mat: pd.DataFrame, key_indicators: Sequence[str], out_path: str) -> None:
    """Plot heatmap of an event–indicator association matrix to disk.

    Gracefully handles empty/degenerate inputs by writing a placeholder figure.

    Parameters
    ----------
    mat:
        DataFrame returned by `build_association_matrix`.
    key_indicators:
        Indicator columns to plot.
    out_path:
        Output image path.
    """
    indicators = list(key_indicators)

    def _write_placeholder(msg: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    if mat is None or mat.empty or not indicators:
        _write_placeholder("No association data to plot.")
        return

    missing_cols = [c for c in indicators if c not in mat.columns]
    if missing_cols:
        _write_placeholder(f"Missing indicator columns: {', '.join(missing_cols)}")
        return

    vals = mat[indicators].to_numpy(dtype=float, copy=True)
    if vals.size == 0 or vals.shape[0] == 0 or vals.shape[1] == 0:
        _write_placeholder("No association data to plot.")
        return

    finite = np.isfinite(vals)
    vmax = float(np.nanmax(np.abs(vals[finite]))) if finite.any() else 0.0
    if vmax == 0.0:
        vmax = 1.0  # avoid singular color scale

    # Build y-axis labels (prefer event_name, fallback to event_record_id)
    if "event_name" in mat.columns:
        ylabels = mat["event_name"].fillna("").astype(str).tolist()
    elif "event_record_id" in mat.columns:
        ylabels = mat["event_record_id"].fillna("").astype(str).tolist()
    else:
        ylabels = [f"event_{i+1}" for i in range(vals.shape[0])]

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(indicators)), max(4, 0.35 * len(ylabels))))
    im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_title("Event–Indicator Association (impact_magnitude_pp)")
    ax.set_xlabel("Indicator")
    ax.set_ylabel("Event")

    ax.set_xticks(np.arange(len(indicators)))
    ax.set_xticklabels(indicators, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Impact magnitude (pp)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)