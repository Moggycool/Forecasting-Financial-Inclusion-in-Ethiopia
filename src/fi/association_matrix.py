"""
Functions for building and plotting event–indicator association matrices.

This module:
- builds an event × indicator matrix using `impact_magnitude_pp` (summed per event)
- renders heatmaps to file (headless; Matplotlib Agg backend)
- also supports inline (notebook) plotting

Improved Signed Heatmaps (audit-friendly):
- diverging colormap centered at 0
- symmetric color limits (±max abs)
- cell annotations with explicit sign (+/-) in percentage points

Conventions:
- Values are interpreted as *signed impacts in percentage points (pp)*.
- If your upstream table stores unsigned magnitudes plus `direction_sign`, apply the sign
  before plotting, or use `apply_direction_sign()` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib

# Headless-safe default for library usage; notebooks can still display figures because
# we'll return fig/ax and/or save to disk.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

EVENT_INDEX_COLS: list[str] = ["event_record_id", "event_name", "event_category", "event_date"]


@dataclass(frozen=True)
class HeatmapStyle:
    """Styling options for improved signed heatmaps."""
    cmap: str = "RdBu_r"           # diverging
    center: float = 0.0            # center at 0 for signed impacts
    annotate: bool = True
    fmt: str = "{:+.2f}"           # explicit sign
    fontsize_x: int = 8
    fontsize_y: int = 8
    annotation_fontsize: int = 7
    colorbar_label: str = "Estimated effect (pp), signed"


def apply_direction_sign(
    df: pd.DataFrame,
    magnitude_col: str = "impact_magnitude_pp",
    sign_col: str = "direction_sign",
    out_col: str = "impact_magnitude_pp_signed",
) -> pd.DataFrame:
    """
    If you have unsigned magnitudes + a direction_sign column (e.g., -1/+1),
    create a signed magnitude column.

    - If direction_sign is missing/NA, falls back to magnitude as-is.
    - Non-numeric values become NaN.
    """
    out = df.copy()
    if magnitude_col not in out.columns:
        out[out_col] = pd.NA
        return out

    mag = pd.to_numeric(out[magnitude_col], errors="coerce")

    if sign_col in out.columns:
        sgn = pd.to_numeric(out[sign_col], errors="coerce")
        # where sign is finite, apply it; else leave magnitude unchanged
        signed = np.where(np.isfinite(sgn.to_numpy()), mag.to_numpy() * sgn.to_numpy(), mag.to_numpy())
        out[out_col] = signed
    else:
        out[out_col] = mag

    return out


def build_association_matrix(df_summary: pd.DataFrame, indicators: Sequence[str]) -> pd.DataFrame:
    """Build an event–indicator association matrix from an impact-links summary DataFrame."""
    indicators_list = list(indicators)

    if df_summary is None or df_summary.empty:
        cols = EVENT_INDEX_COLS + indicators_list
        return pd.DataFrame(columns=cols)

    # Allow partial schemas (common during early joins); fill missing index cols with NA
    df = df_summary.copy()
    for c in EVENT_INDEX_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    required = set(EVENT_INDEX_COLS + ["indicator_code", "impact_magnitude_pp"])
    if not required.issubset(df.columns):
        cols = EVENT_INDEX_COLS + indicators_list
        return pd.DataFrame(columns=cols)

    df = df.loc[df["indicator_code"].isin(indicators_list)].copy()
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

    for c in indicators_list:
        if c not in mat.columns:
            mat[c] = 0.0

    return mat[EVENT_INDEX_COLS + indicators_list]


def _compute_symmetric_vlim(vals: np.ndarray, fallback: float = 1.0) -> float:
    finite = np.isfinite(vals)
    if not finite.any():
        return fallback
    v = float(np.nanmax(np.abs(vals[finite])))
    return v if v > 0 else fallback


def _default_figsize(n_rows: int, n_cols: int) -> tuple[float, float]:
    # Tuned for readability; adjust if you have many events
    w = min(24, 1.1 * n_cols + 5)
    h = min(18, 0.45 * n_rows + 4)
    return (w, h)


def _event_ylabels(mat_with_event_cols: pd.DataFrame) -> list[str]:
    """Human-readable y labels: event_name (YYYY-MM-DD) if available."""
    if "event_name" in mat_with_event_cols.columns:
        name = mat_with_event_cols["event_name"].fillna("").astype(str)
    elif "event_record_id" in mat_with_event_cols.columns:
        name = mat_with_event_cols["event_record_id"].fillna("").astype(str)
    else:
        name = pd.Series([f"event_{i+1}" for i in range(len(mat_with_event_cols))])

    if "event_date" in mat_with_event_cols.columns:
        dt = pd.to_datetime(mat_with_event_cols["event_date"], errors="coerce")
        dt_str = dt.dt.strftime("%Y-%m-%d").fillna("")
        return [f"{n} ({d})".strip() if d else str(n) for n, d in zip(name, dt_str)]

    return name.tolist()


# ---------------------------------------------------------------------
# BASIC heatmap (to mirror earlier behavior)
# ---------------------------------------------------------------------
def plot_heatmap_basic_to_file(mat: pd.DataFrame, key_indicators: Sequence[str], out_path: str) -> None:
    """Plot heatmap (basic) to disk. Mirrors earlier behavior (not centered/diverging/annotated)."""
    indicators = list(key_indicators)

    def _write_placeholder(msg: str) -> None:
        """Write a placeholder figure with a message."""
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

    ylabels = _event_ylabels(mat)

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(indicators)), max(4, 0.35 * len(ylabels))))
    im = ax.imshow(vals, aspect="auto", cmap="viridis")

    ax.set_title("Event–Indicator Association (basic)")
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


# ---------------------------------------------------------------------
# IMPROVED signed + annotated heatmaps
# ---------------------------------------------------------------------
def plot_heatmap_signed_annotated_to_file(
    mat_with_event_cols: pd.DataFrame,
    key_indicators: Sequence[str],
    out_path: str,
    title: str = "Event–Indicator Association (signed impacts, pp)",
    style: HeatmapStyle = HeatmapStyle(),
) -> None:
    """Plot improved signed heatmap to disk (diverging, centered at 0, annotated)."""
    indicators = list(key_indicators)

    def _write_placeholder(msg: str) -> None:
        """Write a placeholder figure with a message."""
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    if mat_with_event_cols is None or mat_with_event_cols.empty or not indicators:
        _write_placeholder("No association data to plot.")
        return

    missing_cols = [c for c in indicators if c not in mat_with_event_cols.columns]
    if missing_cols:
        _write_placeholder(f"Missing indicator columns: {', '.join(missing_cols)}")
        return

    vals_df = mat_with_event_cols[indicators].apply(pd.to_numeric, errors="coerce")
    vals = vals_df.to_numpy(dtype=float, copy=True)

    if vals.size == 0 or vals.shape[0] == 0 or vals.shape[1] == 0:
        _write_placeholder("No association data to plot.")
        return

    vlim = _compute_symmetric_vlim(vals)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=style.center, vmax=+vlim)

    ylabels = _event_ylabels(mat_with_event_cols)

    fig, ax = plt.subplots(figsize=_default_figsize(len(ylabels), len(indicators)))
    im = ax.imshow(vals, aspect="auto", cmap=style.cmap, norm=norm)

    ax.set_title(title)
    ax.set_xlabel("Indicator")
    ax.set_ylabel("Event")

    ax.set_xticks(np.arange(len(indicators)))
    ax.set_xticklabels(indicators, rotation=45, ha="right", fontsize=style.fontsize_x)

    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=style.fontsize_y)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(style.colorbar_label)

    if style.annotate:
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isfinite(v):
                    continue
                color = "white" if abs(v) >= 0.60 * vlim else "black"
                ax.text(
                    j,
                    i,
                    style.fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=style.annotation_fontsize,
                    color=color,
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_signed_annotated_inline(
    assoc_matrix: pd.DataFrame,
    title: str = "Event–Indicator Association (signed impacts, pp)",
    style: HeatmapStyle = HeatmapStyle(),
    vlim: float | None = None,
    figsize=None,
):
    """Plot improved signed heatmap inline (returns fig, ax)."""
    mat = assoc_matrix.copy().apply(pd.to_numeric, errors="coerce")
    vals = mat.to_numpy(dtype=float, copy=True)

    if vlim is None:
        vlim = _compute_symmetric_vlim(vals)

    norm = TwoSlopeNorm(vmin=-vlim, vcenter=style.center, vmax=+vlim)

    if figsize is None:
        figsize = _default_figsize(mat.shape[0], mat.shape[1])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(vals, aspect="auto", cmap=style.cmap, norm=norm)

    ax.set_title(title)
    ax.set_xlabel("Indicator")
    ax.set_ylabel("Event")

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right", fontsize=style.fontsize_x)

    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist(), fontsize=style.fontsize_y)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(style.colorbar_label)

    if style.annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = vals[i, j]
                if not np.isfinite(v):
                    continue
                color = "white" if abs(v) >= 0.60 * vlim else "black"
                ax.text(
                    j,
                    i,
                    style.fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=style.annotation_fontsize,
                    color=color,
                )

    fig.tight_layout()
    return fig, ax
