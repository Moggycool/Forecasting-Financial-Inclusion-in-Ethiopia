""" Chart utility functions for creating various Plotly charts used in the dashboard. """
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def line_with_ci(df: pd.DataFrame, x: str, y: str, lo: str, hi: str, color: str, title: str):
    """Create a line chart with confidence intervals."""
    fig = go.Figure()

    # add each scenario as separate traces with band
    for scn, g in df.groupby(color):
        g = g.sort_values(x)

        fig.add_trace(go.Scatter(
            x=g[x], y=g[hi],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=g[x], y=g[lo],
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,0,0,0.08)",
            name=f"{scn} CI",
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=g[x], y=g[y],
            mode="lines+markers",
            name=str(scn),
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        legend_title=color,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def multi_line(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    """Create a multi-line chart."""
    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title, template="plotly_white")
    fig.update_layout(hovermode="x unified")
    return fig


def bar_sorted(df: pd.DataFrame, x: str, y: str, title: str):
    """Create a bar chart sorted by y values descending."""
    dd = df.sort_values(y, ascending=False)
    fig = px.bar(dd, x=x, y=y, title=title, template="plotly_white")
    return fig


def target_progress(df: pd.DataFrame, x: str, y: str, target: float, title: str):
    """Create a line chart showing progress toward a target."""
    fig = go.Figure()
    df = df.sort_values(x)
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines+markers", name="Projection"))
    fig.add_hline(y=target, line_dash="dash", line_color="red", annotation_text=f"Target {target:.0f}%")
    fig.update_layout(
        title=title, xaxis_title=x, yaxis_title=y,
        template="plotly_white", hovermode="x unified"
    )
    return fig
