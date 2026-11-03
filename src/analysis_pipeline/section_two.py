"""Section 2: daily logarithmic returns, their storage, and volatility charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .section_one import INSTRUMENTS, KOREA_ASSETS, PROCESSED_DIR, USD


def calculate(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate 100 * ln(P_t / P_t-1) on every instrument's own calendar."""

    if (prices <= 0).any().any():
        invalid = prices.columns[(prices <= 0).any()].tolist()
        raise ValueError(f"Log returns require positive prices: {', '.join(invalid)}")
    returns = {}
    for name in prices:
        series = prices[name].dropna()
        returns[name] = (np.log(series / series.shift(1)) * 100).reindex(prices.index)
    return pd.DataFrame(returns, index=prices.index)


def save(returns: pd.DataFrame) -> None:
    """Save combined and per-instrument daily-return tables."""

    returns.to_csv(PROCESSED_DIR / "returns.csv", index_label="Date")
    for name, metadata in INSTRUMENTS.items():
        returns[[name]].to_csv(PROCESSED_DIR / f"returns_{metadata['file']}", index_label="Date")


def figures(returns: pd.DataFrame) -> list[go.Figure]:
    """Build daily-return panels and the standard-deviation comparison."""

    names = [USD] + KOREA_ASSETS
    panel = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.025, subplot_titles=names)
    palette = ["#1f77b4", "#4c78a8", "#59a14f", "#f28e2b", "#b07aa1", "#e15759"]
    for row, (name, color) in enumerate(zip(names, palette), start=1):
        series = returns[name].dropna()
        panel.add_trace(go.Scatter(x=series.index, y=series, mode="lines", line={"width": 1, "color": color}, hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>"), row=row, col=1)
        panel.update_yaxes(title_text="%", row=row, col=1)
    panel.update_layout(title="Daily log returns: separate scale for each series", height=820, showlegend=False)
    panel.update_xaxes(title_text="Date", row=len(names), col=1)
    volatility = returns[names].std(skipna=True).sort_values(ascending=False)
    bar = go.Figure(go.Bar(x=volatility.index, y=volatility.values, marker_color=["#1f77b4" if name == USD else "#4c78a8" for name in volatility.index], text=[f"{value:.2f}" for value in volatility.values], textposition="outside"))
    bar.update_layout(title="Daily volatility comparison", yaxis_title="Standard deviation of daily return, percentage points", height=560, yaxis={"rangemode": "tozero"})
    return [panel, bar]
