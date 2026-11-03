"""Section 5: 252-observation rolling Pearson correlations and their chart."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .section_one import KOREA_ASSETS, PROCESSED_DIR, USD


def calculate(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling Pearson correlations on 252 shared observations per pair."""

    output = {}
    for asset in KOREA_ASSETS:
        pair = returns[[USD, asset]].dropna()
        output[asset] = pair[USD].rolling(window=252, min_periods=252).corr(pair[asset])
    return pd.DataFrame(output).sort_index()


def save(rolling: pd.DataFrame) -> None:
    """Save rolling-correlation time series."""

    rolling.to_csv(PROCESSED_DIR / "rolling_correlations.csv", index_label="Date")


def figures(rolling: pd.DataFrame) -> list[go.Figure]:
    """Build the rolling-correlation chart with a zero reference line."""

    fig = go.Figure()
    for asset in KOREA_ASSETS:
        series = rolling[asset].dropna()
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=asset))
    fig.add_hline(y=0, line_dash="dash", line_color="#4c4c4c")
    fig.update_layout(title="Rolling correlation over 252 shared observations", xaxis_title="Date", yaxis_title="Pearson r", yaxis_range=[-1, 1], height=640)
    return [fig]
