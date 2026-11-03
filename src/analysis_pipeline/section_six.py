"""Section 6: W-FRI weekly returns and Korea-versus-US correlation comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from .section_one import PROCESSED_DIR, USD


WEEKLY_ASSETS = ["KOSPI", "S&P 500"]
WEEKLY_COLUMNS = [USD] + WEEKLY_ASSETS


def calculate_weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Use each calendar week's last available price and calculate own-calendar returns."""

    weekly_prices = prices[WEEKLY_COLUMNS].resample("W-FRI").last()
    output = {}
    for name in WEEKLY_COLUMNS:
        series = weekly_prices[name].dropna()
        output[name] = (np.log(series / series.shift(1)) * 100).reindex(weekly_prices.index)
    return pd.DataFrame(output, index=weekly_prices.index)


def calculate_correlations(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson and Spearman weekly correlations against USD/KRW."""

    rows = []
    for asset in WEEKLY_ASSETS:
        pair = weekly_returns[[USD, asset]].dropna()
        pearson_r, pearson_p = stats.pearsonr(pair[USD], pair[asset])
        spearman_rho, spearman_p = stats.spearmanr(pair[USD], pair[asset])
        rows.append({"Asset": asset, "Pearson r": pearson_r, "Pearson p-value": pearson_p, "Spearman rho": spearman_rho, "Spearman p-value": spearman_p, "N weeks": len(pair)})
    return pd.DataFrame(rows)


def save(weekly_returns: pd.DataFrame, correlations: pd.DataFrame) -> None:
    """Save weekly returns and their two correlation summaries."""

    weekly_returns.to_csv(PROCESSED_DIR / "weekly_returns.csv", index_label="Date")
    correlations.to_csv(PROCESSED_DIR / "weekly_correlations.csv", index=False)


def figures(correlations: pd.DataFrame) -> list[go.Figure]:
    """Build the weekly Pearson-correlation bar chart."""

    fig = go.Figure(go.Bar(x=correlations["Asset"], y=correlations["Pearson r"], marker_color=["#2f6f73", "#9a6a2f"], text=[f"{value:.3f}" for value in correlations["Pearson r"]], textposition="outside"))
    fig.add_hline(y=0, line_dash="dash", line_color="#4c4c4c")
    fig.update_layout(title="Weekly return correlation with USD/KRW", yaxis_title="Pearson r", height=560)
    return [fig]
