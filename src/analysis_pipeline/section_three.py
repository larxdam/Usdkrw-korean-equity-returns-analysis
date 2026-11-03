"""Section 3: daily Pearson and Spearman correlations and their charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from .section_one import KOREA_ASSETS, PROCESSED_DIR, USD


def pair_returns(returns: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Match USD/KRW and one asset only on shared available dates."""

    return returns[[USD, asset]].dropna().rename(columns={USD: "usd", asset: "asset"})


def calculate(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlations with p-values and N."""

    rows = []
    for asset in KOREA_ASSETS:
        pair = pair_returns(returns, asset)
        if len(pair) < 3:
            raise ValueError(f"Not enough shared observations for correlation: {asset}")
        pearson_r, pearson_p = stats.pearsonr(pair["usd"], pair["asset"])
        spearman_rho, spearman_p = stats.spearmanr(pair["usd"], pair["asset"])
        rows.append({"Asset": asset, "Pearson r": pearson_r, "Pearson p-value": pearson_p, "Spearman rho": spearman_rho, "Spearman p-value": spearman_p, "N": len(pair)})
    return pd.DataFrame(rows)


def save(correlations: pd.DataFrame) -> None:
    """Save the daily correlation table."""

    correlations.to_csv(PROCESSED_DIR / "correlations.csv", index=False)


def figures(correlations: pd.DataFrame) -> list[go.Figure]:
    """Build heatmap and signed Pearson-correlation bars."""

    values = correlations[["Pearson r", "Spearman rho"]].T
    heatmap = go.Figure(go.Heatmap(z=values.values, x=correlations["Asset"], y=values.index, colorscale="RdBu", zmid=0, zmin=-0.2, zmax=0.2, text=np.round(values.values, 3), texttemplate="%{text}", hovertemplate="%{y} vs USD/KRW / %{x}: %{z:.3f}<extra></extra>"))
    heatmap.update_layout(title="Daily return correlations with USD/KRW: scale [−0.2, 0.2]", height=560)
    ordered = correlations.sort_values("Pearson r")
    bars = go.Figure(go.Bar(x=ordered["Asset"], y=ordered["Pearson r"], marker_color=["#c44e52" if value < 0 else "#2f6f73" for value in ordered["Pearson r"]], text=[f"{value:.3f}" for value in ordered["Pearson r"]], textposition="outside"))
    bars.add_hline(y=0, line_dash="dash", line_color="#4c4c4c")
    bars.update_layout(title="Direction and strength of the daily relationship with USD/KRW", yaxis_title="Pearson r", height=560, yaxis_range=[min(-0.2, ordered["Pearson r"].min() - 0.03), 0.05])
    return [heatmap, bars]
