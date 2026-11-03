"""Section 4: HAC-robust OLS regressions and five individual scatter plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

from .section_one import KOREA_ASSETS, PROCESSED_DIR, USD


def pair_returns(returns: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Match USD/KRW and one asset only on shared available dates."""

    return returns[[USD, asset]].dropna().rename(columns={USD: "usd", asset: "asset"})


def calculate(returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate five OLS models with HAC-robust errors through five lags."""

    rows = []
    for asset in KOREA_ASSETS:
        pair = pair_returns(returns, asset)
        if len(pair) < 7:
            raise ValueError(f"Not enough shared observations for regression: {asset}")
        result = sm.OLS(pair["asset"], sm.add_constant(pair["usd"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({"Asset": asset, "alpha": result.params["const"], "beta USD/KRW": result.params["usd"], "p-value beta": result.pvalues["usd"], "R²": result.rsquared, "N": int(result.nobs)})
    return pd.DataFrame(rows)


def save(regressions: pd.DataFrame) -> None:
    """Save HAC-robust OLS summary statistics."""

    regressions.to_csv(PROCESSED_DIR / "regressions.csv", index=False)


def figures(returns: pd.DataFrame, regressions: pd.DataFrame) -> list[go.Figure]:
    """Build five separate scatter plots with binned means and OLS lines."""

    output = []
    for _, result in regressions.iterrows():
        asset, pair = result["Asset"], pair_returns(returns, result["Asset"])
        line_x = np.linspace(pair["usd"].min(), pair["usd"].max(), 100)
        groups = pd.qcut(pair["usd"], q=20, duplicates="drop")
        binned = pair.groupby(groups, observed=True).agg(usd=("usd", "mean"), asset=("asset", "mean"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pair["usd"], y=pair["asset"], mode="markers", name="Trading days", marker={"size": 6, "opacity": 0.45, "color": "#2f6f73"}, hovertemplate="USD/KRW: %{x:.2f}%<br>Asset: %{y:.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=binned["usd"], y=binned["asset"], mode="markers", name="Mean of 20 groups", marker={"size": 10, "color": "#111827", "symbol": "diamond"}))
        fig.add_trace(go.Scatter(x=line_x, y=result["alpha"] + result["beta USD/KRW"] * line_x, mode="lines", name="OLS", line={"color": "#c44e52", "width": 3}))
        fig.update_layout(title=f"{asset}: r_asset = α + β × r_USD/KRW", xaxis_title="USD/KRW return, %", yaxis_title=f"{asset} return, %", height=640, annotations=[{"x": 0.02, "y": 0.98, "xref": "paper", "yref": "paper", "showarrow": False, "align": "left", "bgcolor": "rgba(255,255,255,0.86)", "bordercolor": "#d9dee3", "borderwidth": 1, "text": f"β={result['beta USD/KRW']:.3f}<br>R²={result['R²']:.3f}<br>p={format_p(result['p-value beta'])}<br>N={int(result['N'])}"}])
        output.append(fig)
    return output




def format_p(value: float) -> str:
    """Format p-values for plot annotations."""

    return "<0.001" if value < 0.001 else f"{value:.3f}"
