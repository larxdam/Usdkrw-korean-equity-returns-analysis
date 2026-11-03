"""Section 1: source-price validation, processed price tables, and price charts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
USD = "USD/KRW"
INSTRUMENTS = {
    USD: {"ticker": "KRW=X", "file": "usd_krw.csv"},
    "KOSPI": {"ticker": "^KS11", "file": "kospi.csv"},
    "Samsung": {"ticker": "005930.KS", "file": "samsung.csv"},
    "Hyundai": {"ticker": "005380.KS", "file": "hyundai.csv"},
    "Korean Air": {"ticker": "003490.KS", "file": "korean_air.csv"},
    "KB Financial": {"ticker": "105560.KS", "file": "kb_financial.csv"},
    "S&P 500": {"ticker": "^GSPC", "file": "sp500.csv"},
}
KOREA_ASSETS = ["Samsung", "Hyundai", "Korean Air", "KB Financial", "KOSPI"]


def read_raw_price(name: str, metadata: dict[str, str]) -> pd.Series:
    """Read one raw CSV and validate its adjusted-price series."""

    path = RAW_DIR / metadata["file"]
    if not path.exists():
        raise FileNotFoundError(f"Raw source file not found for {name}: {path}")
    frame = pd.read_csv(path, parse_dates=["Date"])
    missing = {"Date", "Adj Close"}.difference(frame.columns)
    if missing:
        raise ValueError(f"File {path} is missing required columns: {', '.join(sorted(missing))}")
    series = pd.to_numeric(frame.set_index("Date")["Adj Close"], errors="coerce").sort_index()
    if series.empty or series.index.has_duplicates:
        raise ValueError(f"File {path} has no prices or contains duplicate dates.")
    return series.rename(name)


def calculate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine raw prices and calculate the coverage and missing-value table."""

    series_by_name = {name: read_raw_price(name, metadata) for name, metadata in INSTRUMENTS.items()}
    prices = pd.concat(series_by_name.values(), axis=1).sort_index()
    rows = [
        {
            "Instrument": name,
            "Ticker": metadata["ticker"],
            "First date": series_by_name[name].index.min().date().isoformat(),
            "Last date": series_by_name[name].index.max().date().isoformat(),
            "Rows": len(series_by_name[name]),
            "Missing prices": int(series_by_name[name].isna().sum()),
            "Missing share": float(series_by_name[name].isna().mean()),
        }
        for name, metadata in INSTRUMENTS.items()
    ]
    return prices, pd.DataFrame(rows)


def save(prices: pd.DataFrame, quality: pd.DataFrame) -> None:
    """Save processed price and data-quality tables."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PROCESSED_DIR / "prices.csv", index_label="Date")
    quality.to_csv(PROCESSED_DIR / "data_quality.csv", index=False)


def figures(prices: pd.DataFrame) -> list[go.Figure]:
    """Build normalized-price and USD/KRW-level figures."""

    normalized = go.Figure()
    for name in INSTRUMENTS:
        series = prices[name].dropna()
        normalized.add_trace(go.Scatter(x=series.index, y=series / series.iloc[0] * 100, mode="lines", name=name))
    normalized.update_layout(title="Indexed prices from the first available date", xaxis_title="Date", yaxis_title="First price of each series = 100", height=620)
    usd = prices[USD].dropna()
    level = go.Figure(go.Scatter(x=usd.index, y=usd, mode="lines", name=USD, line={"color": "#1f77b4"}))
    level.update_layout(title="USD/KRW: daily level", xaxis_title="Date", yaxis_title="KRW per USD", height=620)
    return [normalized, level]
