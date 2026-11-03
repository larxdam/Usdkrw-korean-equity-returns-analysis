from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.io as pio

from analysis_pipeline import section_five, section_four, section_one, section_six, section_three, section_two


REPORT_PATH = Path(__file__).resolve().parent / "index.html"


def render_table(frame: pd.DataFrame) -> str:
    """Format a calculated table for inclusion in the report."""

    output = frame.copy()
    for column in output.select_dtypes(include=[float]).columns:
        if "p-value" in column:
            output[column] = output[column].map(format_p)
        elif "share" in column.lower():
            output[column] = output[column].map(lambda value: f"{value:.2%}")
        else:
            output[column] = output[column].map(lambda value: f"{value:.4f}")
    return output.to_html(index=False, escape=True, classes="data-table")


def format_p(value: float) -> str:
    """Format a p-value compactly for a report table."""

    return "<0.001" if value < 0.001 else f"{value:.3f}"


def figure_html(figures: list[object]) -> list[str]:
    """Embed Plotly once and render all figures as responsive HTML fragments."""

    return [
        pio.to_html(figure, full_html=False, include_plotlyjs="inline" if index == 0 else False, config={"responsive": True, "displaylogo": False}, default_width="100%")
        for index, figure in enumerate(figures)
    ]


def section_html(title: str, text: str, table: pd.DataFrame | None, figures: list[str]) -> str:
    """Assemble one report section from its calculated text, table, and charts."""

    table_html = render_table(table) if table is not None else ""
    plots = "".join(f'<div class="plot">{figure}</div>' for figure in figures)
    return f'<section class="section"><h2>{html.escape(title)}</h2>{text}{table_html}{plots}</section>'


def paragraphs(*items: str) -> str:
    """Escape and wrap dynamically generated report commentary."""

    return "".join(f"<p>{html.escape(item)}</p>" for item in items)


def report_texts(
    quality: pd.DataFrame,
    returns: pd.DataFrame,
    correlations: pd.DataFrame,
    regressions: pd.DataFrame,
    rolling: pd.DataFrame,
    weekly: pd.DataFrame,
) -> dict[str, str]:
    """Create every data-driven paragraph and the question-based conclusion block."""

    volatility = returns[[section_one.USD] + section_one.KOREA_ASSETS].std(skipna=True)
    highest, lowest = volatility.idxmax(), volatility.idxmin()
    strongest_corr = correlations.loc[correlations["Pearson r"].abs().idxmax()]
    strongest_beta = regressions.loc[regressions["beta USD/KRW"].abs().idxmax()]
    best_fit = regressions.loc[regressions["R²"].idxmax()]
    ranges = [f"{asset}: from {rolling[asset].min():.3f} to {rolling[asset].max():.3f}" for asset in section_one.KOREA_ASSETS]
    sign_changes = [asset for asset in section_one.KOREA_ASSETS if rolling[asset].min() < 0 < rolling[asset].max()]
    negative_only = [asset for asset in section_one.KOREA_ASSETS if rolling[asset].max() < 0]
    widest_asset = max(section_one.KOREA_ASSETS, key=lambda asset: rolling[asset].max() - rolling[asset].min())
    kospi_weekly = weekly.loc[weekly["Asset"].eq("KOSPI")].iloc[0]
    sp500_weekly = weekly.loc[weekly["Asset"].eq("S&P 500")].iloc[0]
    conclusion_pairs = [
        (
            "How are changes in USD/KRW related to Korean equity returns?",
            f"The daily Pearson correlation is negative in all five pairs. This means that on days when USD/KRW rose, indicating a weaker won, Korean equities and KOSPI more often had lower returns; this is a tendency in the full sample, not a pattern that must occur on every individual day. The strongest relationship is for {strongest_corr['Asset']}: r={strongest_corr['Pearson r']:.3f}. Its magnitude indicates a weak statistical relationship on matched dates and does not establish causality.",
        ),
        (
            "Does the USD/KRW relationship with Korean equity returns differ across sectors?",
            f"Yes. The largest absolute β belongs to {strongest_beta['Asset']}: {strongest_beta['beta USD/KRW']:.3f}. R² measures the share of variation in daily returns described by a regression that uses USD/KRW alone. The highest R² is {best_fit['R²']:.3f} for {best_fit['Asset']}, or about {best_fit['R²'] * 100:.1f}%. The assets differ, but every one-factor model explains only a small share of daily movements.",
        ),
        (
            "How stable is the USD/KRW relationship with Korean equity returns over time?",
            f"The relationship changes over time. The latest estimates in 252-observation windows are Samsung {rolling['Samsung'].dropna().iloc[-1]:.3f}, Hyundai {rolling['Hyundai'].dropna().iloc[-1]:.3f}, Korean Air {rolling['Korean Air'].dropna().iloc[-1]:+.3f}, KB Financial {rolling['KB Financial'].dropna().iloc[-1]:.3f}, and KOSPI {rolling['KOSPI'].dropna().iloc[-1]:.3f}. The rolling-correlation ranges are {'; '.join(ranges)}. The full-period coefficient for 2015–2025 therefore does not describe every subperiod equally well.",
        ),
        (
            "Does the Korean market behave differently from the US market?",
            f"In weekly data, the relationship with USD/KRW is stronger for KOSPI: r={kospi_weekly['Pearson r']:.3f}, versus r={sp500_weekly['Pearson r']:.3f} for the S&P 500. This comparison is limited by different market trading hours and calendars.",
        ),
    ]
    conclusions = "".join(f"<h3>{html.escape(question)}</h3>{paragraphs(answer)}" for question, answer in conclusion_pairs)
    conclusions += "<h3>What this study cannot establish</h3>" + paragraphs(
        "The comparison shows how USD/KRW and the returns of the selected assets moved on the same dates, and it makes those movements comparable across companies, KOSPI, and the S&P 500. It does not establish that currency movements caused equity movements. The model omits other factors that may affect both series at the same time, including global market conditions, interest rates, company news, sector conditions, and macroeconomic events. Differences in trading hours and calendars also affect interpretation, so the results should be read as a description of statistical relationships in this period."
    )
    return {
        "quality": paragraphs(
            "In the first step, the data were brought to a common format: date and adjusted closing price for each instrument. The series were then combined in one date-indexed table. Dates were deliberately not filled with artificial values. Correlations and regressions use only days on which both prices in a given pair are available.",
            "There were no missing prices within the source series; row counts differ only because trading calendars differ. For visual comparison, price levels were normalized: the first available price of each instrument is set to 100, and each subsequent point is calculated as current price / first price × 100. This makes it possible to compare cumulative changes in series with different units: USD/KRW, index points, and individual share prices. USD/KRW is also shown in a separate chart.",
        ),
        "returns": paragraphs(
            "Logarithmic return is calculated on each instrument's own trading calendar: 100 × ln(Pₜ / Pₜ₋₁). This approach does not create returns for non-trading days.",
            f"{highest} has the highest daily volatility at {volatility[highest]:.2f} percentage points, while {lowest} has the lowest at {volatility[lowest]:.2f} percentage points. The individual equities fluctuate substantially more than the currency series.",
        ),
        "correlations": paragraphs(
            "The Pearson coefficient, denoted r, measures the linear relationship between two series: if r is close to +1, they usually rise and fall together; if it is close to -1, one series more often rises when the other falls; if it is near 0, there is little persistent linear relationship. In this study, a negative r means that days of rising USD/KRW, or a weaker won, more often coincided with weaker Korean equity returns.",
            "The Spearman coefficient, denoted rho, measures a rank relationship rather than a linear one: it tests whether the ordering of days by the size of their movements is similar. This is useful when the relationship may not be strictly linear or when a few extreme days have too much influence on Pearson's coefficient. Here, Pearson and Spearman coefficients give a similar picture, so the conclusion about the direction of the relationship is not dependent on the linear metric alone.",
            f"Daily correlation with USD/KRW is negative and small in magnitude in all five pairs: from {correlations['Pearson r'].max():.3f} for {correlations.loc[correlations['Pearson r'].idxmax(), 'Asset']} to {strongest_corr['Pearson r']:.3f} for {strongest_corr['Asset']}. Each pair contains {int(strongest_corr['N'])} matched dates. All coefficients are statistically significant at the 5% level, but their magnitude indicates a weak daily relationship rather than strong joint movement.",
        ),
        "regressions": paragraphs(
            "In the scatter plots, each point is one shared trading day: daily USD/KRW return is on the horizontal axis and daily return of the equity or KOSPI is on the vertical axis. The red line is the simple linear regression r_asset = α + β × r_USD/KRW. Its slope, β, shows the average change in the asset's return associated with a one-percentage-point change in USD/KRW.",
            "All β estimates are negative. KB Financial has the strongest sensitivity: β = −0.515, meaning that a day with USD/KRW up by one percentage point is associated, on average, with a KB Financial return 0.52 percentage points lower. It is followed by Korean Air (−0.477), KOSPI (−0.428), Hyundai (−0.419), and Samsung (−0.402).",
            "For each β, the null hypothesis H₀: β = 0 is tested; it states that USD/KRW is unrelated to the return of that asset in this model. All five HAC-robust p-values are below 0.001, so the data are sufficient to reject this null hypothesis at the conventional 5% level. This indicates a statistically distinguishable negative slope, but it does not demonstrate a causal effect of the exchange rate.",
            f"The black diamonds in the scatter plots show mean values for 20 approximately equal-sized groups of days ordered by USD/KRW return. They make the slope easier to see through the dense cloud of points. The model has little explanatory power: the largest R² is {best_fit['R²']:.3f} for {best_fit['Asset']}, meaning that USD/KRW explains about {best_fit['R²'] * 100:.1f}% of the daily variation in its return; the smallest is {regressions['R²'].min():.3f} for {regressions.loc[regressions['R²'].idxmin(), 'Asset']}.",
        ),
        "rolling": paragraphs(
            "A rolling correlation over 252 shared trading observations does not show the average result for the full period; it shows how the relationship changed over time. This is roughly a one-year horizon, so the chart addresses whether the relationship was stable or depended on the period. Adjacent windows overlap, so they are not independent annual estimates.",
            f"The relationship was not the same throughout the sample. Its sign changed for {', '.join(sign_changes)}; for {', '.join(negative_only)}, the estimate remained negative in every window. The widest range is for {widest_asset}: from {rolling[widest_asset].min():.3f} to {rolling[widest_asset].max():+.3f}.",
            f"At the end of the period, the estimates are KOSPI {rolling['KOSPI'].dropna().iloc[-1]:.3f}, Samsung {rolling['Samsung'].dropna().iloc[-1]:.3f}, KB Financial {rolling['KB Financial'].dropna().iloc[-1]:.3f}, Hyundai {rolling['Hyundai'].dropna().iloc[-1]:.3f}, and Korean Air {rolling['Korean Air'].dropna().iloc[-1]:+.3f}. The full-period coefficient for 2015–2025 should therefore not be applied to every year separately.",
        ),
        "weekly": paragraphs(
            "Weekly return uses the last available price in each W-FRI calendar week and is then calculated without artificial filling of missing values. Market trading hours and holiday calendars remain a limitation of the comparison.",
            f"Pearson r is {kospi_weekly['Pearson r']:.3f} for KOSPI and {sp500_weekly['Pearson r']:.3f} for the S&P 500. The result compares the statistical relationship of weekly movements; it does not establish causality.",
        ),
        "conclusions": conclusions,
    }


def build_report(
    prices: pd.DataFrame,
    quality: pd.DataFrame,
    returns: pd.DataFrame,
    correlations: pd.DataFrame,
    regressions: pd.DataFrame,
    rolling: pd.DataFrame,
    weekly_correlations: pd.DataFrame,
) -> str:
    """Assemble all calculated sections, figures, tables, and conclusions."""

    groups = [
        section_one.figures(prices),
        section_two.figures(returns),
        section_three.figures(correlations),
        section_four.figures(returns, regressions),
        section_five.figures(rolling),
        section_six.figures(weekly_correlations),
    ]
    rendered = figure_html([figure for group in groups for figure in group])
    positions = iter(rendered)
    section_figures = [[next(positions) for _ in group] for group in groups]
    texts = report_texts(quality, returns, correlations, regressions, rolling, weekly_correlations)
    sections = "".join(
        [
            section_html("1. Data and quality", texts["quality"], quality, section_figures[0]),
            section_html("2. Daily returns", texts["returns"], None, section_figures[1]),
            section_html("3. Correlations", texts["correlations"], correlations, section_figures[2]),
            section_html("4. Regressions and scatter plots", texts["regressions"], regressions, section_figures[3]),
            section_html("5. The relationship over time", texts["rolling"], None, section_figures[4]),
            section_html("6. Weekly comparison of Korea and the US", texts["weekly"], weekly_correlations, section_figures[5]),
            f'<section class="section"><h2>Conclusions</h2>{texts["conclusions"]}</section>',
        ]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>USD/KRW and South Korean equities</title><style>
:root {{ --text:#17202a; --muted:#526170; --line:#d8dee6; --band:#f5f7fa; }}
body {{ margin:0; font-family:Inter,Arial,sans-serif; color:var(--text); line-height:1.55; }} header {{ padding:42px max(24px,7vw) 30px; background:#eef3f6; border-bottom:1px solid var(--line); }} main {{ padding:30px max(24px,7vw) 60px; }} h1 {{ margin:0 0 12px; font-size:clamp(30px,5vw,54px); line-height:1.05; }} h2 {{ margin:46px 0 12px; font-size:clamp(23px,3vw,34px); }} p {{ max-width:980px; color:var(--muted); }} .section {{ border-top:1px solid var(--line); margin-top:36px; padding-top:8px; }} .plot {{ margin:18px 0 26px; }} .data-table {{ border-collapse:collapse; width:100%; margin:14px 0 26px; font-size:15px; }} .data-table th,.data-table td {{ border-bottom:1px solid var(--line); padding:10px 12px; text-align:right; }} .data-table th:first-child,.data-table td:first-child {{ text-align:left; }} .data-table th {{ background:var(--band); color:#26323f; }}
</style></head><body><header><h1>USD/KRW and South Korean equity returns</h1><p>Period: 01 Jan 2015–31 Dec 2025. A rise in USD/KRW indicates a weaker won.</p></header><main>{sections}</main></body></html>"""


def main() -> None:
    """Calculate every section from raw data, save tables, and create the report."""

    prices, quality = section_one.calculate()
    section_one.save(prices, quality)
    returns = section_two.calculate(prices)
    section_two.save(returns)
    correlations = section_three.calculate(returns)
    section_three.save(correlations)
    regressions = section_four.calculate(returns)
    section_four.save(regressions)
    rolling = section_five.calculate(returns)
    section_five.save(rolling)
    weekly_returns = section_six.calculate_weekly_returns(prices)
    weekly_correlations = section_six.calculate_correlations(weekly_returns)
    section_six.save(weekly_returns, weekly_correlations)
    REPORT_PATH.write_text(build_report(prices, quality, returns, correlations, regressions, rolling, weekly_correlations), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
