# USD/KRW and Korean Equity Returns

This project examines the relationship between USD/KRW changes and the returns of Samsung, Hyundai, Korean Air, KB Financial, and KOSPI during 2015–2025. It also compares weekly correlations with KOSPI and the S&P 500.

The analysis includes log returns, volatility, Pearson and Spearman correlations, OLS regressions with HAC standard errors, and rolling correlations.

## Run the project

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/LaraDavydova/Usdkrw-korean-equity-returns-analysis.git
   cd Usdkrw-korean-equity-returns-analysis
   ```

3. Install the Python environment and dependencies:

   ```bash
   uv sync
   ```

4. Run the analysis:

   ```bash
   uv run python main.py
   ```

5. Open `index.html` in your browser to view the interactive report. The generated report works offline.

## Files

- `data/raw/`: fixed local price data; no market data is downloaded during execution.
- `data/processed/`: calculated tables, regenerated on each run.
- `src/analysis_pipeline/`: one module for each of the six analytical sections.
- `main.py`: runs the calculations and generates `index.html`.

The source tickers are `KRW=X`, `005930.KS`, `005380.KS`, `003490.KS`, `105560.KS`, `^KS11`, and `^GSPC`.
