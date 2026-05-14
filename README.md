# VolRisk: Volatility Forecasting and Risk-Aware Portfolio Allocation

This project implements a reproducible finance pipeline for ORIE 5270.

The core question:

> Can volatility forecasts improve portfolio risk management relative to equal-weight and
> simple inverse-volatility baselines?

The package downloads public ETF price data, computes return and volatility features, trains
volatility forecasting models, constructs risk-aware portfolios, backtests strategies, and reports
performance metrics.

## Project structure

```text
volrisk/
  data.py          # price downloading, saving, and loading
  features.py      # returns, realized volatility, supervised model features
  models.py        # historical, EWMA, and Random Forest volatility forecasts
  portfolio.py     # equal-weight and inverse-volatility allocations
  backtest.py      # portfolio backtesting
  metrics.py       # Sharpe, drawdown, turnover, etc.
  plotting.py      # result plots
  cli.py           # command line interface
tests/
examples/
scripts/
data/
reports/
```

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra installs `pytest`, `pytest-cov`, and `ruff`. The default test command in
`pyproject.toml` expects `pytest-cov` to be available.

## Data

The default downloader uses Stooq CSV endpoints. Stooq requires an API key for these downloads, so
set `STOOQ_API_KEY` before downloading fresh data:

```bash
export STOOQ_API_KEY="your-api-key"
```

The main pipeline reads `data/raw/prices.csv` if it already exists. In that case, no network request
is made.

## Quick start

Run the full example pipeline:

```bash
python scripts/run_pipeline.py
```

The pipeline uses the ETF universe:

```text
SPY, QQQ, IWM, TLT, GLD, EFA, EEM, VNQ
```

Or use the CLI:

```bash
volrisk download \
  --tickers SPY --tickers QQQ --tickers IWM --tickers TLT \
  --tickers GLD --tickers EFA --tickers EEM --tickers VNQ \
  --start 2015-01-01 \
  --out data/raw/prices.csv

volrisk backtest --prices data/raw/prices.csv --out reports/results.csv
```

The CLI backtest runs the baseline strategies by default. Add `--include-rf` to include the slower
Random Forest volatility strategy:

```bash
volrisk backtest \
  --prices data/raw/prices.csv \
  --out reports/results.csv \
  --include-rf \
  --initial-train-days 756 \
  --refit-every 21
```

## Strategies implemented

1. **Equal weight**  
   Allocates equally across all assets.

2. **Inverse realized volatility**  
   Uses trailing 21-day realized volatility and allocates inversely to volatility.

3. **Inverse EWMA volatility**  
   Uses an exponentially weighted volatility forecast and allocates inversely to forecasted
   volatility.

4. **Inverse Random Forest forecast volatility**  
   Builds a pooled multi-asset supervised learning dataset from lagged return and volatility
   features, trains a walk-forward `RandomForestRegressor`, forecasts each asset's forward
   volatility, and allocates inversely to those forecasts.

Portfolio weights are long-only and normalized to sum to one. The inverse-volatility strategies use
a maximum per-asset weight cap of `0.35` in the main pipeline.

## Random Forest details

The Random Forest feature set is built in `volrisk.features.make_multi_asset_vol_dataset`.
It includes:

- one-day return and absolute return
- rolling mean absolute returns
- rolling mean returns
- rolling realized volatility
- an `asset` categorical feature for pooled multi-asset training

For the one-day realized-volatility feature, the code uses absolute return times `sqrt(252)`.
A one-observation rolling standard deviation is undefined, so this avoids dropping the entire
training set when `lags` includes `1`.

The multi-asset dataset attaches the target column before concatenating assets. This avoids pandas
alignment problems caused by duplicate date indices across assets.

`walk_forward_rf_forecast` trains only on dates before the forecast date and returns a wide
Date x Asset volatility forecast matrix. It accepts both the current `initial_train_days` parameter
and the legacy `initial_train_size` name for compatibility.

## Outputs

The full pipeline writes:

- `reports/results.csv`
- `reports/equal_weight_backtest.csv`
- `reports/inverse_realized_vol_backtest.csv`
- `reports/inverse_ewma_vol_backtest.csv`
- `reports/inverse_rf_forecast_vol_backtest.csv`
- `reports/rf_vol_forecasts.csv`
- `reports/figures/cumulative_returns.png`
- `reports/figures/drawdowns.png`

## Performance metrics

The package reports:

- annualized return
- annualized volatility
- Sharpe ratio
- maximum drawdown
- Calmar ratio
- average turnover
- final cumulative return

## Testing

Run the configured test suite with coverage:

```bash
pytest
```

Or, if `pytest-cov` is not installed in the active environment, disable the configured coverage
options:

```bash
python -m pytest -q -o addopts=''
```

Run lint checks on the main package and tests:

```bash
ruff check volrisk scripts tests
```

## Notes

This project is for educational purposes only and is not investment advice. Backtest results depend
on the selected asset universe, date range, data source, transaction cost assumption, and model
settings.
