from __future__ import annotations

from pathlib import Path

import click

from .backtest import backtest_portfolio
from .data import download_prices, load_prices, save_prices
from .features import compute_returns, make_multi_asset_vol_dataset, realized_volatility
from .metrics import compare_strategies
from .models import ewma_vol_forecast, walk_forward_rf_forecast
from .portfolio import equal_weight, inverse_vol_weights


@click.group()
def main() -> None:
    """VolRisk command line interface."""


@main.command()
@click.option("--tickers", multiple=True, required=True, help="Ticker symbols.")
@click.option("--start", default="2015-01-01", help="Start date.")
@click.option("--end", default=None, help="End date.")
@click.option("--out", default="data/raw/prices.csv", help="Output CSV path.")
def download(tickers: tuple[str, ...], start: str, end: str | None, out: str) -> None:
    """Download price data."""
    prices = download_prices(list(tickers), start=start, end=end)
    save_prices(prices, out)
    click.echo(f"Saved prices to {out}")


@main.command()
@click.option("--prices", default="data/raw/prices.csv", help="Input prices CSV.")
@click.option("--out", default="reports/results.csv", help="Output metrics CSV.")
@click.option("--transaction-cost-bps", default=1.0, type=float)
@click.option("--include-rf/--no-include-rf", default=False, help="Include RF volatility strategy.")
@click.option(
    "--rf-target-transform",
    type=click.Choice(["raw", "log"]),
    default="raw",
    help="Target transform for the RF volatility strategy.",
)
@click.option(
    "--initial-train-days",
    default=756,
    type=int,
    help="Initial RF training window in trading days.",
)
@click.option("--refit-every", default=21, type=int, help="RF refit frequency in trading days.")
def backtest(
    prices: str,
    out: str,
    transaction_cost_bps: float,
    include_rf: bool,
    rf_target_transform: str,
    initial_train_days: int,
    refit_every: int,
) -> None:
    """Run baseline and optional Random Forest volatility backtests."""
    price_df = load_prices(prices)
    returns = compute_returns(price_df, method="simple")

    ew = equal_weight(returns.index, list(returns.columns))

    hist_vol = realized_volatility(returns, window=21)
    inv_vol = inverse_vol_weights(hist_vol, max_weight=0.35).reindex(returns.index).ffill()

    ewma_vol = ewma_vol_forecast(returns, span=21)
    ewma_weights = inverse_vol_weights(ewma_vol, max_weight=0.35).reindex(returns.index).ffill()

    strategies = {
        "equal_weight": ew,
        "inverse_realized_vol": inv_vol,
        "inverse_ewma_vol": ewma_weights,
    }

    if include_rf:
        click.echo("Building RF volatility forecasts...")
        X_vol, y_vol = make_multi_asset_vol_dataset(
            returns,
            target_window=21,
            lags=(1, 5, 21, 63),
            drop_missing_target=False,
        )
        rf_vol = walk_forward_rf_forecast(
            X_vol,
            y_vol,
            initial_train_days=initial_train_days,
            refit_every=refit_every,
            random_state=42,
            target_transform=rf_target_transform,
        )
        Path("reports").mkdir(exist_ok=True)
        rf_vol.to_csv(f"reports/rf_{rf_target_transform}_vol_forecasts.csv")
        if rf_target_transform == "raw":
            rf_vol.to_csv("reports/rf_vol_forecasts.csv")
        strategies[f"inverse_rf_forecast_vol_{rf_target_transform}"] = inverse_vol_weights(
            rf_vol, max_weight=0.35
        ).reindex(returns.index).ffill()

    results = {
        name: backtest_portfolio(price_df, weights, transaction_cost_bps=transaction_cost_bps)
        for name, weights in strategies.items()
    }

    metrics = compare_strategies(results)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out)
    click.echo(metrics.round(4).to_string())
    click.echo(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
