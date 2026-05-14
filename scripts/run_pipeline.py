from pathlib import Path

from volrisk.backtest import backtest_portfolio
from volrisk.data import download_prices, load_prices, save_prices
from volrisk.features import compute_returns, make_multi_asset_vol_dataset, realized_volatility
from volrisk.metrics import compare_strategies
from volrisk.models import ewma_vol_forecast, walk_forward_rf_forecast
from volrisk.plotting import plot_cumulative_returns, plot_drawdowns
from volrisk.portfolio import equal_weight, inverse_vol_weights


def main() -> None:
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EFA", "EEM", "VNQ"]
    price_path = Path("data/raw/prices.csv")

    if price_path.exists():
        prices = load_prices(price_path)
    else:
        prices = download_prices(tickers, start="2015-01-01", source="stooq", verify_ssl=False)
        save_prices(prices, price_path)

    returns = compute_returns(prices, method="simple")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    weights_equal = equal_weight(returns.index, list(returns.columns))

    realized_vol = realized_volatility(returns, window=21)
    weights_inv_vol = inverse_vol_weights(realized_vol, max_weight=0.35).reindex(returns.index).ffill()

    ewma_vol = ewma_vol_forecast(returns, span=21)
    weights_ewma = inverse_vol_weights(ewma_vol, max_weight=0.35).reindex(returns.index).ffill()

    print("Building Random Forest volatility forecast dataset...")
    X_vol, y_vol = make_multi_asset_vol_dataset(
        returns,
        target_window=21,
        lags=(1, 5, 21, 63),
    )

    print("Running walk-forward Random Forest forecasts...")
    rf_vol = walk_forward_rf_forecast(
        X_vol,
        y_vol,
        initial_train_days=756,
        refit_every=21,
        random_state=42,
    )
    rf_vol.to_csv("reports/rf_vol_forecasts.csv")

    weights_rf = inverse_vol_weights(rf_vol, max_weight=0.35).reindex(returns.index).ffill()

    strategies = {
        "Equal Weight": weights_equal,
        "Inverse Realized Vol": weights_inv_vol,
        "Inverse EWMA Vol": weights_ewma,
        "Inverse RF Forecast Vol": weights_rf,
    }

    results = {
        name: backtest_portfolio(prices, weights, transaction_cost_bps=1.0)
        for name, weights in strategies.items()
    }

    metrics = compare_strategies(results)
    metrics.to_csv("reports/results.csv")

    for name, result in results.items():
        safe_name = name.lower().replace(" ", "_")
        result.to_csv(f"reports/{safe_name}_backtest.csv")

    plot_cumulative_returns(results, "reports/figures/cumulative_returns.png")
    plot_drawdowns(results, "reports/figures/drawdowns.png")

    print(metrics.round(4))


if __name__ == "__main__":
    main()
