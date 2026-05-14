from pathlib import Path

import pandas as pd

from volrisk.backtest import backtest_portfolio
from volrisk.data import download_prices, load_prices, save_prices
from volrisk.features import compute_returns, make_multi_asset_vol_dataset, realized_volatility
from volrisk.metrics import compare_strategies
from volrisk.models import ewma_vol_forecast, walk_forward_rf_forecast
from volrisk.plotting import (
    plot_cumulative_returns,
    plot_cumulative_returns_common_start,
    plot_drawdowns,
)
from volrisk.portfolio import equal_weight, inverse_vol_weights


def _make_common_start_results(
    results: dict[str, pd.DataFrame],
) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, pd.DataFrame]]:
    common_start = max(result.index.min() for result in results.values())
    common_end = min(result.index.max() for result in results.values())
    if common_start > common_end:
        raise ValueError("strategy backtests do not have an overlapping date range")

    common_results = {}
    for name, result in results.items():
        common_result = result.loc[common_start:common_end].copy()
        common_result["cumulative_return"] = (1 + common_result["net_return"]).cumprod()
        common_results[name] = common_result

    return common_start, common_end, common_results


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _metrics_to_markdown(metrics: pd.DataFrame) -> str:
    columns = [
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
        "final_cumulative_return",
    ]
    headers = [
        "Strategy",
        "Ann. Return",
        "Ann. Vol",
        "Sharpe",
        "Max Drawdown",
        "Avg. Turnover",
        "Final Growth",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for strategy, row in metrics[columns].iterrows():
        values = [_format_metric(row[column]) for column in columns]
        lines.append(f"| {strategy} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_brief_report(
    metrics: pd.DataFrame,
    common_start_metrics: pd.DataFrame,
    common_start: pd.Timestamp,
    common_end: pd.Timestamp,
    out_path: str | Path,
) -> None:
    best_full_return = metrics["annualized_return"].idxmax()
    best_full_sharpe = metrics["sharpe_ratio"].idxmax()
    best_common_growth = common_start_metrics["final_cumulative_return"].idxmax()

    rf_raw = "Inverse RF Forecast Vol Raw"
    rf_log = "Inverse RF Forecast Vol Log"
    rf_analysis = ""
    if rf_raw in common_start_metrics.index and rf_log in common_start_metrics.index:
        raw_growth = common_start_metrics.loc[rf_raw, "final_cumulative_return"]
        log_growth = common_start_metrics.loc[rf_log, "final_cumulative_return"]
        raw_sharpe = common_start_metrics.loc[rf_raw, "sharpe_ratio"]
        log_sharpe = common_start_metrics.loc[rf_log, "sharpe_ratio"]
        stronger = "raw" if raw_growth >= log_growth else "log"
        rf_analysis = (
            f"- Over the common-start window, the RF {stronger} target variant produced "
            f"the stronger RF result. RF raw ended at {_format_metric(raw_growth)} "
            f"with Sharpe {_format_metric(raw_sharpe)}, while RF log ended at "
            f"{_format_metric(log_growth)} with Sharpe {_format_metric(log_sharpe)}."
        )

    report = f"""# VolRisk Pipeline Report

## Summary
This report was generated automatically by `scripts/run_pipeline.py`. It compares equal-weight,
inverse-volatility, EWMA-volatility, and Random Forest volatility forecast strategies.

The full-history metrics start when each strategy first has valid weights. The common-start
metrics rebase every strategy to the shared window from {common_start.date()} to
{common_end.date()}, which makes the Random Forest strategies comparable with the baselines.

## Full-History Results
{_metrics_to_markdown(metrics)}

## Common-Start Results
{_metrics_to_markdown(common_start_metrics)}

![Cumulative returns with common start](figures/cumulative_returns_common_start.png)

## Analysis
- The best full-history annualized return came from **{best_full_return}**.
- The best full-history Sharpe ratio came from **{best_full_sharpe}**.
- On the common-start window, the highest final growth came from **{best_common_growth}**.
{rf_analysis}
- The common-start chart is the fairest visual comparison because the RF strategies require
  an initial training window before producing out-of-sample forecasts.

## Conclusion
The RF raw-target strategy is the primary Random Forest benchmark to emphasize if it remains
stronger than the log-target variant. The common-start results should be used when comparing RF
against the baseline strategies, while the full-history results are useful for showing the longer
baseline context.
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")


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
        drop_missing_target=False,
    )

    print("Running walk-forward Random Forest forecasts with raw volatility target...")
    rf_raw_vol = walk_forward_rf_forecast(
        X_vol,
        y_vol,
        initial_train_days=756,
        refit_every=21,
        random_state=42,
        target_transform="raw",
    )
    rf_raw_vol.to_csv("reports/rf_vol_forecasts.csv")
    rf_raw_vol.to_csv("reports/rf_raw_vol_forecasts.csv")

    print("Running walk-forward Random Forest forecasts with log volatility target...")
    rf_log_vol = walk_forward_rf_forecast(
        X_vol,
        y_vol,
        initial_train_days=756,
        refit_every=21,
        random_state=42,
        target_transform="log",
    )
    rf_log_vol.to_csv("reports/rf_log_vol_forecasts.csv")

    weights_rf_raw = inverse_vol_weights(rf_raw_vol, max_weight=0.35).reindex(returns.index).ffill()
    weights_rf_log = inverse_vol_weights(rf_log_vol, max_weight=0.35).reindex(returns.index).ffill()

    strategies = {
        "Equal Weight": weights_equal,
        "Inverse Realized Vol": weights_inv_vol,
        "Inverse EWMA Vol": weights_ewma,
        "Inverse RF Forecast Vol Raw": weights_rf_raw,
        "Inverse RF Forecast Vol Log": weights_rf_log,
    }

    results = {
        name: backtest_portfolio(prices, weights, transaction_cost_bps=1.0)
        for name, weights in strategies.items()
    }

    metrics = compare_strategies(results)
    metrics.to_csv("reports/results.csv")
    common_start, common_end, common_results = _make_common_start_results(results)
    common_start_metrics = compare_strategies(common_results)
    common_start_metrics.to_csv("reports/results_common_start.csv")

    for name, result in results.items():
        safe_name = name.lower().replace(" ", "_")
        result.to_csv(f"reports/{safe_name}_backtest.csv")

    plot_cumulative_returns(results, "reports/figures/cumulative_returns.png")
    plot_cumulative_returns_common_start(
        results,
        "reports/figures/cumulative_returns_common_start.png",
    )
    plot_drawdowns(results, "reports/figures/drawdowns.png")
    _write_brief_report(
        metrics,
        common_start_metrics,
        common_start,
        common_end,
        "reports/report.md",
    )

    print(metrics.round(4))


if __name__ == "__main__":
    main()
