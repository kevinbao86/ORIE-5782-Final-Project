from __future__ import annotations

import pandas as pd

from .features import compute_returns


def backtest_portfolio(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Backtest a long-only portfolio.

    Weights at date t are shifted by one day and applied to returns from t to t+1,
    which avoids look-ahead bias.
    """
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be nonnegative")

    returns = compute_returns(prices, method="simple")
    weights = weights.reindex(returns.index).ffill().dropna(how="all")
    common_index = returns.index.intersection(weights.index)
    common_assets = returns.columns.intersection(weights.columns)

    returns = returns.loc[common_index, common_assets]
    weights = weights.loc[common_index, common_assets]

    shifted_weights = weights.shift(1).dropna()
    returns = returns.loc[shifted_weights.index]

    gross_returns = (shifted_weights * returns).sum(axis=1)

    turnover = shifted_weights.diff().abs().sum(axis=1).fillna(0.0)
    transaction_cost = turnover * (transaction_cost_bps / 10000.0)
    net_returns = gross_returns - transaction_cost

    result = pd.DataFrame(
        {
            "gross_return": gross_returns,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "net_return": net_returns,
        }
    )
    result["cumulative_return"] = (1 + result["net_return"]).cumprod()
    return result
