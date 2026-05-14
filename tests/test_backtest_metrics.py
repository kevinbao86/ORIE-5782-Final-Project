import numpy as np
import pandas as pd

from volrisk.backtest import backtest_portfolio
from volrisk.metrics import max_drawdown, compare_strategies
from volrisk.portfolio import equal_weight


def test_backtest_portfolio_outputs_expected_columns():
    prices = pd.DataFrame(
        {
            "A": [100, 101, 102, 103],
            "B": [100, 99, 100, 101],
        },
        index=pd.date_range("2020-01-01", periods=4),
    )
    weights = equal_weight(prices.index, ["A", "B"])
    result = backtest_portfolio(prices, weights)
    assert {"gross_return", "turnover", "transaction_cost", "net_return", "cumulative_return"}.issubset(result.columns)
    assert len(result) > 0


def test_max_drawdown():
    cumulative = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert np.isclose(max_drawdown(cumulative), -0.25)


def test_compare_strategies():
    result = pd.DataFrame(
        {
            "net_return": [0.01, -0.005, 0.002],
            "turnover": [0.0, 0.1, 0.1],
            "cumulative_return": [1.01, 1.00495, 1.0069599],
        }
    )
    table = compare_strategies({"demo": result})
    assert "sharpe_ratio" in table.columns
    assert "demo" in table.index
