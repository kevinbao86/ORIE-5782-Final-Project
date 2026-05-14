from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized geometric return."""
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    total = (1 + returns).prod()
    years = len(returns) / periods_per_year
    return total ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized volatility."""
    return returns.dropna().std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio."""
    excess = returns.dropna() - risk_free_rate / periods_per_year
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return excess.mean() / vol * np.sqrt(periods_per_year)


def max_drawdown(cumulative_return: pd.Series) -> float:
    """Compute maximum drawdown from cumulative return series."""
    running_max = cumulative_return.cummax()
    drawdown = cumulative_return / running_max - 1
    return float(drawdown.min())


def summarize_backtest(result: pd.DataFrame) -> dict[str, float]:
    """Summarize one backtest result DataFrame."""
    r = result["net_return"]
    ann_ret = annualized_return(r)
    ann_vol = annualized_volatility(r)
    mdd = max_drawdown(result["cumulative_return"])
    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe_ratio(r),
        "max_drawdown": mdd,
        "calmar_ratio": ann_ret / abs(mdd) if mdd < 0 else float("nan"),
        "average_turnover": result["turnover"].mean(),
        "final_cumulative_return": result["cumulative_return"].iloc[-1],
    }


def compare_strategies(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a metrics table for multiple strategies."""
    rows = {name: summarize_backtest(bt) for name, bt in results.items()}
    return pd.DataFrame(rows).T
