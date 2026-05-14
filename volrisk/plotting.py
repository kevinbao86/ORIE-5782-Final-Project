from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative_returns(results: dict[str, pd.DataFrame], out_path: str | Path) -> None:
    """Plot cumulative returns for all strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, result in results.items():
        ax.plot(result.index, result["cumulative_return"], label=name)
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cumulative_returns_common_start(
    results: dict[str, pd.DataFrame],
    out_path: str | Path,
) -> None:
    """Plot cumulative returns after rebasing all strategies to the same start date."""
    if not results:
        raise ValueError("results must be non-empty")

    common_start = max(result.index.min() for result in results.values())
    common_end = min(result.index.max() for result in results.values())
    if common_start > common_end:
        raise ValueError("results do not have an overlapping date range")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, result in results.items():
        common_result = result.loc[common_start:common_end]
        cumulative = (1 + common_result["net_return"]).cumprod()
        ax.plot(cumulative.index, cumulative, label=name)

    ax.set_title("Cumulative Returns (Common Start)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drawdowns(results: dict[str, pd.DataFrame], out_path: str | Path) -> None:
    """Plot drawdowns for all strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, result in results.items():
        cumulative = result["cumulative_return"]
        drawdown = cumulative / cumulative.cummax() - 1
        ax.plot(drawdown.index, drawdown, label=name)
    ax.set_title("Portfolio Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
