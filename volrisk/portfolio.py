from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(raw_weights: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Normalize nonnegative weights so each row sums to one."""
    if (raw_weights < 0).any().any():
        raise ValueError("weights must be nonnegative")

    if isinstance(raw_weights, pd.Series):
        total = raw_weights.sum()
        if total <= 0 or np.isnan(total):
            return pd.Series(1 / len(raw_weights), index=raw_weights.index)
        return raw_weights / total

    totals = raw_weights.sum(axis=1)
    normalized = raw_weights.div(totals.replace(0, np.nan), axis=0)
    return normalized.fillna(1 / raw_weights.shape[1])


def equal_weight(index: pd.Index, assets: list[str]) -> pd.DataFrame:
    """Create equal weights for all dates and assets."""
    if len(assets) == 0:
        raise ValueError("assets must be non-empty")
    return pd.DataFrame(1 / len(assets), index=index, columns=assets)


def inverse_vol_weights(
    vol_forecasts: pd.DataFrame,
    min_vol: float = 1e-4,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """Compute inverse-volatility portfolio weights."""
    clipped = vol_forecasts.clip(lower=min_vol)
    raw = 1.0 / clipped
    weights = normalize_weights(raw)

    if max_weight is not None:
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        weights = weights.clip(upper=max_weight)
        weights = normalize_weights(weights)

    return weights


def apply_rebalance_frequency(weights: pd.DataFrame, frequency: str = "M") -> pd.DataFrame:
    """Convert daily signal weights into periodic rebalanced weights.

    Weights are sampled at period ends and forward-filled.
    """
    if weights.empty:
        raise ValueError("weights must be non-empty")
    sampled = weights.resample(frequency).last()
    rebalanced = sampled.reindex(weights.index).ffill()
    return rebalanced.dropna(how="all")
