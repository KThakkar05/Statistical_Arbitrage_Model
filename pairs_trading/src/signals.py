"""Entry/exit signal generation for a z-score based pairs trading strategy."""
from __future__ import annotations

import pandas as pd


def calc_zscore(spread: pd.Series, window: int) -> pd.Series:
    """Compute the rolling z-score of a spread series.

    Args:
        spread: Residual spread series from ``cointegration.calc_spread``.
        window: Lookback period in trading days for the rolling mean and std.

    Returns:
        ``pd.Series`` of z-score values, named ``"ZScore"``, aligned to
        ``spread``'s index. The first ``window - 1`` values are NaN.
    """
    roll_mean = spread.rolling(window).mean()
    roll_std = spread.rolling(window).std()
    zscore = (spread - roll_mean) / roll_std
    zscore.name = "ZScore"
    return zscore


def generate_signals(zscore: pd.Series, entry_z: float = 2.0, exit_z: float = 0.0) -> pd.Series:
    """Translate a z-score series into a target position at every bar.

    Rules:
        - Enter short the spread (-1) when ``zscore > entry_z``.
        - Enter long the spread (+1) when ``zscore < -entry_z``.
        - Exit (0) once the z-score mean-reverts past ``exit_z``.
        - Otherwise, hold the current position.

    This is a stateful pass over the series rather than a vectorized one,
    because the position at time t depends on the position at t-1, not just
    the current z-score value.

    Args:
        zscore: Rolling z-score series from ``calc_zscore``.
        entry_z: Absolute z-score threshold to open a position.
        exit_z: Z-score level at which to flatten an open position.

    Returns:
        ``pd.Series`` of target positions (``-1``, ``0``, ``+1``), named
        ``"Position"``, aligned to ``zscore``'s index. NaN z-scores (e.g. the
        rolling warm-up period) simply hold the prior position (flat at the
        start).
    """
    position = 0
    positions = []
    for zs in zscore:
        if pd.isna(zs):
            positions.append(position)
            continue
        if position == 0:
            if zs > entry_z:
                position = -1
            elif zs < -entry_z:
                position = 1
        elif position == -1 and zs <= exit_z:
            position = 0
        elif position == 1 and zs >= -exit_z:
            position = 0
        positions.append(position)

    return pd.Series(positions, index=zscore.index, name="Position")
