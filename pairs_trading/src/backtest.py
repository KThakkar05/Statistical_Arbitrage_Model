"""Backtest engine for the z-score pairs trading strategy.

Provides a single-pair backtest given a fixed hedge ratio (``run_backtest``),
and a walk-forward loop that re-estimates the hedge ratio on a rolling
training window and evaluates strictly out-of-sample (``walk_forward_backtest``).
"""
from __future__ import annotations

import pandas as pd

from .cointegration import calc_spread, fit_ols, load_prices
from .signals import calc_zscore, generate_signals

# Two-sided slippage assumption: 5bps applied to each leg, on both entry and exit.
SLIPPAGE_BPS = 0.0005


def run_backtest(
    series_y: pd.Series,
    series_x: pd.Series,
    alpha: float,
    beta: float,
    window: int,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a single-pair backtest given a fixed (already-fitted) hedge ratio.

    Computes the spread, rolling z-score, and target positions, then marks
    daily P&L to the day-over-day change in spread while a position is held.
    Also reconstructs a round-trip trade log with an approximate slippage
    cost charged on both entry and exit.

    Args:
        series_y: Dependent price series to trade over.
        series_x: Independent price series to trade over.
        alpha: OLS intercept (typically fit on a separate in-sample window).
        beta: OLS hedge ratio.
        window: Rolling lookback (days) used for the z-score.
        entry_z: Absolute z-score threshold to open a position.
        exit_z: Z-score level at which to flatten a position.

    Returns:
        Tuple ``(pnl_df, trade_log)``:
            - ``pnl_df``: daily frame indexed by date with columns ``ZScore``,
              ``Spread``, ``Position``, ``DailyPnL``, ``Slippage``,
              ``GrossPnL``, ``CumulativePnL``, ``GrossCumPnL``.
            - ``trade_log``: one row per completed round-trip trade, with
              ``Entry_Date``, ``Direction``, ``Entry_Spread``, ``Exit_Date``,
              ``Exit_Spread``, ``Slippage``, ``Profit/Loss``, ``GrossPnL``.
    """
    spread = calc_spread(series_y, series_x, alpha, beta)
    zscore = calc_zscore(spread, window)
    position = generate_signals(zscore, entry_z, exit_z)

    prev_position = position.shift(1).fillna(0)
    prev_spread = spread.shift(1)
    daily_pnl = ((spread - prev_spread) * prev_position).fillna(0.0)

    position_change = position.diff().fillna(position).abs()
    slippage = position_change * (series_y * SLIPPAGE_BPS + series_x * SLIPPAGE_BPS)

    pnl_df = pd.DataFrame({
        "ZScore": zscore,
        "Spread": spread,
        "Position": position,
        "DailyPnL": daily_pnl,
        "Slippage": slippage,
    })
    pnl_df["GrossPnL"] = pnl_df["DailyPnL"] - pnl_df["Slippage"]
    pnl_df["CumulativePnL"] = pnl_df["DailyPnL"].cumsum()
    pnl_df["GrossCumPnL"] = pnl_df["GrossPnL"].cumsum()

    trade_log = _build_trade_log(pnl_df)
    return pnl_df, trade_log


def _build_trade_log(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct round-trip trades from a daily position series.

    Args:
        pnl_df: Daily frame with ``Position``, ``Spread`` and ``Slippage``
            columns, as produced by ``run_backtest``.

    Returns:
        DataFrame of completed round-trip trades (see ``run_backtest`` for
        columns). Empty (but correctly-columned) if no trade completed.
    """
    columns = ["Entry_Date", "Direction", "Entry_Spread",
               "Exit_Date", "Exit_Spread", "Slippage",
               "Profit/Loss", "GrossPnL"]

    entries, exits = [], []
    prev_pos = 0
    entry_idx = None

    for i, (date, row) in enumerate(pnl_df.iterrows()):
        pos = row["Position"]
        if prev_pos == 0 and pos != 0:
            entry_idx = i
            entries.append({
                "Entry_Date": date,
                "Direction": "Long" if pos > 0 else "Short",
                "Entry_Spread": row["Spread"],
            })
        elif prev_pos != 0 and pos == 0:
            exits.append({
                "Exit_Date": date,
                "Exit_Spread": row["Spread"],
                "Slippage": pnl_df["Slippage"].iloc[entry_idx:i + 1].sum(),
            })
        prev_pos = pos

    if prev_pos != 0 and entry_idx is not None:
        exits.append({
            "Exit_Date": pnl_df.index[-1],
            "Exit_Spread": pnl_df["Spread"].iloc[-1],
            "Slippage": pnl_df["Slippage"].iloc[entry_idx:].sum(),
        })

    n = min(len(entries), len(exits))
    if n == 0:
        return pd.DataFrame(columns=columns)

    trades = pd.concat(
        [pd.DataFrame(entries[:n]).reset_index(drop=True),
         pd.DataFrame(exits[:n]).reset_index(drop=True)],
        axis=1,
    )
    sign = trades["Direction"].map({"Long": 1, "Short": -1})
    trades["Profit/Loss"] = (trades["Exit_Spread"] - trades["Entry_Spread"]) * sign
    trades["GrossPnL"] = trades["Profit/Loss"] - trades["Slippage"]
    return trades[columns]


def walk_forward_backtest(
    ticker_y: str,
    ticker_x: str,
    start: str,
    end: str,
    train_window: int = 252,
    test_window: int = 63,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roll a training window forward through time, re-fitting the hedge
    ratio on each block and evaluating strictly out-of-sample.

    For each step: fit alpha/beta on ``train_window`` days, then trade the
    following ``test_window`` days using that fixed hedge ratio, before
    sliding forward by ``test_window`` and re-fitting. This avoids the
    look-ahead bias of fitting the hedge ratio once on the full sample.

    Args:
        ticker_y: Dependent ticker.
        ticker_x: Independent ticker.
        start: Start date for the full price history to load.
        end: End date for the full price history to load.
        train_window: Number of days used to estimate alpha/beta each step.
        test_window: Number of out-of-sample days traded before re-fitting.
        zscore_window: Rolling window for the z-score within each test block.
        entry_z: Entry threshold.
        exit_z: Exit threshold.

    Returns:
        Tuple ``(pnl_df, trade_log)`` concatenated across all walk-forward
        blocks, with ``CumulativePnL`` / ``GrossCumPnL`` recomputed over the
        full stitched series. Both are empty DataFrames if the history is too
        short for even one ``train_window + test_window`` block.
    """
    s1, s2 = load_prices(ticker_y, ticker_x, start, end)
    n = len(s1)

    pnl_blocks, trade_blocks = [], []
    start_idx = 0
    while start_idx + train_window + test_window <= n:
        train_y = s1.iloc[start_idx: start_idx + train_window]
        train_x = s2.iloc[start_idx: start_idx + train_window]
        alpha, beta, _ = fit_ols(train_y, train_x)

        # Include zscore_window extra days before the test block purely so the
        # rolling z-score has a warm-up; those warm-up rows are dropped after.
        test_start = max(start_idx + train_window - zscore_window, 0)
        test_end = start_idx + train_window + test_window
        test_y = s1.iloc[test_start:test_end]
        test_x = s2.iloc[test_start:test_end]

        pnl_df, trade_log = run_backtest(test_y, test_x, alpha, beta, zscore_window, entry_z, exit_z)
        warm_up = start_idx + train_window - test_start
        pnl_df = pnl_df.iloc[warm_up:]

        pnl_blocks.append(pnl_df)
        if not trade_log.empty:
            trade_blocks.append(trade_log)

        start_idx += test_window

    pnl_all = pd.concat(pnl_blocks) if pnl_blocks else pd.DataFrame()
    if not pnl_all.empty:
        pnl_all["CumulativePnL"] = pnl_all["DailyPnL"].cumsum()
        pnl_all["GrossCumPnL"] = pnl_all["GrossPnL"].cumsum()

    trades_all = pd.concat(trade_blocks, ignore_index=True) if trade_blocks else pd.DataFrame()
    return pnl_all, trades_all
