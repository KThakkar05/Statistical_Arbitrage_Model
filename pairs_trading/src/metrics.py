"""Performance statistics for a pairs trading backtest.

Sharpe ratio, max drawdown, win rate, payoff ratio, and average trade
duration, plus a convenience ``perf_metrics`` that bundles them all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(daily_pnl: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe ratio of a daily P&L series (assumes zero risk-free rate).

    Args:
        daily_pnl: Daily P&L series (e.g. ``pnl_df["DailyPnL"]``).
        periods_per_year: Annualization factor. Default 252 trading days.

    Returns:
        Annualized Sharpe ratio, or ``nan`` if the series is empty or has
        zero variance.
    """
    if daily_pnl.empty or daily_pnl.std() == 0:
        return float("nan")
    return (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(periods_per_year)


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Largest peak-to-trough drawdown of a cumulative P&L series.

    Args:
        cumulative_pnl: Cumulative P&L series (e.g. ``pnl_df["CumulativePnL"]``).

    Returns:
        Largest drawdown as a positive number, or ``nan`` if the series is empty.
    """
    if cumulative_pnl.empty:
        return float("nan")
    return (cumulative_pnl.cummax() - cumulative_pnl).max()


def win_rate(trade_pnl: pd.Series) -> float:
    """Fraction of trades with positive P&L.

    Args:
        trade_pnl: Per-trade P&L series (e.g. ``trade_log["Profit/Loss"]``).

    Returns:
        Win rate in ``[0, 1]``, or ``nan`` if there are no trades.
    """
    if trade_pnl.empty:
        return float("nan")
    return (trade_pnl > 0).sum() / len(trade_pnl)


def payoff_ratio(trade_pnl: pd.Series) -> float:
    """Ratio of average winning trade size to average losing trade size.

    Args:
        trade_pnl: Per-trade P&L series.

    Returns:
        ``avg_win / abs(avg_loss)``, or ``nan`` if there are no losing trades
        (or no trades at all).
    """
    avg_win = trade_pnl[trade_pnl > 0].mean()
    avg_loss = trade_pnl[trade_pnl < 0].mean()
    if pd.isna(avg_loss) or avg_loss == 0:
        return float("nan")
    return avg_win / abs(avg_loss)


def avg_trade_duration(trade_log: pd.DataFrame) -> float:
    """Average holding period of completed trades, in calendar days.

    Args:
        trade_log: Trade log with ``Entry_Date`` and ``Exit_Date`` columns.

    Returns:
        Mean duration in days, or ``nan`` if there are no trades or the
        required columns are missing.
    """
    if trade_log.empty or "Entry_Date" not in trade_log or "Exit_Date" not in trade_log:
        return float("nan")
    return (trade_log["Exit_Date"] - trade_log["Entry_Date"]).dt.days.mean()


def perf_metrics(pnl_df: pd.DataFrame, trade_log: pd.DataFrame, cost: bool = False) -> dict:
    """Compute a summary dict of backtest performance statistics.

    Args:
        pnl_df: Daily P&L frame from ``backtest.run_backtest`` (or
            ``walk_forward_backtest``), with ``DailyPnL``/``GrossPnL`` and
            ``CumulativePnL``/``GrossCumPnL`` columns.
        trade_log: Round-trip trade log from the same backtest run, with a
            ``Profit/Loss`` (and optionally ``GrossPnL``) column.
        cost: If ``True``, use the post-slippage ("Gross...") columns.

    Returns:
        Dict with keys ``sharpe_ratio``, ``max_drawdown``, ``win_rate``,
        ``payoff_ratio``, ``avg_trade_duration_days``, ``total_pnl``,
        ``n_trades``.
    """
    pnl_col = "GrossPnL" if cost else "DailyPnL"
    cum_col = "GrossCumPnL" if cost else "CumulativePnL"
    trade_pnl_col = "GrossPnL" if (cost and "GrossPnL" in trade_log) else "Profit/Loss"
    trade_pnl = trade_log[trade_pnl_col] if trade_pnl_col in trade_log else pd.Series(dtype=float)

    return {
        "sharpe_ratio": sharpe_ratio(pnl_df[pnl_col]),
        "max_drawdown": max_drawdown(pnl_df[cum_col]),
        "win_rate": win_rate(trade_pnl),
        "payoff_ratio": payoff_ratio(trade_pnl),
        "avg_trade_duration_days": avg_trade_duration(trade_log),
        "total_pnl": pnl_df[pnl_col].sum() if not pnl_df.empty else float("nan"),
        "n_trades": len(trade_log),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dict returned by ``perf_metrics``.

    Args:
        metrics: Dict returned by ``perf_metrics``.
    """
    print(f"Sharpe Ratio   : {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown   : {metrics['max_drawdown']:.4f}")
    print(f"Win Rate       : {metrics['win_rate']:.2%}")
    print(f"Payoff Ratio   : {metrics['payoff_ratio']:.4f}")
    print(f"Trade Duration : {metrics['avg_trade_duration_days']:.2f} days")
    print(f"Total P&L      : {metrics['total_pnl']:.2f}")
    print(f"Trades         : {metrics['n_trades']}")
