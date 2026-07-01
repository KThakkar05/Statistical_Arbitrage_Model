"""Plotting helpers for pairs trading diagnostics: the OLS spread with an ADF
readout, the spread/z-score pair, and the three-panel backtest summary.

Kept separate from the analytical modules (cointegration/signals/backtest/
metrics) so those stay import-light and easy to unit test without matplotlib.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

_BG = "#0f1117"
_PANEL = "#1a1a2e"


def plot_adf_spread(spread: pd.Series, adf_result: dict, ticker_1: str, ticker_2: str,
                     start: str, end: str, save_path: str | None = None):
    """Plot an OLS spread with mean/1-sigma/2-sigma bands and an ADF readout.

    Args:
        spread: Residual spread series from ``cointegration.calc_spread``.
        adf_result: Dict returned by ``cointegration.adf_test``.
        ticker_1: Dependent ticker (for the title).
        ticker_2: Independent ticker (for the title).
        start: Start date string (for the title).
        end: End date string (for the title).
        save_path: File path to save the figure. Pass ``None`` to skip saving.
    """
    mean, std = spread.mean(), spread.std()
    p_value = adf_result["p_value"]
    sig_color = "#27ae60" if p_value < 0.05 else "#e74c3c"

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    ax.fill_between(spread.index, mean + std, mean + 2 * std, color="#2c7bb6", alpha=0.08)
    ax.fill_between(spread.index, mean - std, mean - 2 * std, color="#2c7bb6", alpha=0.08)
    ax.fill_between(spread.index, mean - std, mean + std, color="#2c7bb6", alpha=0.04)
    ax.fill_between(spread.index, spread, mean, where=(spread >= mean),
                     color="#27ae60", alpha=0.18, interpolate=True)
    ax.fill_between(spread.index, spread, mean, where=(spread < mean),
                     color="#e74c3c", alpha=0.18, interpolate=True)

    ax.plot(spread, color="#4fc3f7", linewidth=1.0, zorder=3)
    ax.axhline(mean, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.5, label="Mean")
    ax.axhline(mean + std, color="#2c7bb6", linewidth=0.6, linestyle=":", alpha=0.7, label="±1σ")
    ax.axhline(mean - std, color="#2c7bb6", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.axhline(mean + 2 * std, color="#5b8dd9", linewidth=0.6, linestyle=":", alpha=0.5, label="±2σ")
    ax.axhline(mean - 2 * std, color="#5b8dd9", linewidth=0.6, linestyle=":", alpha=0.5)

    ax.set_title(f"OLS Spread  ·  {ticker_1} / {ticker_2}  ·  {start} → {end}",
                 color="#e0e0e0", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Spread", color="#9e9e9e", fontsize=10)
    ax.tick_params(colors="#9e9e9e", labelsize=8)
    ax.tick_params(axis="x", rotation=30)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#ffffff", alpha=0.05, linewidth=0.6)

    ax.annotate(
        f"ADF stat: {adf_result['adf_stat']:.3f}   p-value: {p_value:.4f}",
        xy=(0.01, 0.04), xycoords="axes fraction", color=sig_color, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc=_PANEL, ec=sig_color, alpha=0.85),
    )
    ax.legend(fontsize=8, facecolor=_PANEL, edgecolor="#333333", labelcolor="#cccccc", loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.show()


def plot_spread_zscore(spread: pd.Series, zscore: pd.Series, save_path: str | None = "charts/zscore_signal.png"):
    """Two-panel plot: raw spread on top, rolling z-score below.

    Args:
        spread: Residual spread series from ``cointegration.calc_spread``.
        zscore: Z-score series from ``signals.calc_zscore``.
        save_path: File path to save the figure. Pass ``None`` to skip saving.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor(_BG)

    s_mean, s_std = spread.mean(), spread.std()

    ax0 = axes[0]
    ax0.set_facecolor(_BG)
    ax0.fill_between(spread.index, spread, s_mean, where=(spread >= s_mean),
                      color="#27ae60", alpha=0.18, interpolate=True)
    ax0.fill_between(spread.index, spread, s_mean, where=(spread < s_mean),
                      color="#e74c3c", alpha=0.18, interpolate=True)
    ax0.fill_between(spread.index, s_mean - s_std, s_mean + s_std, color="#2c7bb6", alpha=0.06)
    ax0.plot(spread, color="#4fc3f7", linewidth=1.0, zorder=3)
    ax0.axhline(s_mean, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.5, label="Mean")
    ax0.axhline(s_mean + s_std, color="#2c7bb6", linewidth=0.6, linestyle=":", alpha=0.7, label="±1σ")
    ax0.axhline(s_mean - s_std, color="#2c7bb6", linewidth=0.6, linestyle=":", alpha=0.7)
    ax0.set_title(f"{spread.name or 'Spread'}  ·  Raw Spread (residuals)",
                  color="#e0e0e0", fontsize=12, fontweight="bold", pad=10)
    ax0.set_ylabel("Spread", color="#9e9e9e", fontsize=10)
    ax0.tick_params(colors="#9e9e9e", labelsize=8)
    ax0.legend(fontsize=8, facecolor=_PANEL, edgecolor="#333333", labelcolor="#cccccc", loc="upper right")
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.grid(axis="y", color="#ffffff", alpha=0.05, linewidth=0.6)

    ax1 = axes[1]
    ax1.set_facecolor(_BG)
    ax1.fill_between(zscore.index, 2, zscore.where(zscore > 2), color="#e74c3c", alpha=0.20, interpolate=True)
    ax1.fill_between(zscore.index, -2, zscore.where(zscore < -2), color="#27ae60", alpha=0.20, interpolate=True)
    ax1.plot(zscore, color="#4fc3f7", linewidth=1.0, zorder=3)
    ax1.axhline(2, color="#e74c3c", linewidth=1.0, linestyle="--", label="Entry +2σ")
    ax1.axhline(-2, color="#27ae60", linewidth=1.0, linestyle="--", label="Entry −2σ")
    ax1.axhline(0, color="#ffffff", linewidth=0.6, linestyle="-", alpha=0.3)
    ax1.set_title("Rolling Z-Score", color="#e0e0e0", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Z-Score", color="#9e9e9e", fontsize=10)
    ax1.tick_params(colors="#9e9e9e", labelsize=8)
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(fontsize=8, facecolor=_PANEL, edgecolor="#333333", labelcolor="#cccccc", loc="upper right")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.grid(axis="y", color="#ffffff", alpha=0.05, linewidth=0.6)

    fig.subplots_adjust(hspace=0.12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.show()


def plot_backtest(pnl_df: pd.DataFrame, save_path: str | None = "charts/backtest.png"):
    """Three-panel backtest summary: z-score, position, and cumulative P&L.

    Args:
        pnl_df: DataFrame returned by ``backtest.run_backtest`` with columns
            ``ZScore``, ``Position``, and ``CumulativePnL``, indexed by date.
        save_path: File path to save the figure. Pass ``None`` to skip saving.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor(_BG)

    ax0 = axes[0]
    ax0.set_facecolor(_BG)
    ax0.fill_between(pnl_df.index, 2, pnl_df["ZScore"].where(pnl_df["ZScore"] > 2),
                      color="#e74c3c", alpha=0.20, interpolate=True)
    ax0.fill_between(pnl_df.index, -2, pnl_df["ZScore"].where(pnl_df["ZScore"] < -2),
                      color="#27ae60", alpha=0.20, interpolate=True)
    ax0.plot(pnl_df.index, pnl_df["ZScore"], color="#4fc3f7", linewidth=0.9, zorder=3)
    ax0.axhline(2, color="#e74c3c", linewidth=1.0, linestyle="--", label="Entry +2σ")
    ax0.axhline(-2, color="#27ae60", linewidth=1.0, linestyle="--", label="Entry −2σ")
    ax0.axhline(0, color="#ffffff", linewidth=0.6, linestyle="-", alpha=0.3)
    ax0.set_title("Rolling Z-Score", color="#e0e0e0", fontsize=12, fontweight="bold", pad=10)
    ax0.set_ylabel("Z-Score", color="#9e9e9e", fontsize=10)
    ax0.tick_params(colors="#9e9e9e", labelsize=8)
    ax0.legend(fontsize=8, facecolor=_PANEL, edgecolor="#333333", labelcolor="#cccccc", loc="upper right")
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.grid(axis="y", color="#ffffff", alpha=0.05, linewidth=0.6)

    ax1 = axes[1]
    ax1.set_facecolor(_BG)
    ax1.fill_between(pnl_df.index, 0, pnl_df["Position"], where=(pnl_df["Position"] > 0),
                      color="#27ae60", alpha=0.30, interpolate=True, label="Long")
    ax1.fill_between(pnl_df.index, 0, pnl_df["Position"], where=(pnl_df["Position"] < 0),
                      color="#e74c3c", alpha=0.30, interpolate=True, label="Short")
    ax1.plot(pnl_df.index, pnl_df["Position"], color="#b39ddb", linewidth=0.8, zorder=3)
    ax1.axhline(0, color="#ffffff", linewidth=0.6, linestyle="-", alpha=0.3)
    ax1.set_title("Position  (1 = long spread, −1 = short spread, 0 = flat)",
                  color="#e0e0e0", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Position", color="#9e9e9e", fontsize=10)
    ax1.tick_params(colors="#9e9e9e", labelsize=8)
    ax1.legend(fontsize=8, facecolor=_PANEL, edgecolor="#333333", labelcolor="#cccccc", loc="upper right")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.grid(axis="y", color="#ffffff", alpha=0.05, linewidth=0.6)

    ax2 = axes[2]
    ax2.set_facecolor(_BG)
    final_pnl = pnl_df["CumulativePnL"].iloc[-1]
    pnl_color = "#27ae60" if final_pnl >= 0 else "#e74c3c"
    ax2.fill_between(pnl_df.index, 0, pnl_df["CumulativePnL"], where=(pnl_df["CumulativePnL"] >= 0),
                      color="#27ae60", alpha=0.18, interpolate=True)
    ax2.fill_between(pnl_df.index, 0, pnl_df["CumulativePnL"], where=(pnl_df["CumulativePnL"] < 0),
                      color="#e74c3c", alpha=0.18, interpolate=True)
    ax2.plot(pnl_df.index, pnl_df["CumulativePnL"], color=pnl_color, linewidth=1.1, zorder=3)
    ax2.axhline(0, color="#ffffff", linewidth=0.6, linestyle="-", alpha=0.3)
    ax2.annotate(
        f"Final P&L: {final_pnl:+.2f}p",
        xy=(0.01, 0.06), xycoords="axes fraction", color=pnl_color, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc=_PANEL, ec=pnl_color, alpha=0.85),
    )
    ax2.set_title("Cumulative P&L (pre-cost)", color="#e0e0e0", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("P&L", color="#9e9e9e", fontsize=10)
    ax2.tick_params(colors="#9e9e9e", labelsize=8)
    ax2.tick_params(axis="x", rotation=30)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.grid(axis="y", color="#ffffff", alpha=0.05, linewidth=0.6)

    fig.subplots_adjust(hspace=0.14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.show()
