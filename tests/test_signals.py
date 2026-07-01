"""Tests for src/signals.py."""
import numpy as np
import pandas as pd

from src.signals import calc_zscore, generate_signals


def test_calc_zscore_warm_up_is_nan_and_center_is_finite():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    spread = pd.Series(np.linspace(0, 10, 100), index=dates)

    zscore = calc_zscore(spread, window=20)

    assert zscore.name == "ZScore"
    assert zscore.iloc[:19].isna().all()
    assert zscore.iloc[19:].notna().all()


def test_generate_signals_enters_and_exits_as_expected():
    # Hand-built z-score path: flat -> short entry -> hold -> exit -> long entry -> exit.
    zscores = [0.5, 2.5, 2.1, 1.9, 0.4, -0.1, -2.2, -2.0, 0.1]
    dates = pd.date_range("2020-01-01", periods=len(zscores), freq="B")
    zscore = pd.Series(zscores, index=dates)

    position = generate_signals(zscore, entry_z=2.0, exit_z=0.0)

    expected = [0, -1, -1, -1, -1, 0, 1, 1, 0]
    assert position.tolist() == expected
    assert position.name == "Position"


def test_generate_signals_handles_leading_nan_from_warmup():
    zscore = pd.Series([np.nan, np.nan, 2.5, 0.0], index=pd.date_range("2020-01-01", periods=4, freq="B"))
    position = generate_signals(zscore, entry_z=2.0, exit_z=0.0)
    assert position.tolist() == [0, 0, -1, 0]
