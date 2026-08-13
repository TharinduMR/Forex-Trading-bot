"""
Unit tests for features.py to verify 0% lookahead bias and numerical correctness.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import features


@pytest.fixture
def sample_df():
    """Create a synthetic OHLCV dataframe with 100 bars."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=100, freq="15min")
    close = 2300.0 + np.cumsum(np.random.randn(100) * 2.0)
    high = close + np.random.uniform(0.5, 3.0, 100)
    low = close - np.random.uniform(0.5, 3.0, 100)
    open_p = close + np.random.randn(100) * 0.5
    volume = np.random.randint(100, 5000, 100).astype(float)
    
    df = pd.DataFrame({
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'tick_volume': volume
    }, index=dates)
    return df


def test_get_confirmed_swings_no_lookahead(sample_df):
    """
    Test that modifying price data after timestamp t does NOT alter confirmed swing values at timestamp t.
    """
    s_highs_orig, s_lows_orig = features.get_confirmed_swings(sample_df, k=2)
    
    # Modify future bars from index 50 onwards (simulate a massive spike in the future)
    df_modified = sample_df.copy()
    df_modified.iloc[50:, df_modified.columns.get_loc('high')] += 500.0
    df_modified.iloc[50:, df_modified.columns.get_loc('low')] -= 500.0
    
    s_highs_mod, s_lows_mod = features.get_confirmed_swings(df_modified, k=2)
    
    # Values from index 0 to 49 MUST be identical!
    np.testing.assert_array_equal(
        s_highs_orig.iloc[:50].values,
        s_highs_mod.iloc[:50].values,
        err_msg="Lookahead bias detected in swing highs!"
    )
    np.testing.assert_array_equal(
        s_lows_orig.iloc[:50].values,
        s_lows_mod.iloc[:50].values,
        err_msg="Lookahead bias detected in swing lows!"
    )


def test_calculate_fvg_no_lookahead(sample_df):
    """
    Test that FVG calculation at timestamp t does not depend on future candles.
    """
    atr = features.calculate_atr(sample_df)
    df_orig = features.calculate_fvg(sample_df.copy(), atr)
    
    # Modify future candles from index 60 onwards
    df_mod = sample_df.copy()
    df_mod.iloc[60:, df_mod.columns.get_loc('low')] += 200.0  # Eliminate any future bull FVGs
    df_mod_res = features.calculate_fvg(df_mod, atr)
    
    # FVG features up to index 59 must be identical
    for col in ['bull_fvg_size', 'bear_fvg_size', 'dist_bull_fvg', 'dist_bear_fvg', 'fvg_active']:
        np.testing.assert_array_almost_equal(
            df_orig[col].iloc[:60].values,
            df_mod_res[col].iloc[:60].values,
            err_msg=f"Lookahead bias detected in FVG feature '{col}'!"
        )


def test_merge_timeframes_no_lookahead():
    """
    Test that merging an H1 dataframe onto an M15 dataframe shifts H1 features by 1 bar
    so that an M15 candle at 10:15 does NOT see the 10:00 H1 candle (which closes at 11:00).
    """
    ltf_dates = pd.date_range("2026-01-01 09:00:00", periods=8, freq="15min")
    ltf_df = pd.DataFrame({'close': range(8)}, index=ltf_dates)
    
    # H1 bars at 09:00 (closes at 10:00) and 10:00 (closes at 11:00)
    htf_dates = pd.to_datetime(["2026-01-01 09:00:00", "2026-01-01 10:00:00"])
    htf_df = pd.DataFrame({'pd_zone': [10.0, 20.0]}, index=htf_dates)
    
    merged = features.merge_timeframes(ltf_df, htf_df, prefix="htf_")
    
    # At 09:00, 09:15, 09:30, 09:45, there is no previous completed H1 bar (index 0 is 09:00, shifted by 1 gives NaN -> filled to 0)
    assert merged.loc["2026-01-01 09:15:00", "htf_pd_zone"] == 0.0
    
    # At 10:00, 10:15, 10:30, 10:45, the only completed H1 bar is the 09:00 bar (value 10.0)!
    # It must NOT see value 20.0 (from 10:00 bar which closes at 11:00).
    assert merged.loc["2026-01-01 10:15:00", "htf_pd_zone"] == 10.0
    assert merged.loc["2026-01-01 10:45:00", "htf_pd_zone"] == 10.0
