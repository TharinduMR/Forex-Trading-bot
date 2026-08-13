"""
Unit tests for labeling.py to verify Triple-Barrier target generation and NaN handling.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import labeling


def test_triple_barrier_hits():
    """Test that clear directional moves trigger correct barrier labels (+1, -1, 0, NaN)."""
    n = 50
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    
    close = np.full(n, 2000.0)
    high = np.full(n, 2001.0)
    low = np.full(n, 1999.0)
    
    # At index 5, let price shoot up by 50 points over the next 3 bars (hit Take Profit)
    close[6:10] = [2010.0, 2030.0, 2060.0, 2060.0]
    high[6:10] = [2012.0, 2032.0, 2065.0, 2065.0]
    
    # At index 20, let price drop by 50 points over the next 3 bars (hit Stop Loss)
    close[21:25] = [1990.0, 1970.0, 1940.0, 1940.0]
    low[21:25] = [1988.0, 1968.0, 1935.0, 1935.0]
    
    df = pd.DataFrame({'close': close, 'high': high, 'low': low}, index=dates)
    
    # Apply triple barrier with max_holding=10
    df_labeled = labeling.apply_triple_barrier(df, tp_mult=2.0, sl_mult=1.5, max_holding=10)
    
    # At index 5, tp barrier should be triggered -> target 1.0
    assert df_labeled.loc[dates[5], 'target'] == 1.0, f"Expected 1.0 at index 5, got {df_labeled.loc[dates[5], 'target']}"
    
    # At index 20, sl barrier should be triggered -> target -1.0
    assert df_labeled.loc[dates[20], 'target'] == -1.0, f"Expected -1.0 at index 20, got {df_labeled.loc[dates[20], 'target']}"
    
    # At index 30, price stays flat at 2000 for 10 bars -> target 0.0 (Time stop)
    assert df_labeled.loc[dates[30], 'target'] == 0.0, f"Expected 0.0 at index 30, got {df_labeled.loc[dates[30], 'target']}"
    
    # At the tail (last 10 bars from index 40 to 49), target must be NaN!
    assert np.isnan(df_labeled.loc[dates[42], 'target']), "Expected NaN at tail where horizon is incomplete!"
    assert np.isnan(df_labeled.loc[dates[49], 'target']), "Expected NaN at last bar!"


def test_sample_weights():
    """Test sample weight computation."""
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    df = pd.DataFrame({
        'close': 2000.0 + np.random.randn(n) * 5.0,
        'high': 2005.0 + np.random.randn(n) * 5.0,
        'low': 1995.0 + np.random.randn(n) * 5.0,
        'target': np.random.choice([-1.0, 0.0, 1.0], size=n)
    }, index=dates)
    
    weights = labeling.compute_sample_weights(df, 'target')
    
    assert len(weights) == n
    assert not weights.isna().any(), "Sample weights should not contain NaNs!"
    assert np.isclose(weights.mean(), 1.0, atol=0.2), "Normalized weights should have mean ~1.0"
