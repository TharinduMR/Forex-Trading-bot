"""
Unit tests for backtester.py to verify event-driven execution, PnL accounting, and metrics calculation.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backtester


def test_backtester_execution_and_pnl():
    """Test that Backtester executes trades, respects stop loss / take profit, and tracks equity."""
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    
    close = np.full(n, 2000.0)
    high = np.full(n, 2001.0)
    low = np.full(n, 1999.0)
    open_p = np.full(n, 2000.0)
    
    # Let price jump to 2050 at bar 10 to hit Take Profit on a long trade entered at bar 5
    close[10] = 2050.0
    high[10] = 2055.0
    
    df = pd.DataFrame({
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'trend_direction': np.ones(n),
        'choch_flag': np.zeros(n)
    }, index=dates)
    
    preds = np.zeros(n)
    preds[4] = 1.0  # Buy signal at bar 4 close -> entry at bar 5 open (price 2000)
    probs = np.full(n, 0.80)
    
    bt = backtester.Backtester(initial_capital=10000.0, spread_dollar=0.20, slippage_dollar=0.05, commission_per_lot=5.0)
    equity_curve, df_trades = bt.run_simulation(df, preds, probs)
    
    assert len(equity_curve) == n, "Equity curve length must match dataframe length!"
    assert not df_trades.empty, "Should have executed at least 1 trade!"
    
    # Check trade details
    trade = df_trades.iloc[0]
    assert trade['type'] == 'LONG'
    assert trade['reason'] == 'TAKE_PROFIT'
    assert trade['pnl'] > 0, "Trade should be profitable after hitting Take Profit!"
    assert equity_curve[-1] > 10000.0, "Ending equity must exceed initial capital!"
