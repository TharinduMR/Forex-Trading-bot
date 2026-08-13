import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from risk_manager import RiskManager


def test_risk_manager_blocks_after_daily_loss_limit(tmp_path):
    state_file = tmp_path / "risk_state.json"
    manager = RiskManager(max_daily_loss=100.0, max_consecutive_losses=3, state_file=str(state_file))

    manager.record_trade_result(pnl=-120.0, current_bar=2)

    can_trade, reason = manager.can_trade(current_bar=3)
    assert not can_trade
    assert reason == "daily_loss_limit"

    reloaded = RiskManager(max_daily_loss=100.0, max_consecutive_losses=3, state_file=str(state_file))
    can_trade, reason = reloaded.can_trade(current_bar=3)
    assert not can_trade
    assert reason == "daily_loss_limit"


def test_risk_manager_blocks_after_consecutive_losses(tmp_path):
    state_file = tmp_path / "risk_state.json"
    manager = RiskManager(max_daily_loss=1000.0, max_consecutive_losses=2, state_file=str(state_file))

    manager.record_trade_result(pnl=-10.0, current_bar=1)
    manager.record_trade_result(pnl=-10.0, current_bar=2)

    can_trade, reason = manager.can_trade(current_bar=3)
    assert not can_trade
    assert reason == "consecutive_loss_limit"


def test_risk_manager_blocks_after_drawdown_limit(tmp_path):
    state_file = tmp_path / "risk_state.json"
    manager = RiskManager(max_daily_loss=1000.0, max_consecutive_losses=5, max_drawdown_pct=0.10, state_file=str(state_file))

    manager.record_trade_result(pnl=0.0, current_bar=1, equity=1000.0)
    can_trade, reason = manager.can_trade(current_bar=2, equity=850.0)

    assert not can_trade
    assert reason == "drawdown_limit"
