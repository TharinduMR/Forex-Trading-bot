import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live_engine import has_open_position_for_symbol


class DummyPosition:
    def __init__(self, symbol):
        self.symbol = symbol


def test_has_open_position_for_symbol_matches_symbol():
    positions = [DummyPosition("EURUSD"), DummyPosition("XAUUSD")]

    assert has_open_position_for_symbol(positions, "EURUSD") is True
    assert has_open_position_for_symbol(positions, "GBPUSD") is False
