import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live_engine import infer_signal_from_probabilities


def test_directional_signal_is_emitted_when_edge_is_small():
    probs = [0.38, 0.30, 0.32]
    signal_code, confidence = infer_signal_from_probabilities(probs, confidence_threshold=0.35, edge_margin=0.02, min_directional_prob=0.32)
    assert signal_code == -1.0
    assert confidence > 0.3


def test_flat_signal_is_emitted_when_flat_is_strongest():
    probs = [0.25, 0.45, 0.30]
    signal_code, confidence = infer_signal_from_probabilities(probs, confidence_threshold=0.35, edge_margin=0.02, min_directional_prob=0.32)
    assert signal_code == 0.0
    assert confidence >= 0.45
