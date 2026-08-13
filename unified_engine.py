"""
Unified Global Trade Engine — "One Signal, All Timeframes"

Runs in the background, scans ALL configured timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d)
simultaneously, fuses their signals into ONE locked-in trade decision via weighted voting.
The dashboard reads the global state file and displays the same signal, TP, and SL
regardless of which timeframe the user is viewing.

Usage:
    python unified_engine.py                  # default: XAUUSD
    python unified_engine.py --symbol EURUSD  # specify symbol
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import config
from config import get_symbol_profile, resolve_symbol

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [UnifiedEngine] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("UnifiedEngine")

# Lazy import of LivePredictionEngine
_LivePredictionEngine = None


def _get_engine_class():
    global _LivePredictionEngine
    if _LivePredictionEngine is None:
        from live_engine import LivePredictionEngine
        _LivePredictionEngine = LivePredictionEngine
    return _LivePredictionEngine


# Paths
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_STATE_FILE = os.path.join(_BASE_DIR, "global_trade_state.json")
LEGACY_STATE_FILE = os.path.join(_BASE_DIR, "monitor_state.json")

# Timeframe weights for the weighted voting fusion.
TIMEFRAME_WEIGHTS: Dict[str, float] = {
    "1m": 0.05,
    "5m": 0.10,
    "15m": 0.25,
    "30m": 0.15,
    "1h": 0.20,
    "4h": 0.15,
    "1d": 0.10,
}

ALL_TIMEFRAMES: List[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
FUSION_ENTRY_THRESHOLD: float = 0.25
MIN_AGREEING_TIMEFRAMES: int = 2
DEFAULT_POLL_INTERVAL_SEC: float = 15.0


class TradeStateMachine:
    """Thread-safe State Machine for managing the active global trade lifecycle."""

    STATES: List[str] = ["MONITORING", "ACTIVE_TRADE"]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state: str = "MONITORING"
        self.global_signal: float = 0.0  # 0.0=Flat, 1.0=Long, -1.0=Short
        self.entry_price: float = 0.0
        self.take_profit: float = 0.0
        self.stop_loss: float = 0.0
        self.locked_at: Optional[str] = None

    def enter_trade(self, signal_code: float, entry_price: float, tp: float, sl: float) -> bool:
        """Lock in a new active trade decision."""
        with self._lock:
            self.state = "ACTIVE_TRADE"
            self.global_signal = signal_code
            self.entry_price = entry_price
            self.take_profit = tp
            self.stop_loss = sl
            self.locked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return True

    def reset_to_flat(self) -> None:
        """Reset the state machine back to monitoring mode."""
        with self._lock:
            self.state = "MONITORING"
            self.global_signal = 0.0
            self.entry_price = 0.0
            self.take_profit = 0.0
            self.stop_loss = 0.0
            self.locked_at = None

    def get_snapshot(self) -> dict:
        """Get a thread-safe copy of the current state."""
        with self._lock:
            return {
                "state": self.state,
                "global_signal": self.global_signal,
                "entry_price": self.entry_price,
                "take_profit": self.take_profit,
                "stop_loss": self.stop_loss,
                "locked_at": self.locked_at,
            }


class UnifiedGlobalEngine:
    """
    Background engine that scans ALL timeframes concurrently, fuses AI predictions
    via weighted voting into a single locked trade decision, and monitors MT5.
    """

    def __init__(self, symbol: str = "XAUUSD") -> None:
        self.symbol_key: str = resolve_symbol(symbol)
        self.profile: dict = get_symbol_profile(self.symbol_key)
        self.state_machine: TradeStateMachine = TradeStateMachine()

        self._tf_lock = threading.Lock()
        self._tf_predictions: Dict[str, Optional[dict]] = {}
        self._tf_engines: Dict[str, Any] = {}

        EngineClass = _get_engine_class()
        logger.info(f"Initializing timeframe engines for {self.profile['display_name']}...")

        for tf in ALL_TIMEFRAMES:
            try:
                engine = EngineClass(
                    data_source="mt5",
                    symbol=self.symbol_key,
                    timeframe=tf,
                )
                self._tf_engines[tf] = engine
                logger.info(f"  [OK] Loaded {tf} prediction engine")
            except FileNotFoundError:
                logger.warning(f"  [--] {tf} engine skipped (no model file found)")
            except Exception as e:
                logger.error(f"  [ERR] {tf} engine initialization error: {e}")

        if not self._tf_engines:
            logger.critical("No timeframe engines could be loaded. Exiting.")
            sys.exit(1)

        self._write_global_state()
        logger.info(f"Ready. Monitoring {len(self._tf_engines)} timeframes: {list(self._tf_engines.keys())}")

    def _write_global_state(self) -> None:
        """Atomically persist global state to JSON for web dashboard consumption."""
        snap = self.state_machine.get_snapshot()
        sig_code = snap["global_signal"]

        if sig_code == 1.0:
            signal_label = "BUY / LONG"
        elif sig_code == -1.0:
            signal_label = "SELL / SHORT"
        else:
            signal_label = "NO TRADE"

        tf_summary: Dict[str, dict] = {}
        with self._tf_lock:
            for tf, pred in self._tf_predictions.items():
                if pred is not None:
                    tf_summary[tf] = {
                        "signal_code": pred.get("signal_code", 0.0),
                        "confidence": round(pred.get("confidence", 0.0), 1),
                        "prob_long": round(pred.get("prob_long", 0.0), 1),
                        "prob_short": round(pred.get("prob_short", 0.0), 1),
                    }

        state_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": self.profile["mt5_ticker"],
            "symbol_key": self.symbol_key,
            "display_name": self.profile["display_name"],
            "global_signal": signal_label,
            "global_signal_code": sig_code,
            "entry_price": snap["entry_price"],
            "take_profit": snap["take_profit"],
            "stop_loss": snap["stop_loss"],
            "locked_at": snap["locked_at"],
            "status": "ACTIVE TRADE" if sig_code != 0.0 else "MONITORING ALL TIMEFRAMES",
            "price_decimals": self.profile["price_decimals"],
            "price_format": self.profile["price_format"],
            "timeframe_signals": tf_summary,
            "active_timeframes": list(self._tf_engines.keys()),
        }

        # Atomic write for global_trade_state.json
        tmp_file = GLOBAL_STATE_FILE + ".tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
            os.replace(tmp_file, GLOBAL_STATE_FILE)
        except OSError as e:
            logger.error(f"Error writing global state file: {e}")

        # Mirror write for monitor_state.json
        try:
            legacy_data = {}
            if os.path.exists(LEGACY_STATE_FILE):
                with open(LEGACY_STATE_FILE, "r", encoding="utf-8") as f:
                    legacy_data = json.load(f)
            legacy_data.update(state_data)
            tmp2 = LEGACY_STATE_FILE + ".tmp"
            with open(tmp2, "w", encoding="utf-8") as f:
                json.dump(legacy_data, f, indent=2)
            os.replace(tmp2, LEGACY_STATE_FILE)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Legacy state write notice: {e}")

    def _scan_all_timeframes(self) -> None:
        """Concurrently fetch predictions across all active timeframe engines."""
        def scan_tf(tf: str, engine: Any) -> None:
            try:
                pred = engine.get_live_prediction(timeframe=tf, target_bar_idx=-2)
                with self._tf_lock:
                    self._tf_predictions[tf] = pred
            except Exception as e:
                logger.error(f"Error scanning timeframe {tf}: {e}")
                with self._tf_lock:
                    self._tf_predictions[tf] = None

        if not self._tf_engines:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._tf_engines)) as executor:
            futures = [executor.submit(scan_tf, tf, engine) for tf, engine in self._tf_engines.items()]
            concurrent.futures.wait(futures)

    def _fuse_signals(self) -> None:
        """
        Fuse multi-timeframe predictions into a single trade signal via weighted voting.
        """
        weighted_score = 0.0
        total_weight = 0.0
        agreeing_long = 0
        agreeing_short = 0
        htf_bias = 0.0

        with self._tf_lock:
            predictions = dict(self._tf_predictions)

        for tf, pred in predictions.items():
            if pred is None:
                continue

            signal = pred.get("signal_code", 0.0)
            confidence = pred.get("confidence", 0.0) / 100.0  # Normalize to 0-1
            weight = TIMEFRAME_WEIGHTS.get(tf, 0.1)

            weighted_score += signal * confidence * weight
            total_weight += weight

            if signal == 1.0:
                agreeing_long += 1
            elif signal == -1.0:
                agreeing_short += 1

            if tf in ("1h", "4h"):
                htf_bias = pred.get("htf_bias", signal)

        if total_weight <= 0:
            return

        fused_score = weighted_score / total_weight

        if fused_score > FUSION_ENTRY_THRESHOLD and agreeing_long >= MIN_AGREEING_TIMEFRAMES:
            direction = 1.0
        elif fused_score < -FUSION_ENTRY_THRESHOLD and agreeing_short >= MIN_AGREEING_TIMEFRAMES:
            direction = -1.0
        else:
            return

        # Select highest-confidence trigger prediction matching fused direction
        best_pred = None
        best_conf = -1.0
        for tf, pred in predictions.items():
            if pred and pred.get("signal_code", 0.0) == direction:
                if pred.get("confidence", 0.0) > best_conf:
                    best_conf = pred.get("confidence", 0.0)
                    best_pred = pred

        if best_pred:
            self._lock_trade(best_pred, htf_bias)

    def _lock_trade(self, trigger_pred: dict, htf_bias: float) -> None:
        """Calculate TP/SL targets and lock trade state."""
        sig_code = trigger_pred["signal_code"]
        entry_price = trigger_pred["close"]

        atr_val = self.profile.get("default_atr", 4.50)
        engine_15m = self._tf_engines.get("15m")
        if engine_15m and engine_15m.latest_live_df is not None and len(engine_15m.latest_live_df) >= 2:
            row = engine_15m.latest_live_df.iloc[-2]
            atr_val = float(row["high"] - row["low"])

        is_counter = (sig_code != htf_bias) and (htf_bias != 0.0)
        tp_mult = 1.0 if is_counter else 2.0
        sl_mult = 1.2 if is_counter else 1.5

        decimals = self.profile.get("price_decimals", 2)

        if sig_code == 1.0:
            tp = round(entry_price + tp_mult * atr_val, decimals)
            sl = round(entry_price - sl_mult * atr_val, decimals)
        else:
            tp = round(entry_price - tp_mult * atr_val, decimals)
            sl = round(entry_price + sl_mult * atr_val, decimals)

        self.state_machine.enter_trade(sig_code, entry_price, tp, sl)
        logger.info(f"TRADE LOCKED -> Signal: {sig_code} | Entry: {entry_price} | TP: {tp} | SL: {sl}")
        self._write_global_state()

    def _check_exit_conditions(self) -> None:
        """Check if MT5 open position has closed to reset state machine."""
        try:
            import MetaTrader5 as mt5
            with config.MT5_LOCK:
                if not mt5.initialize():
                    return
                positions = mt5.positions_get(symbol=self.profile["mt5_ticker"])

            if positions is None or len(positions) == 0:
                logger.info("No active MT5 position detected. Resetting to FLAT.")
                self.reset()
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error checking position status: {e}")

    def reset(self) -> None:
        """Public reset to clear active trade and return to monitoring mode."""
        self.state_machine.reset_to_flat()
        self._write_global_state()
        logger.info("Engine reset to FLAT monitoring mode.")

    def run(self, poll_interval: float = DEFAULT_POLL_INTERVAL_SEC) -> None:
        """Main engine execution loop."""
        logger.info(f"Starting Multi-Timeframe Engine Loop (Poll: {poll_interval}s)...")

        while True:
            try:
                self._scan_all_timeframes()

                snap = self.state_machine.get_snapshot()
                if snap["global_signal"] == 0.0:
                    self._fuse_signals()
                else:
                    self._check_exit_conditions()

                self._write_global_state()
                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                break
            except Exception as e:
                logger.error(f"Engine loop exception: {e}")
                time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Global Trade Engine")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol to monitor")
    args = parser.parse_args()

    engine = UnifiedGlobalEngine(symbol=args.symbol)
    engine.run()


if __name__ == "__main__":
    main()
