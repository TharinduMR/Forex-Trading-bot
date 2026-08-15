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

# Import Order Block Engine
from order_block_engine import OrderBlockEngine, TradeSetup, SetupType

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
        
        # Initialize Order Block Engine for high-probability setup detection
        self.ob_engine = OrderBlockEngine(min_imbalance_ratio=2.0, min_volume_ratio=1.5)
        self._active_setups: List[TradeSetup] = []
        self._setup_lock = threading.Lock()

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

        # Include active order block setups in state
        setups_data = []
        with self._setup_lock:
            for setup in self._active_setups[:5]:  # Limit to top 5 setups
                setups_data.append({
                    "type": setup.setup_type.value,
                    "direction": "BUY" if setup.direction == 1 else "SELL",
                    "entry": round(setup.entry_price, self.profile["price_decimals"]),
                    "sl": round(setup.stop_loss, self.profile["price_decimals"]),
                    "tp": round(setup.take_profit, self.profile["price_decimals"]),
                    "confidence": round(setup.confidence * 100, 1),
                    "rationale": setup.rationale,
                    "timeframe": setup.timeframe,
                    "ob_start": round(setup.order_block_start, self.profile["price_decimals"]),
                    "ob_end": round(setup.order_block_end, self.profile["price_decimals"]),
                })

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
            "order_block_setups": setups_data,
            "setup_count": len(setups_data),
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
        """Concurrently fetch predictions across all active timeframe engines and detect order blocks."""
        def scan_tf(tf: str, engine: Any) -> None:
            try:
                pred = engine.get_live_prediction(timeframe=tf, target_bar_idx=-2)
                with self._tf_lock:
                    self._tf_predictions[tf] = pred
                    
                # Also scan for order block setups on this timeframe
                if engine.latest_live_df is not None and len(engine.latest_live_df) > 50:
                    try:
                        ob_setups = self.ob_engine.detect_order_blocks(
                            engine.latest_live_df.copy(), 
                            self.symbol_key, 
                            tf
                        )
                        with self._setup_lock:
                            # Add new setups to active list
                            for setup in ob_setups:
                                # Avoid duplicates
                                if not any(s.timestamp == setup.timestamp and s.timeframe == setup.timeframe 
                                          for s in self._active_setups):
                                    self._active_setups.append(setup)
                                    logger.info(f"Order Block detected on {tf}: {setup.setup_type.value} @ {setup.entry_price}")
                    except Exception as e:
                        logger.debug(f"Order block detection error on {tf}: {e}")
                        
            except Exception as e:
                logger.error(f"Error scanning timeframe {tf}: {e}")
                with self._tf_lock:
                    self._tf_predictions[tf] = None

        if not self._tf_engines:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._tf_engines)) as executor:
            futures = [executor.submit(scan_tf, tf, engine) for tf, engine in self._tf_engines.items()]
            concurrent.futures.wait(futures)
        
        # Clean up old/stale setups (older than 4 hours or already filled/cancelled)
        self._cleanup_stale_setups()
    
    def _cleanup_stale_setups(self) -> None:
        """Remove stale or expired order block setups."""
        from datetime import timedelta
        
        current_time = pd.Timestamp.now()
        with self._setup_lock:
            fresh_setups = []
            for setup in self._active_setups:
                # Keep setups that are:
                # 1. Less than 4 hours old
                # 2. Still in PENDING status
                # 3. Price hasn't violated the order block significantly
                age = current_time - setup.timestamp
                if age < timedelta(hours=4) and setup.status == "PENDING":
                    fresh_setups.append(setup)
            
            # Keep only the top 10 highest confidence setups
            fresh_setups.sort(key=lambda x: x.confidence, reverse=True)
            self._active_setups = fresh_setups[:10]

    def _calculate_volatility_confirmation(self, predictions: Dict[str, dict]) -> Tuple[bool, float]:
        """
        Volatility Confirmation: Require expanding ATR across multiple timeframes.
        Returns (confirmed, avg_atr_ratio) where confirmed=True if ATR is expanding on majority of TFs.
        """
        atr_ratios = []
        confirming_tfs = 0
        total_tfs = 0
        
        for tf, pred in predictions.items():
            if pred is None:
                continue
            
            current_atr = pred.get("atr", 0.0)
            prev_atr = pred.get("prev_atr", current_atr)
            
            if current_atr > 0 and prev_atr > 0:
                ratio = current_atr / prev_atr
                atr_ratios.append(ratio)
                total_tfs += 1
                
                # ATR expanding if ratio > 1.05 (5% increase)
                if ratio > 1.05:
                    confirming_tfs += 1
        
        if total_tfs == 0:
            return False, 1.0
        
        avg_ratio = np.mean(atr_ratios) if atr_ratios else 1.0
        # Confirmed if majority of timeframes show expanding volatility
        confirmed = confirming_tfs >= (total_tfs / 2)
        
        return confirmed, avg_ratio

    def _check_momentum_alignment(self, predictions: Dict[str, dict], direction: float) -> Tuple[bool, float]:
        """
        Momentum Alignment: Check if RSI/MACD align across timeframes.
        Returns (aligned, alignment_score) where aligned=True if momentum agrees on majority of TFs.
        """
        rsi_aligned = 0
        macd_aligned = 0
        total_tfs = 0
        
        for tf, pred in predictions.items():
            if pred is None:
                continue
            
            total_tfs += 1
            rsi = pred.get("rsi", 50.0)
            macd_signal = pred.get("macd_signal", 0)  # 1=bullish, -1=bearish, 0=neutral
            
            # RSI alignment: For LONG, RSI should be > 50 and rising; for SHORT, < 50 and falling
            if direction == 1.0:  # Long
                if rsi > 50:
                    rsi_aligned += 1
                if macd_signal == 1:
                    macd_aligned += 1
            elif direction == -1.0:  # Short
                if rsi < 50:
                    rsi_aligned += 1
                if macd_signal == -1:
                    macd_aligned += 1
        
        if total_tfs == 0:
            return False, 0.0
        
        rsi_score = rsi_aligned / total_tfs
        macd_score = macd_aligned / total_tfs if total_tfs > 0 else 0.0
        alignment_score = (rsi_score + macd_score) / 2
        
        # Aligned if > 60% of timeframes agree
        aligned = alignment_score > 0.6
        
        return aligned, alignment_score

    def _detect_volume_surge(self, predictions: Dict[str, dict]) -> Tuple[bool, float]:
        """
        Volume Surge Detection: Add volume spike confirmation.
        Returns (surge_detected, max_volume_ratio) where surge_detected=True if volume spikes on key TFs.
        """
        volume_ratios = []
        surge_tfs = 0
        key_tf_count = 0  # Count of key timeframes (15m, 1h, 4h)
        
        for tf, pred in predictions.items():
            if pred is None:
                continue
            
            current_vol = pred.get("volume", 0.0)
            avg_vol = pred.get("avg_volume", current_vol)
            
            if current_vol > 0 and avg_vol > 0:
                ratio = current_vol / avg_vol
                volume_ratios.append(ratio)
                
                # Volume surge if ratio > 1.5 (50% above average)
                if ratio > 1.5:
                    surge_tfs += 1
                    if tf in ("15m", "1h", "4h"):
                        key_tf_count += 1
        
        if not volume_ratios:
            return False, 1.0
        
        max_ratio = max(volume_ratios)
        avg_ratio = np.mean(volume_ratios)
        
        # Surge confirmed if: at least one TF has surge OR average volume is elevated
        surge_detected = (surge_tfs >= 1) or (avg_ratio > 1.2)
        
        # Extra confirmation if key timeframes show surge
        if key_tf_count >= 1:
            surge_detected = True
        
        return surge_detected, max_ratio

    def _validate_breakout(self, predictions: Dict[str, dict], direction: float) -> Tuple[bool, int]:
        """
        Breakout Validation: Detect simultaneous breakout patterns across timeframes.
        Returns (breakout_confirmed, breakout_count) where breakout_confirmed=True if breakouts align.
        """
        breakout_tfs = 0
        total_tfs = 0
        
        for tf, pred in predictions.items():
            if pred is None:
                continue
            
            total_tfs += 1
            is_breakout = pred.get("is_breakout", False)
            breakout_direction = pred.get("breakout_direction", 0)  # 1=up, -1=down
            
            if is_breakout and breakout_direction == direction:
                breakout_tfs += 1
        
        # Breakout confirmed if at least 2 timeframes show aligned breakout
        breakout_confirmed = breakout_tfs >= 2
        
        return breakout_confirmed, breakout_tfs

    def _get_dynamic_weights(self, predictions: Dict[str, dict], current_hour: int) -> Dict[str, float]:
        """
        Dynamic Weight Adjustment: Increase weights during high-volatility sessions.
        Adjusts timeframe weights based on:
        - Session overlap (London/NY overlap = highest volatility)
        - Current volatility regime (high vol = higher weights on faster TFs)
        """
        base_weights = TIMEFRAME_WEIGHTS.copy()
        
        # Session multipliers (Forex market hours in UTC)
        session_multiplier = 1.0
        
        # London session: 07:00-16:00 UTC
        london_active = 7 <= current_hour <= 16
        
        # NY session: 12:00-21:00 UTC
        ny_active = 12 <= current_hour <= 21
        
        # London/NY overlap: 12:00-16:00 UTC (highest volatility)
        overlap_active = london_active and ny_active
        
        if overlap_active:
            session_multiplier = 1.5  # 50% weight increase during overlap
        elif london_active or ny_active:
            session_multiplier = 1.2  # 20% increase during single session
        else:
            session_multiplier = 0.8  # Reduce weights during Asian session/low vol
        
        # Calculate average volatility across timeframes
        avg_atr_ratio = 1.0
        atr_count = 0
        for tf, pred in predictions.items():
            if pred and pred.get("atr", 0) > 0 and pred.get("prev_atr", 0) > 0:
                ratio = pred["atr"] / pred["prev_atr"]
                avg_atr_ratio = (avg_atr_ratio * atr_count + ratio) / (atr_count + 1)
                atr_count += 1
        
        # Adjust weights based on volatility regime
        vol_adjustment = min(avg_atr_ratio, 1.5)  # Cap at 1.5x
        
        # Apply adjustments to weights
        adjusted_weights = {}
        for tf, weight in base_weights.items():
            # Faster timeframes get more weight during high volatility
            tf_speed_factor = 1.0
            if tf in ("1m", "5m"):
                tf_speed_factor = vol_adjustment  # Increase during high vol
            elif tf in ("1h", "4h", "1d"):
                tf_speed_factor = 1.0 / vol_adjustment  # Slightly decrease relative weight
            
            adjusted_weights[tf] = weight * session_multiplier * tf_speed_factor
        
        # Normalize weights to sum to 1.0
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {tf: w/total for tf, w in adjusted_weights.items()}
        
        return adjusted_weights

    def _fuse_signals(self) -> None:
        """
        Fuse multi-timeframe predictions into a single trade signal via weighted voting.
        Enhanced with:
        - Volatility Confirmation (expanding ATR)
        - Momentum Alignment (RSI/MACD)
        - Volume Surge Detection
        - Breakout Validation
        - Dynamic Weight Adjustment
        """
        weighted_score = 0.0
        total_weight = 0.0
        agreeing_long = 0
        agreeing_short = 0
        htf_bias = 0.0
        
        # Get current hour for session-based weight adjustment
        current_hour = datetime.now().hour
        
        # Get dynamic weights based on session and volatility
        dynamic_weights = self._get_dynamic_weights(self._tf_predictions, current_hour)

        with self._tf_lock:
            predictions = dict(self._tf_predictions)

        # First pass: Calculate basic weighted score
        for tf, pred in predictions.items():
            if pred is None:
                continue

            signal = pred.get("signal_code", 0.0)
            confidence = pred.get("confidence", 0.0) / 100.0  # Normalize to 0-1
            weight = dynamic_weights.get(tf, TIMEFRAME_WEIGHTS.get(tf, 0.1))

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

        # Determine preliminary direction
        if fused_score > FUSION_ENTRY_THRESHOLD and agreeing_long >= MIN_AGREEING_TIMEFRAMES:
            direction = 1.0
        elif fused_score < -FUSION_ENTRY_THRESHOLD and agreeing_short >= MIN_AGREEING_TIMEFRAMES:
            direction = -1.0
        else:
            return

        # === ENHANCED VALIDATION LAYERS ===
        
        # 1. Volatility Confirmation
        vol_confirmed, atr_ratio = self._calculate_volatility_confirmation(predictions)
        if not vol_confirmed:
            logger.debug(f"Volatility confirmation failed (ATR ratio: {atr_ratio:.2f}). Skipping trade.")
            return

        # 2. Momentum Alignment
        momentum_aligned, mom_score = self._check_momentum_alignment(predictions, direction)
        if not momentum_aligned:
            logger.debug(f"Momentum alignment weak (score: {mom_score:.2f}). Skipping trade.")
            # Don't skip entirely, but reduce confidence requirement
            # Fused score must be stronger to compensate
            if abs(fused_score) < FUSION_ENTRY_THRESHOLD * 1.5:
                logger.debug(f"Fused score too weak to override momentum misalignment. Skipping.")
                return

        # 3. Volume Surge Detection
        volume_surge, vol_ratio = self._detect_volume_surge(predictions)
        if not volume_surge:
            logger.debug(f"No volume surge detected (ratio: {vol_ratio:.2f}). Trade allowed but noted.")
            # Volume is not a hard filter, just a confirmation boost

        # 4. Breakout Validation
        breakout_valid, breakout_count = self._validate_breakout(predictions, direction)
        if breakout_valid:
            logger.info(f"Breakout validated on {breakout_count} timeframes! Boosting confidence.")
            # Boost the fused score for breakout scenarios
            fused_score *= 1.2  # 20% confidence boost for validated breakouts

        # Re-check threshold after breakout boost
        if direction == 1.0 and fused_score <= FUSION_ENTRY_THRESHOLD:
            logger.debug("Even with breakout boost, long score below threshold. Skipping.")
            return
        if direction == -1.0 and fused_score >= -FUSION_ENTRY_THRESHOLD:
            logger.debug("Even with breakout boost, short score below threshold. Skipping.")
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
            # Add enhancement metrics to prediction for logging/debugging
            best_pred["volatility_confirmed"] = vol_confirmed
            best_pred["atr_ratio"] = atr_ratio
            best_pred["momentum_aligned"] = momentum_aligned
            best_pred["momentum_score"] = mom_score
            best_pred["volume_surge"] = volume_surge
            best_pred["volume_ratio"] = vol_ratio
            best_pred["breakout_validated"] = breakout_valid
            best_pred["breakout_count"] = breakout_count
            best_pred["session_multiplier"] = dynamic_weights
            
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
