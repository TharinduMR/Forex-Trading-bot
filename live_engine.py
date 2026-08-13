"""
Real-Time Inference and Simulated/Live Execution Engine for Multi-Symbol Trading.
Supports Gold (XAUUSD / GC=F) and Forex pairs (EURUSD).
Continuously buffers recent market data, calculates lookahead-safe features, loads the trained
XGBoost/LightGBM ensemble, generates directional predictions, and manages execution with safety checks.
"""
import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import config
from config import get_symbol_profile, get_model_paths, get_ticker_for_source, resolve_symbol
from data_loader import get_data_loader
from features import engineer_all_features
from trade_manager import ExpertAdvisorManager
from risk_manager import RiskManager
import threading
import queue

_monitor_queue = queue.Queue()

def _monitor_writer_worker():
    while True:
        try:
            task = _monitor_queue.get()
            if task is None:
                break
            state, state_file, legacy_file = task
            temp_file = state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(temp_file, state_file)
            
            legacy_temp = legacy_file + ".tmp"
            with open(legacy_temp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(legacy_temp, legacy_file)
        except Exception:
            pass
        finally:
            _monitor_queue.task_done()

_writer_thread = threading.Thread(target=_monitor_writer_worker, daemon=True)
_writer_thread.start()


def has_open_position_for_symbol(positions, symbol):
    """Return True if any position already targets the provided symbol."""
    if not positions:
        return False
    symbol_upper = str(symbol).upper()
    for pos in positions:
        pos_symbol = getattr(pos, "symbol", None)
        if pos_symbol is None:
            continue
        if str(pos_symbol).upper() == symbol_upper:
            return True
    return False


def infer_signal_from_probabilities(probabilities, confidence_threshold=0.35, edge_margin=0.02, min_directional_prob=0.32):
    """Convert model probabilities into a directional signal without over-biasing toward flat."""
    prob_short, prob_flat, prob_long = probabilities
    if prob_long >= prob_short:
        directional_prob = prob_long
        directional_code = 1.0
        other_dir_prob = prob_short
    else:
        directional_prob = prob_short
        directional_code = -1.0
        other_dir_prob = prob_long

    if directional_prob >= max(confidence_threshold * 0.9, min_directional_prob) and directional_prob >= prob_flat + edge_margin:
        return directional_code, directional_prob

    if prob_flat >= max(confidence_threshold, min_directional_prob) and prob_flat >= directional_prob + edge_margin:
        return 0.0, prob_flat

    if directional_prob >= max(confidence_threshold * 0.7, min_directional_prob * 0.95) and directional_prob > prob_flat:
        return directional_code, directional_prob

    return 0.0, prob_flat


class LivePredictionEngine:
    def __init__(self, data_source=config.DEFAULT_DATA_SOURCE, symbol="XAUUSD", timeframe="15m"):
        self.data_source = data_source
        self.loader = get_data_loader(data_source)
        self.symbol_key = resolve_symbol(symbol)
        self.timeframe = timeframe
        self.profile = get_symbol_profile(self.symbol_key)
        self.model_paths = get_model_paths(self.symbol_key, timeframe=self.timeframe)
        self.xgb_model = None
        self.lgb_model = None
        self.meta_model = None
        self.smc_model = None
        self.smc_feature_names = None
        self.ict_model = None
        self.ict_feature_names = None
        self.feature_names = None
        self.latest_live_df = None
        self.ea_manager = None
        self.last_candle_time = None
        self.last_pred_data = None
        self.current_signal_state = 0.0    # Signal hysteresis state: 0.0=Flat, 1.0=Long, -1.0=Short
        self.last_entry_trigger_bar = None  # Tracks the bar timestamp of the last entry trigger
        self.risk_manager = RiskManager(
            max_daily_loss=config.MAX_DAILY_LOSS if hasattr(config, 'MAX_DAILY_LOSS') else 200.0,
            max_consecutive_losses=config.MAX_CONSECUTIVE_LOSSES if hasattr(config, 'MAX_CONSECUTIVE_LOSSES') else 3,
            max_drawdown_pct=config.MAX_DRAWDOWN_PCT if hasattr(config, 'MAX_DRAWDOWN_PCT') else 0.10,
            state_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'risk_state.json')
        )
        self._load_models()
        
    def _load_models(self):
        """Load trained XGBoost, LightGBM models, Secondary Decision Maker, and feature schema."""
        xgb_path = self.model_paths["xgb"]
        lgb_path = self.model_paths["lgb"]
        feat_path = self.model_paths["feature_names"]
        meta_path = self.model_paths["meta"]
        
        if not (os.path.exists(xgb_path) and os.path.exists(lgb_path) and os.path.exists(feat_path)):
            raise FileNotFoundError(
                f"Trained model files not found for {self.profile['display_name']}! "
                f"Please run 'python main.py --mode train --symbol {self.symbol_key}' first.\n"
                f"  Expected: {xgb_path}"
            )
            
        self.xgb_model = joblib.load(xgb_path)
        self.lgb_model = joblib.load(lgb_path)
        self.feature_names = joblib.load(feat_path)
        if os.path.exists(meta_path):
            self.meta_model = joblib.load(meta_path)
            print(f"[LiveEngine/{self.symbol_key}] Loaded Secondary AI Decision Maker from {meta_path}")
        else:
            print(f"[LiveEngine/{self.symbol_key}] Note: Secondary AI Decision Maker not found at {meta_path}.")
        
        # Load SMC Specialist Regressor (optional — if not found, classic flow is used)
        smc_reg_path = self.model_paths.get("smc_reg", "")
        smc_feat_path = self.model_paths.get("smc_feature_names", "")
        if smc_reg_path and os.path.exists(smc_reg_path) and smc_feat_path and os.path.exists(smc_feat_path):
            self.smc_model = joblib.load(smc_reg_path)
            self.smc_feature_names = joblib.load(smc_feat_path)
            print(f"[LiveEngine/{self.symbol_key}] Loaded SMC Specialist Regressor ({len(self.smc_feature_names)} features)")
        else:
            print(f"[LiveEngine/{self.symbol_key}] SMC Specialist Model not found. Using classic ensemble-only mode.")
            
        # Load ICT Specialist Regressor (optional)
        ict_reg_path = self.model_paths.get("ict_reg", "")
        ict_feat_path = self.model_paths.get("ict_feature_names", "")
        if ict_reg_path and os.path.exists(ict_reg_path) and ict_feat_path and os.path.exists(ict_feat_path):
            self.ict_model = joblib.load(ict_reg_path)
            self.ict_feature_names = joblib.load(ict_feat_path)
            print(f"[LiveEngine/{self.symbol_key}] Loaded ICT Specialist Regressor ({len(self.ict_feature_names)} features)")
        else:
            print(f"[LiveEngine/{self.symbol_key}] ICT Specialist Model not found.")
        
        print(f"[LiveEngine/{self.symbol_key}] Successfully loaded ensemble models with {len(self.feature_names)} features.")
        
        self.ea_manager = ExpertAdvisorManager(
            self.symbol_key, 
            self.xgb_model, 
            self.meta_model, 
            self.feature_names,
            live_engine=self
        )
        
    def _get_monitor_state_file(self):
        """Return per-symbol monitor state file path."""
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"monitor_state_{self.symbol_key}_{self.timeframe}.json"
        )
        
    def _write_monitor_state(self, pred_data, action_str, tp_price=0.0, sl_price=0.0, spread_val=0.0, llm_reasoning="NVIDIA LLM Standby"):
        """Write current engine state to JSON for non-blocking browser monitoring."""
        try:
            state = {
                "timestamp": str(pred_data['timestamp']) if pred_data else str(datetime.now()),
                "symbol": pred_data['symbol'] if pred_data else self.profile["mt5_ticker"],
                "symbol_key": self.symbol_key,
                "display_name": self.profile["display_name"],
                "timeframe": pred_data['timeframe'] if pred_data else config.TIMEFRAME_LTF,
                "close": float(pred_data['close']) if pred_data else 0.0,
                "signal": pred_data['signal'] if pred_data else "NO TRADE (0)",
                "signal_code": float(pred_data['signal_code']) if pred_data else 0.0,
                "confidence": float(pred_data['confidence']) if pred_data else 0.0,
                "prob_short": float(pred_data['prob_short']) if pred_data else 0.0,
                "prob_flat": float(pred_data['prob_flat']) if pred_data else 100.0,
                "prob_long": float(pred_data['prob_long']) if pred_data else 0.0,
                "meta_prob": float(pred_data.get('meta_prob', 1.0)) if pred_data else 1.0,
                "htf_bias": float(pred_data.get('htf_bias', 0.0)) if pred_data else 0.0,
                "atr": float(pred_data['atr']) if pred_data else 0.0,
                "tp_price": float(tp_price),
                "sl_price": float(sl_price),
                "spread": float(spread_val),
                "action": action_str,
                "conviction_state": self.current_signal_state,
                "conviction_held_since": str(self.last_entry_trigger_bar) if self.last_entry_trigger_bar else None,
                "llm_reasoning": str(llm_reasoning),
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "live_trading_enabled": config.ENABLE_LIVE_TRADING,
                "data_source": self.data_source,
                "price_decimals": self.profile["price_decimals"],
                "price_format": self.profile["price_format"],
            }
            state_file = self._get_monitor_state_file()
            legacy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_state.json")
            _monitor_queue.put((state, state_file, legacy_file))
        except Exception:
            pass  # Suppress errors to never block trading loop
        
    def get_live_prediction(self, symbol=None, timeframe=config.TIMEFRAME_LTF, target_bar_idx=-2):
        """
        Fetch live data buffer, engineer features, and predict direction for the target candle.
        target_bar_idx = -2 evaluates on the last CLOSED candle to prevent flickering.
        Implements Signal Hysteresis to maintain state confidence.
        """
        if symbol is None:
            symbol = get_ticker_for_source(self.symbol_key, self.data_source)
        
        # 1. Fetch Primary LTF buffer
        df_ltf = self.loader.fetch_latest_buffer(symbol, timeframe, buffer_size=min(config.BUFFER_SIZE, 300))
        if df_ltf is None or len(df_ltf) < 50:
            print(f"[Warning/{self.symbol_key}] Insufficient buffer data ({len(df_ltf) if df_ltf is not None else 0} bars).")
            return None
            
        # 2. Fetch HTF buffers for confluence dynamically
        df_htf1 = None
        df_htf2 = None
        htf1_tf, htf2_tf = config.get_htf_timeframes(timeframe)
        try:
            df_htf1 = self.loader.fetch_latest_buffer(symbol, htf1_tf, buffer_size=100)
            df_htf2 = self.loader.fetch_latest_buffer(symbol, htf2_tf, buffer_size=50)
        except Exception as e:
            # If HTF fetch fails, continue without HTF or use empty
            pass
            
        # 3. Apply Feature Engineering
        df_proc = engineer_all_features(df_ltf, df_htf1, df_htf2)
        
        # 4. Verify feature schema alignment
        for col in self.feature_names:
            if col not in df_proc.columns:
                df_proc[col] = 0.0
                
        self.latest_live_df = df_proc
                
        # Extract features for the target bar
        target_row = df_proc.iloc[target_bar_idx]
        X_live = target_row[self.feature_names].values.reshape(1, -1)
        
        # 5. Ensemble Inference on Full Buffer for EMA Smoothing
        X_buffer = df_proc[self.feature_names].values
        probs_xgb = self.xgb_model.predict_proba(X_buffer)
        probs_lgb = self.lgb_model.predict_proba(X_buffer)
        probs_avg = (probs_xgb + probs_lgb) / 2.0
        
        # Apply EMA Smoothing to probabilities
        df_probs = pd.DataFrame(probs_avg, columns=['prob_short', 'prob_flat', 'prob_long'])
        df_probs_smoothed = df_probs.ewm(span=config.SIGNAL_SMOOTHING_WINDOW).mean()
        
        # Get smoothed probabilities for target bar
        target_probs = df_probs_smoothed.iloc[target_bar_idx].values
        
        # --- SIGNAL HYSTERESIS LOGIC (STATE MACHINE) ---
        # Requires strong evidence to ENTER a new signal, weak evidence to HOLD,
        # and sustained weakness to EXIT. Prevents BUY->FLAT->BUY flickering.
        entry_thresh = getattr(config, 'CONVICTION_ENTRY_THRESHOLD', 0.55)
        hold_thresh = getattr(config, 'CONVICTION_HOLD_THRESHOLD', 0.35)
        edge_margin = getattr(config, 'CONVICTION_ENTRY_EDGE', 0.10)
        
        prev_state = self.current_signal_state
        
        if prev_state == 1.0:  # Currently LONG — hold unless conviction lost
            if target_probs[2] >= hold_thresh:
                signal_code = 1.0
                signal_confidence = target_probs[2]
            else:
                signal_code = 0.0
                signal_confidence = target_probs[1]
                self.current_signal_state = 0.0  # Commit exit immediately
                print(f"  [Hysteresis] LONG conviction lost (Long prob {target_probs[2]*100:.1f}% < {hold_thresh*100:.1f}%). Exiting to FLAT.")
                
        elif prev_state == -1.0:  # Currently SHORT — hold unless conviction lost
            if target_probs[0] >= hold_thresh:
                signal_code = -1.0
                signal_confidence = target_probs[0]
            else:
                signal_code = 0.0
                signal_confidence = target_probs[1]
                self.current_signal_state = 0.0  # Commit exit immediately
                print(f"  [Hysteresis] SHORT conviction lost (Short prob {target_probs[0]*100:.1f}% < {hold_thresh*100:.1f}%). Exiting to FLAT.")
                
        else:  # Currently FLAT — look for strong entry signal
            if target_probs[2] > entry_thresh and target_probs[2] > target_probs[0] + edge_margin:
                signal_code = 1.0
                signal_confidence = target_probs[2]
                # State NOT committed yet — execute_trade gates must approve first
            elif target_probs[0] > entry_thresh and target_probs[0] > target_probs[2] + edge_margin:
                signal_code = -1.0
                signal_confidence = target_probs[0]
                # State NOT committed yet — execute_trade gates must approve first
            else:
                signal_code = 0.0
                signal_confidence = target_probs[1]
        
        confidence = signal_confidence * 100.0
        signal_str = "BUY / LONG (+1)" if signal_code == 1.0 else ("SELL / SHORT (-1)" if signal_code == -1.0 else "NO TRADE (0)")
        
        # 6. Secondary AI Decision Maker evaluation
        meta_prob = 1.0
        if self.meta_model is not None and signal_code in [1.0, -1.0]:
            meta_prob = self.meta_model.predict_proba(X_live)[0, 1]
            
        # STATEFUL OVERRIDE: Check for active MT5 position
        try:
            active_override = False
            if config.ENABLE_LIVE_TRADING and self.data_source == "mt5":
                import MetaTrader5 as mt5
                with config.MT5_LOCK:
                    if mt5.initialize():
                        positions = mt5.positions_get(symbol=self.profile["mt5_ticker"])
                        if positions and len(positions) > 0:
                            pos = positions[0]
                            if pos.type == mt5.ORDER_TYPE_BUY:
                                signal_code = 1.0
                                signal_str = "ACTIVE LONG (In Trade)"
                                confidence = 100.0
                                meta_prob = 1.0
                                active_override = True
                            elif pos.type == mt5.ORDER_TYPE_SELL:
                                signal_code = -1.0
                                signal_str = "ACTIVE SHORT (In Trade)"
                                confidence = 100.0
                                meta_prob = 1.0
                                active_override = True
        except Exception:
            pass

            
        # Compute ATR for risk scaling
        atr_val = target_row['high'] - target_row['low']  # fallback proxy if atr column not directly saved
        if 'volatility' in target_row:
            atr_val = target_row['close'] * target_row['volatility']
            
        result = {
            'timestamp': df_proc.index[target_bar_idx],
            'symbol': symbol,
            'symbol_key': self.symbol_key,
            'timeframe': timeframe,
            'close': target_row['close'],
            'signal': signal_str,
            'signal_code': signal_code,
            'confidence': confidence,
            'prob_short': target_probs[0] * 100.0,
            'prob_flat': target_probs[1] * 100.0,
            'prob_long': target_probs[2] * 100.0,
            'meta_prob': meta_prob,
            'htf_bias': target_row.get('htf_directional_bias', 0.0),
            'atr': atr_val
        }
        return result
    
    def evaluate_smc_setup(self, target_row):
        """
        SMC Gate: Evaluates whether the latest bar has a valid SMC confluence event
        and, if so, asks the SMC Specialist Regressor to predict the expected return.
        
        Returns a dict with:
            'approved': bool - whether the SMC gate approves the trade
            'direction': float - +1.0 long, -1.0 short, 0.0 none
            'expected_return': float - predicted vol-adjusted forward return
            'reason': str - human-readable reason for the decision
        """
        smc_trigger = target_row.get('smc_trigger', 0.0)
        threshold = getattr(config, 'SMC_ENTRY_THRESHOLD', 0.5)
        
        # No SMC confluence — stay flat
        if smc_trigger == 0.0:
            return {
                'approved': False,
                'direction': 0.0,
                'expected_return': 0.0,
                'reason': 'NO TRADE: Waiting for SMC Confluence (Sweep + CHoCH + FVG)'
            }
        
        direction = "LONG" if smc_trigger == 1.0 else "SHORT"
        
        # Ask the SMC Regressor to evaluate the micro-structure
        X_live = target_row[self.smc_feature_names].values.reshape(1, -1)
        expected_return = float(self.smc_model.predict(X_live)[0])
        
        # Decision gate: expected return must exceed threshold
        if expected_return > threshold:
            return {
                'approved': True,
                'direction': smc_trigger,
                'expected_return': expected_return,
                'reason': f'EXECUTE {direction} | AI Expected Return: {expected_return:+.2f} (threshold: {threshold})'
            }
        else:
            return {
                'approved': False,
                'direction': smc_trigger,
                'expected_return': expected_return,
                'reason': f'REJECT {direction} SETUP | AI Expected Return too low ({expected_return:+.2f} < {threshold})'
            }
        
    def evaluate_ict_setup(self, target_row):
        """
        ICT Gate: Evaluates whether the latest bar has a valid ICT setup
        and, if so, asks the ICT Specialist Regressor to predict the distribution return.
        
        Returns a dict with:
            'approved': bool - whether the ICT gate approves the trade
            'direction': float - +1.0 long, -1.0 short, 0.0 none
            'expected_return': float - predicted vol-adjusted forward return
            'reason': str - human-readable reason
        """
        ict_trigger = target_row.get('ict_trigger', 0.0)
        threshold = getattr(config, 'ICT_ENTRY_THRESHOLD', 0.75)
        
        # No ICT confluence
        if ict_trigger == 0.0:
            return {
                'approved': False,
                'direction': 0.0,
                'expected_return': 0.0,
                'reason': 'NO TRADE: Waiting for ICT Confluence (Killzone + Judas + FVG + OTE)'
            }
            
        direction = "LONG" if ict_trigger == 1.0 else "SHORT"
        
        # Ask the ICT Regressor to evaluate setup
        X_live = target_row[self.ict_feature_names].values.reshape(1, -1)
        expected_return = float(self.ict_model.predict(X_live)[0])
        
        if expected_return > threshold:
            return {
                'approved': True,
                'direction': ict_trigger,
                'expected_return': expected_return,
                'reason': f'EXECUTE {direction} (ICT Setup) | AI Expected Distribution: {expected_return:+.2f} (threshold: {threshold})'
            }
        else:
            return {
                'approved': False,
                'direction': ict_trigger,
                'expected_return': expected_return,
                'reason': f'REJECT {direction} ICT SETUP | Weak Expected Distribution ({expected_return:+.2f} < {threshold})'
            }
        
        
        
    def manage_open_trade_mt5(self, pred_data):
        """
        Smart Money Risk Manager.
        Checks if 1:1 RR is hit. If so, moves SL to Breakeven and takes 50% partials.
        """
        try:
            import MetaTrader5 as mt5
            with config.MT5_LOCK:
                if not mt5.initialize():
                    return
                
                mt5_symbol = self.profile["mt5_ticker"]
                positions = mt5.positions_get(symbol=mt5_symbol)
            if not positions or len(positions) == 0:
                return
                
            pos = positions[0]
            entry_price = pos.price_open
            current_price = pos.price_current
            original_sl = pos.sl
            volume = pos.volume
            
            # Check if SL exists
            if original_sl == 0.0:
                return
                
            # Has breakeven already been applied? (if SL is at or past entry)
            is_long = pos.type == mt5.ORDER_TYPE_BUY
            if is_long and original_sl >= entry_price:
                return
            if not is_long and original_sl <= entry_price and original_sl > 0:
                return
                
            risk = abs(entry_price - original_sl)
            if risk <= 0: return
            
            current_rr = (current_price - entry_price) / risk if is_long else (entry_price - current_price) / risk
            
            if current_rr >= 1.0:
                print(f"  [Risk Mgmt] 1:1 RR hit (Current RR: {current_rr:.2f})! Moving SL to Breakeven & taking 50% partials.")
                
                # 1. Take 50% Partial
                partial_vol = round(volume / 2.0, 2)
                min_vol = mt5.symbol_info(mt5_symbol).volume_min
                if partial_vol >= min_vol:
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": mt5_symbol,
                        "volume": partial_vol,
                        "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
                        "position": pos.ticket,
                        "price": mt5.symbol_info(mt5_symbol).bid if is_long else mt5.symbol_info(mt5_symbol).ask,
                        "deviation": 20,
                        "magic": 234000,
                        "comment": "50% Partial at 1:1 RR",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    result = mt5.order_send(request)
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"  [Risk Mgmt Error] Partial failed: {result.comment}")
                
                # 2. Move SL to Breakeven
                be_sl = entry_price
                request_sl = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": mt5_symbol,
                    "sl": be_sl,
                    "tp": pos.tp,
                    "position": pos.ticket,
                }
                res_sl = mt5.order_send(request_sl)
                if res_sl.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"  [Risk Mgmt Error] SL to BE failed: {res_sl.comment}")
                else:
                    print(f"  [Risk Mgmt] Stop Loss successfully moved to {be_sl}")
                    
        except Exception as e:
            print(f"  [Risk Mgmt Error] {e}")

    def execute_trade(self, pred_data):
        """
        Execute trade signal or log simulated execution based on safety configuration.
        Uses per-symbol profile for spread limits, contract sizes, and pip buffers.
        """
        if pred_data is None:
            return
            
        timestamp = pred_data['timestamp']
        symbol = pred_data['symbol']
        close = pred_data['close']
        signal_code = pred_data['signal_code']
        confidence = pred_data['confidence']
        meta_prob = pred_data.get('meta_prob', 1.0)
        atr = pred_data['atr']
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [{self.symbol_key}] Candle Processed: {timestamp}")
        print(f"  Symbol: {self.profile['display_name']} | Close: {close:,.{self.profile['price_decimals']}f}")
        print(f"  Primary Signal: {pred_data['signal']} | Primary Confidence: {confidence:.2f}%")
        print(f"  Secondary AI Decision Maker Confidence: {meta_prob*100:.1f}% (Min Required: {config.META_CONFIDENCE_THRESHOLD*100:.1f}%)")
        print(f"  Probabilities -> Long: {pred_data['prob_long']:.1f}% | Flat: {pred_data['prob_flat']:.1f}% | Short: {pred_data['prob_short']:.1f}%")
        
        if "ACTIVE" in pred_data['signal']:
            action_str = f"Managing Active Trade (Target TP/SL via Quant Adjuster)"
            print(f"  [Action] {action_str}")
            print("-" * 60)
            self._write_monitor_state(pred_data, action_str)
            if config.ENABLE_LIVE_TRADING and self.data_source == "mt5":
                self.manage_open_trade_mt5(pred_data)
            return
            
        active_override = False

        if signal_code == 0.0 or confidence < max(config.CONFIDENCE_THRESHOLD * 100.0, 15.0):
            action_str = f"Holding flat (Primary confidence < {max(config.CONFIDENCE_THRESHOLD*100.0, 15.0):.1f}%)"
            print(f"  [Action] {action_str}")
            print("-" * 60)
            self._write_monitor_state(pred_data, action_str)
            return
        
        # --- ICT/SMC PRE-GATES: Only apply entry gates when transitioning from FLAT to directional ---
        # If we are already in a held signal (hysteresis), skip the gates to prevent them from
        # killing active signals when the event trigger naturally disappears on the next bar.
        is_new_entry = (self.current_signal_state == 0.0 and signal_code != 0.0)
        
        if is_new_entry:
            # --- ICT PRE-GATE ---
            ict_triggered_trade = False
            if self.ict_model is not None and self.latest_live_df is not None:
                target_row = self.latest_live_df.iloc[-2] if len(self.latest_live_df) >= 2 else self.latest_live_df.iloc[-1]
                ict_result = self.evaluate_ict_setup(target_row)
                if ict_result['direction'] != 0.0:  # An ICT event actually occurred
                    print(f"  [ICT Gate] {ict_result['reason']}")
                    if not ict_result['approved']:
                        action_str = f"ICT GATE: {ict_result['reason']}"
                        print(f"  [Action] {action_str}")
                        print("-" * 60)
                        self._write_monitor_state(pred_data, action_str)
                        return
                    else:
                        # Approved by ICT Specialist! Override trade signals.
                        signal_code = ict_result['direction']
                        pred_data['signal_code'] = signal_code
                        pred_data['signal'] = "BUY / LONG (+1) [ICT]" if signal_code == 1.0 else "SELL / SHORT (-1) [ICT]"
                        pred_data['ict_expected_return'] = ict_result['expected_return']
                        ict_triggered_trade = True
            
            # --- SMC PRE-GATE ---
            if not ict_triggered_trade and self.smc_model is not None and self.latest_live_df is not None:
                target_row = self.latest_live_df.iloc[-2] if len(self.latest_live_df) >= 2 else self.latest_live_df.iloc[-1]
                smc_result = self.evaluate_smc_setup(target_row)
                print(f"  [SMC Gate] {smc_result['reason']}")
                if not smc_result['approved']:
                    action_str = f"SMC GATE: {smc_result['reason']}"
                    print(f"  [Action] {action_str}")
                    print("-" * 60)
                    self._write_monitor_state(pred_data, action_str)
                    return
                # If SMC approved, override signal direction with SMC direction
                if smc_result['direction'] != 0.0:
                    signal_code = smc_result['direction']
                    pred_data['signal_code'] = signal_code
                    pred_data['signal'] = "BUY / LONG (+1)" if signal_code == 1.0 else "SELL / SHORT (-1)"
                    pred_data['smc_expected_return'] = smc_result['expected_return']
        
        # All gates passed — commit the signal state for hysteresis persistence
        self.current_signal_state = signal_code
        if is_new_entry and signal_code != 0.0:
            self.last_entry_trigger_bar = pred_data.get('timestamp')
            print(f"  [Hysteresis] Signal state committed: {'LONG' if signal_code == 1.0 else 'SHORT'} (Entry locked)")

        reason = self.risk_manager.get_trade_block_reason(current_bar=None, equity=pred_data.get('equity'))
        if reason is not None:
            action_str = f"REJECTED BY RISK MANAGER ({reason})"
            print(f"  [Action] {action_str}")
            print("-" * 60)
            self._write_monitor_state(pred_data, action_str)
            return
            
        # Skip rejection gates if we are already in an active trade (override)
        if active_override:
            # Bypass meta-model confidence and top‑down bias checks
            pass
        else:
            # META confidence gate
            if meta_prob < config.META_CONFIDENCE_THRESHOLD:
                action_str = f"REJECTED BY AI DECISION MAKER (Meta Confidence {meta_prob*100:.1f}% < {config.META_CONFIDENCE_THRESHOLD*100:.1f}%)"
                print(f"  [Action] {action_str}")
                print("-" * 60)
                self._write_monitor_state(pred_data, action_str)
                return
            # Top‑down bias gates disabled to avoid suppressing otherwise valid directional signals.
            htf_bias = pred_data.get('htf_bias', 0.0)
            
        # Calculate Target and Stop Loss in Dollar / Points
        sl_dist = config.SL_MULT * atr
        tp_dist = config.TP_MULT * atr
        
        if signal_code == 1.0:
            tp_price = close + tp_dist
            sl_price = close - sl_dist
        else:
            tp_price = close - tp_dist
            sl_price = close + sl_dist
            
        decimals = self.profile["price_decimals"]
        print(f"  [Risk Rules] Target TP: {tp_price:,.{decimals}f} | Target SL: {sl_price:,.{decimals}f}")
        
        # --- NVIDIA LLM LIVE CONSENSUS VALIDATION ---
        llm_reasoning_log = "Local Meta-Model Approved"
        if getattr(config, "LLM_VALIDATION_ENABLED", False):
            print(f"  [NVIDIA LLM] Sending quantitative feature matrix to NVIDIA API for live consensus validation...")
            try:
                from llm_reasoner import validate_trade_signal
                feat_subset = {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in pred_data.items() if k in [
                    'rsi', 'volatility', 'dist_to_equilibrium', 'fib_ote_zone', 'macd_hist', 'macd_momentum',
                    'flod_cluster', 'bpr_cluster', 'kz_ny_am', 'dist_to_val', 'in_value_area', 'fear_greed_index',
                    'htf1_trend_direction', 'htf2_trend_direction', 'htf2_fib_ote_zone', 'htf2_macd_hist'
                ]}
                llm_val = validate_trade_signal(signal_code, confidence/100.0, meta_prob, close, sl_price, tp_price, feat_subset, symbol=self.symbol_key)
                llm_reasoning_log = f"[NVIDIA LLM] {llm_val.get('reason', 'Validated')}"
                print(f"  [NVIDIA LLM Result] Approved: {llm_val.get('approved', True)} | {llm_reasoning_log}")
                if not llm_val.get("approved", True):
                    action_str = f"REJECTED BY NVIDIA LLM CONSENSUS ({llm_reasoning_log})"
                    print(f"  [Action] {action_str}")
                    print("-" * 60)
                    self._write_monitor_state(pred_data, action_str, tp_price, sl_price, llm_reasoning=llm_reasoning_log)
                    return
                if llm_val.get("optimized_sl", 0) > 0:
                    sl_price = float(llm_val.get("optimized_sl"))
                if llm_val.get("optimized_tp", 0) > 0:
                    tp_price = float(llm_val.get("optimized_tp"))
                print(f"  [NVIDIA Consensus Risk] Optimized Target TP: {tp_price:,.{decimals}f} | Optimized Target SL: {sl_price:,.{decimals}f}")
            except Exception as e:
                print(f"  [NVIDIA LLM Warning] Validation exception: {e}. Defaulting to Local Meta-Model decision.")
        
        # Safety Gate: Advisory / Dry-Run Mode vs Live Execution
        if not config.ENABLE_LIVE_TRADING or self.data_source != "mt5":
            action_str = "Advisory Mode: Trade Signal Logged (Live Switch False)"
            print(f"  [Advisory Mode] Trade Signal Logged. (Live execution switch ENABLE_LIVE_TRADING is False or source is Yahoo Finance).")
            print("-" * 60)
            self._write_monitor_state(pred_data, action_str, tp_price, sl_price, llm_reasoning=llm_reasoning_log)
            return
            
        # --- LIVE MT5 EXECUTION LOGIC (Gated) ---
        mt5_symbol = self.profile["mt5_ticker"]
        max_spread = self.profile["max_spread_points"]
        
        try:
            import MetaTrader5 as mt5
            with config.MT5_LOCK:
                existing_positions = mt5.positions_get(symbol=mt5_symbol)
                if has_open_position_for_symbol(existing_positions, mt5_symbol):
                    action_str = f"REJECTED: Open position already exists for {mt5_symbol}"
                    print(f"  [Action] {action_str}")
                    self._write_monitor_state(pred_data, action_str, tp_price, sl_price, spread_val=0.0, llm_reasoning=llm_reasoning_log)
                    return

                symbol_info = mt5.symbol_info(mt5_symbol)
                if symbol_info is None:
                    action_str = f"Error: MT5 symbol info not found for {mt5_symbol}"
                    print(f"  [Error] MT5 symbol info not found for {mt5_symbol}.")
                    self._write_monitor_state(pred_data, action_str, tp_price, sl_price, llm_reasoning=llm_reasoning_log)
                    return
                    
                spread_points = symbol_info.spread
                
                spread_dollar = spread_points * self.profile["pip_scale"]
                relative_spread = spread_dollar / (atr + 1e-5)
                if relative_spread > 0.15:
                    action_str = f"Suppressed: Relative spread ({relative_spread:.2%}) > 15% of ATR"
                    print(f"  [Suppressed] {action_str}")
                    self._write_monitor_state(pred_data, action_str, tp_price, sl_price, spread_val=spread_points, llm_reasoning=llm_reasoning_log)
                    return
                
                if spread_points > max_spread:
                    action_str = f"Suppressed: Spread ({spread_points} pts) > Max ({max_spread} pts)"
                    print(f"  [Suppressed] Spread ({spread_points} pts) exceeds maximum allowable ({max_spread} pts).")
                    self._write_monitor_state(pred_data, action_str, tp_price, sl_price, spread_val=spread_points, llm_reasoning=llm_reasoning_log)
                    return
                    
                try:
                    account_info = mt5.account_info()
                    equity = account_info.equity if account_info else 10000.0
                except Exception:
                    equity = 10000.0
                    
                risk_pct = 0.01
                sl_dist_val = config.SL_MULT * atr
                prof_spread = self.profile['spread_dollar']
                prof_slippage = self.profile['slippage_dollar']
                commission = self.profile['commission_per_lot']
                contract_size = self.profile['contract_size']
                
                loss_per_lot = (sl_dist_val + prof_spread + prof_slippage) * contract_size + commission
                trade_lots = max(0.01, round((equity * risk_pct) / loss_per_lot, 2)) if loss_per_lot > 0 else 0.01
                trade_lots = min(trade_lots, 5.0)
                
            order_type = mt5.ORDER_TYPE_BUY if signal_code == 1.0 else mt5.ORDER_TYPE_SELL
            price = symbol_info.ask if signal_code == 1.0 else symbol_info.bid
            
            sl_rounded = round(sl_price, symbol_info.digits)
            tp_rounded = round(tp_price, symbol_info.digits)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": trade_lots,
                "type": order_type,
                "price": price,
                "sl": sl_rounded,
                "tp": tp_rounded,
                "deviation": 20,
                "magic": 20260727,
                "comment": f"AI {self.symbol_key} {confidence:.1f}%",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            with config.MT5_LOCK:
                result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                if result.retcode == 10027:
                    action_str = "Order Failed: 10027 (Auto Trading Disabled in MT5)"
                    print(f"  [Order Failed] MT5 order_send error: 10027 (Auto Trading Disabled). Please enable 'Algo Trading' in MT5.")
                else:
                    action_str = f"Order Failed: retcode {result.retcode} ({result.comment})"
                    print(f"  [Order Failed] MT5 order_send error code: {result.retcode} ({result.comment})")
                self._write_monitor_state(pred_data, action_str, tp_rounded, sl_rounded, spread_val=spread_points, llm_reasoning=llm_reasoning_log)
            else:
                action_str = f"ORDER EXECUTED #{result.order} ({result.volume} lots @ {result.price:,.{decimals}f})"
                print(f"  [ORDER EXECUTED] Ticket #{result.order} | Volume: {result.volume} lots at {result.price:,.{decimals}f}")
                
                # Enforce SL/TP on Market Execution brokers where initial SL/TP might be ignored
                mod_request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": result.order,
                    "symbol": mt5_symbol,
                    "sl": sl_rounded,
                    "tp": tp_rounded,
                    "magic": 20260727
                }
                with config.MT5_LOCK:
                    mod_res = mt5.order_send(mod_request)
                if mod_res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"  [SL/TP Enforced] Attached SL: {sl_rounded:,.{decimals}f} | TP: {tp_rounded:,.{decimals}f} to ticket #{result.order}")
                else:
                    print(f"  [Warning] Could not attach SL/TP (retcode {mod_res.retcode}: {mod_res.comment}). Check if Algo Trading is enabled.")
                    
                self._write_monitor_state(pred_data, action_str, tp_rounded, sl_rounded, spread_val=spread_points, llm_reasoning=llm_reasoning_log)
        except Exception as e:
            action_str = f"Execution Error: {e}"
            print(f"  [Execution Error] {e}")
            self._write_monitor_state(pred_data, action_str, tp_price, sl_price, llm_reasoning=llm_reasoning_log)
        print("-" * 60)
        
    def _is_new_candle(self, symbol, timeframe):
        """Checks if a new candle has officially closed."""
        try:
            if self.data_source == "mt5":
                import MetaTrader5 as mt5
                with config.MT5_LOCK:
                    if not mt5.initialize():
                        return False
                    
                    mapping = {
                    "1m": mt5.TIMEFRAME_M1,
                    "5m": mt5.TIMEFRAME_M5,
                    "15m": mt5.TIMEFRAME_M15,
                    "30m": mt5.TIMEFRAME_M30,
                    "1h": mt5.TIMEFRAME_H1,
                    "4h": mt5.TIMEFRAME_H4,
                    "1d": mt5.TIMEFRAME_D1,
                }
                tf_const = mapping.get(timeframe, mt5.TIMEFRAME_M15)
                rates = mt5.copy_rates_from_pos(self.profile["mt5_ticker"], tf_const, 0, 2)
                if rates is None or len(rates) < 2:
                    return False
                
                # rates[-2] is the last closed candle. rates[-1] is the current forming candle.
                current_closed_time = pd.to_datetime(rates[-2]['time'], unit='s')
                if current_closed_time != self.last_candle_time:
                    self.last_candle_time = current_closed_time
                    return True
                return False
            else:
                df = self.loader.fetch_latest_buffer(symbol, timeframe, buffer_size=5)
                if df is None or df.empty:
                    return False
                current_closed_time = df.index[-1]
                if current_closed_time != self.last_candle_time:
                    self.last_candle_time = current_closed_time
                    return True
                return False
        except Exception as e:
            print(f"[LiveEngine/{self.symbol_key}] _is_new_candle error: {e}")
            return False

    def run_polling_loop(self, symbol=None, timeframe=config.TIMEFRAME_LTF, max_iterations=None):
        """
        Single-Threaded State Machine loop to monitor the market, manage trades, and run predictions safely.
        """
        if symbol is None:
            symbol = get_ticker_for_source(self.symbol_key, self.data_source)
        
        print(f"\n==================================================")
        print(f"STARTING REAL-TIME LIVE PREDICTION ENGINE — {self.profile['display_name']}")
        print(f"==================================================")
        print(f"Symbol: {symbol} | Timeframe: {timeframe} | Source: {self.data_source.upper()}")
        print(f"Architecture: Single-Threaded State Machine")
        print(f"Live Trading Execution: {'ENABLED (REAL ORDERS)' if config.ENABLE_LIVE_TRADING else 'ADVISORY / DRY-RUN MODE'}")
        print(f"Press Ctrl+C to stop.\n")
        
        iteration = 0
        
        try:
            while True:
                if max_iterations is not None and iteration >= max_iterations:
                    print(f"\n[LiveEngine/{self.symbol_key}] Reached max_iterations ({max_iterations}). Terminating loop.")
                    break
                    
                iteration += 1
                
                try:
                    # STATE 1: Manage existing trades EVERY TICK (Fast tick)
                    self.ea_manager.manage_open_trades(self.latest_live_df)
                    
                    # Optional LLM trailing stop updates
                    try:
                        from llm_reasoner import auto_adjust_all_open_positions
                        auto_adjust_all_open_positions(verbose=False)
                    except Exception:
                        pass
                    
                    # STATE 2: Check for new candle (Slow tick)
                    if self._is_new_candle(symbol, timeframe):
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{self.symbol_key}] New candle closed. Calculating features...")
                        
                        # STATE 3: Heavy Calculation & Entry (EVALUATE ON CLOSED CANDLE -2)
                        pred_data = self.get_live_prediction(symbol, timeframe, target_bar_idx=-2)
                        if pred_data is not None:
                            self.last_pred_data = pred_data
                            self.execute_trade(pred_data)
                    else:
                        if iteration % 15 == 0:  # Heartbeat ~every 30 seconds (if 2s sleep)
                            action_str = f"Monitoring Market (Waiting for next bar close...)"
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{self.symbol_key}] Heartbeat: Waiting for new candle close...")
                            self._write_monitor_state(self.last_pred_data, action_str)
                            
                    # Yield CPU (2 seconds)
                    if max_iterations is not None:
                        time.sleep(1)
                    else:
                        time.sleep(2)
                        
                except Exception as loop_e:
                    print(f"  [LiveEngine/{self.symbol_key}] Loop iteration error: {loop_e}")
                    time.sleep(5)  # Prevent crash loop on disconnect
                    
        except KeyboardInterrupt:
            print(f"\n[LiveEngine/{self.symbol_key}] Stopped by user.")
        except Exception as e:
            print(f"\n[LiveEngine/{self.symbol_key}] Fatal Error in polling loop: {e}")
            raise
