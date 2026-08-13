"""
ANTIGRAVITY QUANT // EXPERT ADVISOR TRADE MANAGER
Manages active MetaTrader 5 positions, evaluates real-time emergency exits
(AI reversals, market structure breaks, news blackouts), and coordinates risk tracking.
"""

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - fallback for non-windows / test environments
    mt5 = None

import os
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Any

import pandas as pd
import numpy as np

import config
from risk_manager import RiskManager


class ExpertAdvisorManager:
    """
    Expert Advisor Manager for real-time MT5 position management, risk checks,
    and emergency exit signals.
    """

    def __init__(
        self,
        symbol: str,
        primary_model: Any,
        meta_model: Any,
        feature_cols: List[str],
        live_engine: Optional[Any] = None
    ) -> None:
        self.symbol: str = symbol
        self.profile: dict = config.get_symbol_profile(symbol)
        self.primary_model: Any = primary_model
        self.meta_model: Any = meta_model
        self.feature_cols: List[str] = feature_cols
        self.live_engine: Optional[Any] = live_engine  # Reference to fetch latest_live_df
        
        self.risk_manager: Optional[RiskManager] = None
        if live_engine is not None and hasattr(live_engine, 'risk_manager'):
            self.risk_manager = live_engine.risk_manager

    def _is_mt5_available(self) -> bool:
        """Helper to check if MT5 package is loaded."""
        return mt5 is not None

    def get_open_positions(self) -> Optional[Any]:
        """Fetch all open positions for this symbol from MT5."""
        with config.MT5_LOCK:
            if not self._is_mt5_available() or not mt5.initialize():
                print("[EA MANAGER] MT5 initialization failed or package not available.")
                return None
            mt5_symbol = self.profile.get("mt5_ticker", self.symbol)
            return mt5.positions_get(symbol=mt5_symbol)

    def close_position(self, position: Any, reason: str = "AI_REVERSAL") -> None:
        """Closes a specific MT5 position immediately."""
        with config.MT5_LOCK:
            if not self._is_mt5_available():
                print("[EA MANAGER] MetaTrader5 is not available in this environment.")
                return

            mt5_symbol = self.profile.get("mt5_ticker", self.symbol)
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                print(f"[EA MANAGER] Could not fetch tick for {mt5_symbol}")
                return

            request_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": position.volume,
                "type": request_type,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"EA_EXIT: {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[EA MANAGER] Closed Ticket #{position.ticket} | Reason: {reason} | Price: {price}")
        else:
            retcode = result.retcode if result else "UNKNOWN"
            if retcode == 10027:
                print("[EA MANAGER] Close Failed (Auto Trading Disabled in MT5). Please enable 'Algo Trading'.")
            else:
                print(f"[EA MANAGER] Close Failed. Retcode: {retcode}")

    def _fetch_calendar_data(self) -> List[dict]:
        """Fetches economic calendar data from cache or ForexFactory JSON feed."""
        cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calendar.json")
        now = datetime.now()

        # 1. Check local cache (fresh for up to 6 hours)
        if os.path.exists(cache_file):
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (now - mtime) < timedelta(hours=6):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    print(f"[EA MANAGER] Cache reading error ({cache_file}): {e}")

        # 2. Fetch fresh data from live API
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            print(f"[EA MANAGER] Live news fetch failed: {e}")
            # Fallback to existing cache if available
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
            return []

    def _check_news_blackout(self, events: List[dict]) -> bool:
        """Checks if current UTC time falls within any high-impact news blackout window."""
        if not events:
            return False

        now_utc = datetime.now(timezone.utc)
        for event in events:
            if event.get("impact") != "High":
                continue

            country = event.get("country", "")
            if country not in ["USD", "EUR"]:
                continue

            event_date_str = event.get("date")
            if not event_date_str:
                continue

            try:
                event_time = datetime.fromisoformat(event_date_str)
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            # Blackout window: [-5 min, +10 min] around event
            time_diff_min = (now_utc - event_time).total_seconds() / 60.0
            if -5.0 <= time_diff_min <= 10.0:
                print(f"[EA MANAGER] NEWS BLACKOUT ACTIVE: {event.get('title')} ({country}) at {event_date_str}")
                return True

        return False

    def is_news_blackout(self) -> bool:
        """Main entry point for dynamic news blackout checks."""
        events = self._fetch_calendar_data()
        return self._check_news_blackout(events)

    def _check_news_exit(self, position: Any) -> Tuple[bool, str]:
        """Evaluates emergency exit if a news blackout event is currently occurring."""
        if self.is_news_blackout():
            profit = getattr(position, "profit", 0.0)
            if profit > 0:
                return True, "NEWS_PROFIT_TAKE"
            # Cut loss before SL gets jumped by news slippage
            if profit < -(position.volume * 50.0):
                return True, "NEWS_CUT_LOSS"
        return False, "HOLD"

    def _check_ai_reversal_exit(self, direction: int, live_features_df: pd.DataFrame) -> Tuple[bool, str]:
        """Evaluates AI model ensemble probability for direction reversal."""
        if live_features_df is None or len(live_features_df) < 2:
            return False, "HOLD"

        try:
            X_live = live_features_df[self.feature_cols].iloc[-2:].values

            # Use Ensemble (XGBoost + LightGBM) if available
            if self.live_engine and hasattr(self.live_engine, "xgb_model") and hasattr(self.live_engine, "lgb_model"):
                xgb_proba = self.live_engine.xgb_model.predict_proba(X_live)[-1]
                lgb_proba = self.live_engine.lgb_model.predict_proba(X_live)[-1]
                current_proba = (xgb_proba + lgb_proba) / 2.0
            elif hasattr(self.primary_model, "predict_proba"):
                current_proba = self.primary_model.predict_proba(X_live)[-1]
            else:
                return False, "HOLD"

            current_pred = int(np.argmax(current_proba)) - 1  # [0, 1, 2] -> [-1, 0, 1]
            max_prob = float(np.max(current_proba))

            # Long trade (1) but AI predicts Short (-1) with >60% confidence
            if direction == 1 and current_pred == -1 and max_prob > 0.60:
                return True, "AI_BEARISH_REVERSAL"

            # Short trade (-1) but AI predicts Long (1) with >60% confidence
            if direction == -1 and current_pred == 1 and max_prob > 0.60:
                return True, "AI_BULLISH_REVERSAL"

        except Exception as e:
            print(f"[EA MANAGER] Error evaluating AI reversal: {e}")

        return False, "HOLD"

    def _check_choch_break_exit(self, direction: int, live_features_df: pd.DataFrame) -> Tuple[bool, str]:
        """Evaluates Change of Character (CHoCH) market structure breaks."""
        if live_features_df is None or 'trend_direction' not in live_features_df.columns:
            return False, "HOLD"

        try:
            current_trend = live_features_df['trend_direction'].iloc[-2]
            if direction == 1 and current_trend == -1.0:
                return True, "BEARISH_CHOCH_BREAK"
            if direction == -1 and current_trend == 1.0:
                return True, "BULLISH_CHOCH_BREAK"
        except Exception as e:
            print(f"[EA MANAGER] Error evaluating CHoCH break: {e}")

        return False, "HOLD"

    def evaluate_emergency_exit(self, position: Any, live_features_df: Optional[pd.DataFrame]) -> Tuple[bool, str]:
        """
        Evaluates whether an open position should be exited early due to news,
        AI reversals, or market structure shifts.
        """
        if not self._is_mt5_available():
            return False, "HOLD"

        direction = 1 if position.type == mt5.POSITION_TYPE_BUY else -1

        # 1. News Exit Check
        should_close, reason = self._check_news_exit(position)
        if should_close:
            return True, reason

        # 2. AI Reversal Check
        should_close, reason = self._check_ai_reversal_exit(direction, live_features_df)
        if should_close:
            return True, reason

        # 3. Market Structure CHoCH Check
        should_close, reason = self._check_choch_break_exit(direction, live_features_df)
        if should_close:
            return True, reason

        return False, "HOLD"

    def manage_open_trades(self, live_features_df: Optional[pd.DataFrame]) -> None:
        """Called every tick to inspect and manage open positions."""
        try:
            positions = self.get_open_positions()
            if positions is None or len(positions) == 0:
                return

            for pos in positions:
                # Ignore manual trades. Only manage trades opened by this EA (magic: 234000)
                if getattr(pos, "magic", 0) != 234000:
                    continue

                should_close, reason = self.evaluate_emergency_exit(pos, live_features_df)
                if should_close:
                    self.close_position(pos, reason)
                    if self.risk_manager is not None:
                        equity = None
                        if self.live_engine is not None and hasattr(self.live_engine, 'latest_live_df'):
                            df = self.live_engine.latest_live_df
                            if df is not None and not df.empty:
                                equity = float(df.iloc[-1].get('close', 0.0))
                        self.risk_manager.record_trade_result(
                            float(getattr(pos, 'profit', 0.0)),
                            current_bar=None,
                            equity=equity
                        )
        except Exception as e:
            print(f"[EA MANAGER] Exception during trade management: {e}")
