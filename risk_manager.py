import json
import os
from datetime import datetime, timezone


class RiskManager:
    """Persisted risk guard for daily loss, consecutive-loss, and drawdown limits."""

    def __init__(self, max_daily_loss=200.0, max_consecutive_losses=3, max_drawdown_pct=0.10, state_file="risk_state.json"):
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_pct = max_drawdown_pct
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self):
        if not self.state_file:
            return {
                "daily_loss": 0.0,
                "consecutive_losses": 0.0,
                "peak_equity": 0.0,
                "last_reset": datetime.now(timezone.utc).date().isoformat(),
            }

        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if not isinstance(data, dict):
                    raise ValueError("Invalid risk state")
                return {
                    "daily_loss": float(data.get("daily_loss", 0.0)),
                    "consecutive_losses": int(data.get("consecutive_losses", 0)),
                    "peak_equity": float(data.get("peak_equity", 0.0)),
                    "last_reset": data.get("last_reset", datetime.now(timezone.utc).date().isoformat()),
                }
            except Exception:
                pass

        return {
            "daily_loss": 0.0,
            "consecutive_losses": 0,
            "peak_equity": 0.0,
            "last_reset": datetime.now(timezone.utc).date().isoformat(),
        }

    def _save_state(self):
        if not self.state_file:
            return
        directory = os.path.dirname(self.state_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2)

    def _reset_daily_if_needed(self, current_bar=None):
        today = datetime.now(timezone.utc).date().isoformat()
        if self.state.get("last_reset") != today:
            self.state["daily_loss"] = 0.0
            self.state["consecutive_losses"] = 0
            self.state["peak_equity"] = 0.0
            self.state["last_reset"] = today
            self._save_state()

    def can_trade(self, current_bar=None, equity=None):
        reason = self.get_trade_block_reason(current_bar=current_bar, equity=equity)
        return (reason is None, reason)

    def get_trade_block_reason(self, current_bar=None, equity=None):
        self._reset_daily_if_needed(current_bar=current_bar)
        if abs(self.state["daily_loss"]) >= self.max_daily_loss:
            return "daily_loss_limit"
        if self.state["consecutive_losses"] >= self.max_consecutive_losses:
            return "consecutive_loss_limit"
        if equity is not None and self.state["peak_equity"] > 0:
            drawdown_pct = (self.state["peak_equity"] - equity) / self.state["peak_equity"]
            if drawdown_pct >= self.max_drawdown_pct:
                return "drawdown_limit"
        return None

    def record_trade_result(self, pnl, current_bar=None, equity=None):
        self._reset_daily_if_needed(current_bar=current_bar)
        self.state["daily_loss"] += float(pnl)
        if pnl < 0:
            self.state["consecutive_losses"] += 1
        else:
            self.state["consecutive_losses"] = 0
        if equity is not None:
            self.state["peak_equity"] = max(self.state.get("peak_equity", 0.0), float(equity))
        self._save_state()

    def get_state(self):
        self._reset_daily_if_needed()
        return dict(self.state)
