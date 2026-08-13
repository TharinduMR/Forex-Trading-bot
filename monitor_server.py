import os
import glob
import json
import time
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import config

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

# MT5 Connection with cooldown to prevent log spam
_mt5_connected = False
_mt5_last_attempt = 0

def ensure_mt5():
    """Initialize MT5 with 30-second cooldown between retry attempts."""
    global _mt5_connected, _mt5_last_attempt
    if mt5 is None:
        return False
    if _mt5_connected:
        # Verify connection is still alive
        try:
            info = mt5.account_info()
            if info is not None:
                return True
            _mt5_connected = False
        except Exception:
            _mt5_connected = False
    now = time.time()
    if now - _mt5_last_attempt < 30:  # Cooldown: retry every 30s
        return False
    _mt5_last_attempt = now
    try:
        _mt5_connected = mt5.initialize()
        if not _mt5_connected:
            print(f"  [MT5 Monitor] MT5 connection unavailable (retry in 30s). Dashboard uses cached engine state.")
    except Exception:
        _mt5_connected = False
    return _mt5_connected

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/symbols")
def get_symbols():
    """Return all configured symbol profiles for the dashboard to enumerate."""
    profiles = {}
    for key, profile in config.SYMBOL_PROFILES.items():
        profiles[key] = {
            "name": profile["name"],
            "display_name": profile["display_name"],
            "mt5_ticker": profile["mt5_ticker"],
            "price_decimals": profile["price_decimals"],
            "price_format": profile["price_format"],
            "max_spread_points": profile["max_spread_points"],
            "trade_volume_lots": profile["trade_volume_lots"],
        }
    return jsonify(profiles)

@app.route("/api/status")
def get_status():
    """Return status for a specific symbol (query param ?symbol=XAUUSD, default XAUUSD)."""
    ensure_mt5()
    
    symbol_key = request.args.get("symbol", "XAUUSD").upper().strip()
    timeframe = request.args.get("timeframe", "15m").lower().strip()
    try:
        profile = config.get_symbol_profile(symbol_key)
    except KeyError:
        profile = config.SYMBOL_PROFILES.get("XAUUSD", list(config.SYMBOL_PROFILES.values())[0])
        symbol_key = profile["name"]
    
    mt5_ticker = profile["mt5_ticker"]
    decimals = profile["price_decimals"]
    
    # Base account stats (shared across all symbols)
    account_data = {
        "connected": False,
        "login": 0,
        "balance": 0.0,
        "equity": 0.0,
        "margin": 0.0,
        "free_margin": 0.0,
        "margin_level": 0.0,
        "profit": 0.0,
        "realized_pnl": 0.0,
        "total_pnl": 0.0,
        "server_time": datetime.now().strftime("%H:%M:%S")
    }
    
    if mt5 is not None:
        info = mt5.account_info()
        if info is not None:
            # Calculate closed deals / realized PnL
            realized_val = 0.0
            from_date = datetime(2020, 1, 1)
            to_date = datetime.now() + timedelta(days=7)
            deals = mt5.history_deals_get(from_date, to_date)
            if deals:
                for d in deals:
                    entry_val = getattr(d, "entry", -1)
                    if entry_val in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT, mt5.DEAL_ENTRY_OUT_BY]:
                        realized_val += (getattr(d, "profit", 0.0) + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0))

            account_data.update({
                "connected": True,
                "login": info.login,
                "balance": info.balance,
                "equity": info.equity,
                "margin": info.margin,
                "free_margin": info.margin_free,
                "margin_level": info.margin_level if info.margin > 0 else 9999.0,
                "profit": info.profit,
                "realized_pnl": realized_val,
                "total_pnl": realized_val + info.profit
            })
            
    # Market Quote for the requested symbol
    quote_data = {
        "symbol": mt5_ticker,
        "bid": 0.0,
        "ask": 0.0,
        "spread": 0,
        "spread_dollar": 0.0
    }
    
    if mt5 is not None:
        tick = mt5.symbol_info_tick(mt5_ticker)
        if tick is not None:
            pip_scale = profile.get("pip_scale", 0.01)
            spread_pts = int(round((tick.ask - tick.bid) / pip_scale))
            quote_data.update({
                "bid": tick.bid,
                "ask": tick.ask,
                "spread": spread_pts,
                "spread_dollar": tick.ask - tick.bid
            })
            
    # Read per-symbol monitor state JSON from live engine
    engine_state = {
        "timestamp": "--",
        "symbol": mt5_ticker,
        "symbol_key": symbol_key,
        "display_name": profile["display_name"],
        "timeframe": timeframe,
        "close": 0.0,
        "signal": "WAITING FOR ENGINE...",
        "signal_code": 0.0,
        "confidence": 0.0,
        "prob_long": 0.0,
        "prob_flat": 100.0,
        "prob_short": 0.0,
        "meta_prob": 1.0,
        "atr": 0.0,
        "tp_price": 0.0,
        "sl_price": 0.0,
        "action": "Engine idle or starting...",
        "last_update": "--",
        "live_trading_enabled": config.ENABLE_LIVE_TRADING
    }
    
    # Read the per-timeframe engine state JSON (model probabilities, ATR, etc.)
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"monitor_state_{symbol_key}_{timeframe}.json")
    
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                saved_state = json.load(f)
                engine_state.update(saved_state)
        except Exception:
            pass
    
    # ── GLOBAL TRADE STATE: Read unified signal from unified_engine ──
    # This overrides the per-timeframe signal so the dashboard shows ONE
    # locked signal regardless of which timeframe chart the user views.
    global_state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_trade_state.json")
    global_trade = {
        "active": False,
        "global_signal": "NO TRADE",
        "global_signal_code": 0.0,
        "entry_price": 0.0,
        "take_profit": 0.0,
        "stop_loss": 0.0,
        "locked_at": None,
        "status": "UNIFIED ENGINE NOT RUNNING",
    }
    
    if os.path.exists(global_state_file):
        try:
            with open(global_state_file, "r") as f:
                gs = json.load(f)
                global_trade.update(gs)
                global_trade["active"] = gs.get("global_signal_code", 0.0) != 0.0
                
                # Override engine_state signal fields with the locked global decision
                if global_trade["active"]:
                    engine_state["signal"] = gs.get("global_signal", engine_state["signal"])
                    engine_state["signal_code"] = gs.get("global_signal_code", engine_state["signal_code"])
                    engine_state["tp_price"] = gs.get("take_profit", engine_state["tp_price"])
                    engine_state["sl_price"] = gs.get("stop_loss", engine_state["sl_price"])
                    engine_state["action"] = f"GLOBAL LOCKED: {gs.get('global_signal', 'N/A')} (since {gs.get('locked_at', '?')})"
        except Exception:
            pass
            
    return jsonify({
        "account": account_data,
        "quote": quote_data,
        "engine": engine_state,
        "global_trade": global_trade,
        "config": {
            "meta_threshold": config.META_CONFIDENCE_THRESHOLD * 100.0,
            "max_spread_points": profile["max_spread_points"],
            "trade_volume_lots": profile["trade_volume_lots"],
            "conf_threshold": config.CONFIDENCE_THRESHOLD * 100.0,
            "symbol": mt5_ticker,
            "symbol_key": symbol_key,
            "display_name": profile["display_name"],
            "timeframe": config.TIMEFRAME_LTF,
            "price_decimals": decimals,
            "price_format": profile["price_format"],
        }
    })

@app.route("/api/positions")
def get_positions():
    """Return open positions, optionally filtered by ?symbol= query param."""
    ensure_mt5()
    filter_symbol = request.args.get("symbol", "").upper().strip()
    
    positions_list = []
    if mt5 is not None:
        if filter_symbol:
            try:
                profile = config.get_symbol_profile(filter_symbol)
                mt5_ticker = profile["mt5_ticker"]
                pos = mt5.positions_get(symbol=mt5_ticker)
            except KeyError:
                pos = mt5.positions_get(symbol=filter_symbol)
        else:
            pos = mt5.positions_get()
            
        if pos is not None:
            for p in pos:
                positions_list.append({
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "time": datetime.fromtimestamp(p.time).strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "BUY / LONG" if p.type == 0 else "SELL / SHORT",
                    "volume": p.volume,
                    "price_open": p.price_open,
                    "price_current": p.price_current,
                    "sl": p.sl,
                    "tp": p.tp,
                    "profit": p.profit,
                    "comment": p.comment
                })
    return jsonify(positions_list)

@app.route("/api/trade_history")
def get_trade_history():
    ensure_mt5()
    history_list = []
    if mt5 is not None:
        from_date = datetime(2020, 1, 1)
        to_date = datetime.now() + timedelta(days=7)
        deals = mt5.history_deals_get(from_date, to_date)
        if deals:
            # Reverse so newest trades are at top
            for d in reversed(deals):
                sym_val = getattr(d, "symbol", "")
                entry_val = getattr(d, "entry", -1)
                if sym_val != "" and entry_val in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT, mt5.DEAL_ENTRY_OUT_BY]:
                    # To exit a Long (BUY) position, MT5 creates a SELL deal.
                    # So if the exit deal is SELL, the original trade was BUY.
                    was_long = (getattr(d, "type", 0) == mt5.DEAL_TYPE_SELL)
                    type_str = "BUY" if was_long else "SELL"
                    history_list.append({
                        "ticket": getattr(d, "ticket", 0),
                        "order": getattr(d, "order", 0),
                        "time": datetime.fromtimestamp(getattr(d, "time", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": sym_val,
                        "type": type_str,
                        "volume": float(getattr(d, "volume", 0.0)),
                        "price": float(getattr(d, "price", 0.0)),
                        "profit": float(getattr(d, "profit", 0.0) + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0)),
                        "comment": getattr(d, "comment", "")
                    })
    return jsonify(history_list)

@app.route("/api/chart_data")
def get_chart_data():
    ensure_mt5()
    tf_param = request.args.get("tf", "15m").lower()
    symbol_key = request.args.get("symbol", "XAUUSD").upper().strip()
    
    try:
        limit_param = int(request.args.get("limit", 0))
    except ValueError:
        limit_param = 0
    
    try:
        profile = config.get_symbol_profile(symbol_key)
        mt5_ticker = profile["mt5_ticker"]
    except KeyError:
        mt5_ticker = config.SYMBOL_MT5
    
    candles = []
    
    tf_mapping = {}
    if mt5 is not None:
        tf_mapping = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "2h": mt5.TIMEFRAME_H2,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
            "2d": mt5.TIMEFRAME_D1,
            "5d": mt5.TIMEFRAME_W1,
            "7d": mt5.TIMEFRAME_W1,
        }
    
    mt5_tf = tf_mapping.get(tf_param, mt5.TIMEFRAME_M15 if mt5 else None)
    
    if limit_param > 0:
        num_candles = min(limit_param, 2000)
    else:
        num_candles = 80
        if tf_param in ["1d", "2d", "5d", "7d"]:
            num_candles = 120
        
    if mt5 is not None and mt5_tf is not None:
        rates = mt5.copy_rates_from_pos(mt5_ticker, mt5_tf, 0, num_candles)
        if rates is not None and len(rates) > 0:
            is_daily = tf_param in ["1d", "2d", "5d", "7d"]
            for r in rates:
                dt = datetime.fromtimestamp(r['time'])
                time_str = dt.strftime("%b %d") if is_daily else dt.strftime("%H:%M")
                candles.append({
                    "time": time_str,
                    "full_time": dt.strftime("%Y-%m-%d %H:%M"),
                    "open": float(r['open']),
                    "high": float(r['high']),
                    "low": float(r['low']),
                    "close": float(r['close']),
                    "volume": int(r['tick_volume'])
                })
    return jsonify(candles)

@app.route("/api/logs")
def get_logs():
    # Scan task logs directory for most recent log file dynamically
    brain_root = r"C:\Users\Tharindu Madhusanka\.gemini\antigravity-ide\brain"
    log_lines = []
    try:
        if os.path.exists(brain_root):
            # Find all *.log files under all session tasks directories
            log_files = glob.glob(os.path.join(brain_root, "*", ".system_generated", "tasks", "*.log"))
            if log_files:
                newest_file = max(log_files, key=os.path.getmtime)
                with open(newest_file, "r", errors="replace") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    log_lines = lines[-50:]  # get last 50 lines
    except Exception as e:
        log_lines = [f"Error reading logs: {e}"]
        
    if not log_lines:
        log_lines = ["No recent logs found. Start 'python main.py --mode live' to begin log streaming."]
        
    return jsonify({"logs": log_lines})

@app.route("/api/close_position", methods=["POST"])
def close_position():
    ensure_mt5()
    if mt5 is None:
        return jsonify({"success": False, "message": "MT5 module not installed"}), 500
        
    data = request.get_json()
    ticket = data.get("ticket")
    if not ticket:
        return jsonify({"success": False, "message": "Position ticket required"}), 400
        
    pos = mt5.positions_get(ticket=int(ticket))
    if not pos or len(pos) == 0:
        return jsonify({"success": False, "message": f"Position #{ticket} not found in MT5"}), 404
    p = pos[0]
    
    symbol_info = mt5.symbol_info(p.symbol)
    if not symbol_info:
        return jsonify({"success": False, "message": f"Symbol info for {p.symbol} not found"}), 404
        
    order_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
    price = symbol_info.bid if order_type == mt5.ORDER_TYPE_SELL else symbol_info.ask
    
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": p.ticket,
        "symbol": p.symbol,
        "volume": p.volume,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 20260727,
        "comment": "Web UI Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(close_request)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        return jsonify({"success": True, "message": f"Position #{ticket} ({p.symbol}) closed at {price}"})
    elif res.retcode == 10027:
        return jsonify({"success": False, "message": "AutoTrading is disabled! Please click the green 'Algo Trading' button in MT5 toolbar."}), 400
    else:
        return jsonify({"success": False, "message": f"Close failed: retcode {res.retcode} ({res.comment})"}), 400

@app.route("/api/set_sltp", methods=["POST"])
def set_sltp():
    ensure_mt5()
    if mt5 is None:
        return jsonify({"success": False, "message": "MT5 module not installed"}), 500
        
    data = request.get_json()
    ticket = data.get("ticket")
    sl = float(data.get("sl", 0.0))
    tp = float(data.get("tp", 0.0))
    
    if not ticket:
        return jsonify({"success": False, "message": "Position ticket required"}), 400
        
    pos = mt5.positions_get(ticket=int(ticket))
    if not pos or len(pos) == 0:
        return jsonify({"success": False, "message": f"Position #{ticket} not found in MT5"}), 404
    p = pos[0]
    
    symbol_info = mt5.symbol_info(p.symbol)
    if not symbol_info:
        return jsonify({"success": False, "message": f"Symbol info for {p.symbol} not found"}), 404
        
    sl_val = round(sl, symbol_info.digits) if sl > 0 else 0.0
    tp_val = round(tp, symbol_info.digits) if tp > 0 else 0.0
    
    mod_request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": p.ticket,
        "symbol": p.symbol,
        "sl": sl_val,
        "tp": tp_val,
        "magic": 20260727
    }
    
    res = mt5.order_send(mod_request)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        return jsonify({"success": True, "message": f"SL/TP updated for #{ticket} ({p.symbol})"})
    elif res.retcode == 10027:
        return jsonify({"success": False, "message": "AutoTrading is disabled! Please click the green 'Algo Trading' button in MT5 toolbar."}), 400
    else:
        return jsonify({"success": False, "message": f"Update failed: retcode {res.retcode} ({res.comment})"}), 400


@app.route("/api/llm_analyze", methods=["POST"])
def llm_analyze():
    try:
        from llm_reasoner import analyze_market_and_positions
        data = request.get_json() or {}
        api_key = data.get("api_key") or None
        model_name = data.get("model_name") or None
        api_base_url = data.get("api_base_url") or None
        res = analyze_market_and_positions(api_key=api_key, model_name=model_name, api_base_url=api_base_url)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/llm_chat", methods=["POST"])
def llm_chat():
    try:
        from llm_reasoner import chat_with_quant
        data = request.get_json() or {}
        message = data.get("message", "")
        history = data.get("history", [])
        api_key = data.get("api_key") or None
        model_name = data.get("model_name") or None
        api_base_url = data.get("api_base_url") or None
        res = chat_with_quant(user_message=message, chat_history=history, api_key=api_key, model_name=model_name, api_base_url=api_base_url)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chart_overlays")
def get_chart_overlays():
    """Compute Support/Resistance zones, position overlays, and SMC/ICT setup markers for chart rendering."""
    connected = ensure_mt5()
    symbol_key = request.args.get("symbol", "XAUUSD").upper().strip()
    tf_param = request.args.get("tf", "15m").lower()
    
    try:
        profile = config.get_symbol_profile(symbol_key)
        mt5_ticker = profile["mt5_ticker"]
    except KeyError:
        mt5_ticker = config.SYMBOL_MT5
        profile = config.SYMBOL_PROFILES.get("XAUUSD", {})
    
    result = {
        "support_zones": [],
        "resistance_zones": [],
        "positions": [],
        "smc_setups": [],
        "ict_setups": [],
        "conviction_state": {"signal": "FLAT", "signal_code": 0.0, "held_since": None}
    }
    
    # ── Read conviction state: prefer global unified state, fall back to per-TF ──
    global_state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_trade_state.json")
    timeframe = tf_param
    per_tf_state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"monitor_state_{symbol_key}_{timeframe}.json")
    
    # Try global state first (unified engine)
    _conviction_loaded = False
    if os.path.exists(global_state_path):
        try:
            with open(global_state_path, "r") as f:
                gs = json.load(f)
                sig_code = gs.get("global_signal_code", 0.0)
                sig_label = "LONG" if sig_code == 1.0 else ("SHORT" if sig_code == -1.0 else "FLAT")
                result["conviction_state"] = {
                    "signal": sig_label,
                    "signal_code": sig_code,
                    "held_since": gs.get("locked_at", None)
                }
                _conviction_loaded = True
        except Exception:
            pass
    
    # Fall back to per-timeframe state if global state is unavailable
    if not _conviction_loaded and os.path.exists(per_tf_state_path):
        try:
            with open(per_tf_state_path, "r") as f:
                saved = json.load(f)
                sig_code = saved.get("conviction_state", saved.get("signal_code", 0.0))
                sig_label = "LONG" if sig_code == 1.0 else ("SHORT" if sig_code == -1.0 else "FLAT")
                result["conviction_state"] = {
                    "signal": sig_label,
                    "signal_code": sig_code,
                    "held_since": saved.get("conviction_held_since", None)
                }
        except Exception:
            pass
    
    if not connected or mt5 is None:
        return jsonify(result)
    
    # Fetch candle data for analysis
    tf_mapping = {
        "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5, "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30, "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    mt5_tf = tf_mapping.get(tf_param, mt5.TIMEFRAME_M15)
    num_candles = 120
    
    import numpy as np
    rates = mt5.copy_rates_from_pos(mt5_ticker, mt5_tf, 0, num_candles)
    if rates is None or len(rates) == 0:
        return jsonify(result)
    
    highs = np.array([r['high'] for r in rates], dtype=float)
    lows = np.array([r['low'] for r in rates], dtype=float)
    closes = np.array([r['close'] for r in rates], dtype=float)
    opens = np.array([r['open'] for r in rates], dtype=float)
    times = [datetime.fromtimestamp(r['time']).strftime("%H:%M") for r in rates]
    
    n = len(highs)
    swing_window = getattr(config, 'SWING_WINDOW', 2)
    
    # --- Detect Swing Highs and Swing Lows ---
    swing_highs = []
    swing_lows = []
    for i in range(swing_window, n - swing_window):
        # Swing High: high[i] is the max of the surrounding window
        if highs[i] == max(highs[i - swing_window: i + swing_window + 1]):
            swing_highs.append({"index": i, "price": float(highs[i])})
        # Swing Low: low[i] is the min of the surrounding window
        if lows[i] == min(lows[i - swing_window: i + swing_window + 1]):
            swing_lows.append({"index": i, "price": float(lows[i])})
    
    # --- Cluster swing levels into Support/Resistance zones ---
    atr_arr = np.zeros(n)
    for i in range(1, n):
        atr_arr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    avg_atr = float(np.mean(atr_arr[1:])) if n > 1 else 1.0
    cluster_threshold = avg_atr * 0.5  # Levels within 0.5 ATR are clustered
    
    def cluster_levels(swings, zone_type):
        if not swings:
            return []
        sorted_swings = sorted(swings, key=lambda x: x['price'])
        zones = []
        cluster = [sorted_swings[0]]
        for s in sorted_swings[1:]:
            if s['price'] - cluster[-1]['price'] <= cluster_threshold:
                cluster.append(s)
            else:
                prices = [c['price'] for c in cluster]
                zones.append({
                    "price_low": round(min(prices) - avg_atr * 0.1, profile.get('price_decimals', 2)),
                    "price_high": round(max(prices) + avg_atr * 0.1, profile.get('price_decimals', 2)),
                    "strength": len(cluster),
                    "label": f"{zone_type} (x{len(cluster)})"
                })
                cluster = [s]
        if cluster:
            prices = [c['price'] for c in cluster]
            zones.append({
                "price_low": round(min(prices) - avg_atr * 0.1, profile.get('price_decimals', 2)),
                "price_high": round(max(prices) + avg_atr * 0.1, profile.get('price_decimals', 2)),
                "strength": len(cluster),
                "label": f"{zone_type} (x{len(cluster)})"
            })
        return zones
    
    # Only use recent swing levels (last 80 bars visible on chart)
    visible_start = max(0, n - 80)
    recent_highs = [s for s in swing_highs if s['index'] >= visible_start]
    recent_lows = [s for s in swing_lows if s['index'] >= visible_start]
    
    result["resistance_zones"] = cluster_levels(recent_highs, "Resistance")
    result["support_zones"] = cluster_levels(recent_lows, "Support")
    
    # --- Detect SMC Setups (FVG, CHoCH, Sweep) in last 30 bars ---
    smc_setups = []
    lookback_start = max(0, n - 30)
    
    # Fair Value Gaps (FVG)
    for i in range(lookback_start + 2, n):
        # Bullish FVG: bar[i].low > bar[i-2].high (gap up)
        if lows[i] > highs[i - 2]:
            smc_setups.append({
                "type": "FVG",
                "bar_index": int(i - 1),  # Middle bar
                "price_top": float(lows[i]),
                "price_bottom": float(highs[i - 2]),
                "direction": "bullish"
            })
        # Bearish FVG: bar[i].high < bar[i-2].low (gap down)
        elif highs[i] < lows[i - 2]:
            smc_setups.append({
                "type": "FVG",
                "bar_index": int(i - 1),
                "price_top": float(lows[i - 2]),
                "price_bottom": float(highs[i]),
                "direction": "bearish"
            })
    
    # Change of Character (CHoCH) - structural break
    recent_swing_highs_idx = [s for s in swing_highs if s['index'] >= lookback_start]
    recent_swing_lows_idx = [s for s in swing_lows if s['index'] >= lookback_start]
    
    # Bearish CHoCH: price breaks below the last swing low
    for sl in recent_swing_lows_idx:
        for i in range(sl['index'] + 1, min(sl['index'] + 10, n)):
            if closes[i] < sl['price']:
                smc_setups.append({
                    "type": "CHoCH",
                    "bar_index": int(i),
                    "price": float(sl['price']),
                    "direction": "bearish"
                })
                break
    
    # Bullish CHoCH: price breaks above the last swing high
    for sh in recent_swing_highs_idx:
        for i in range(sh['index'] + 1, min(sh['index'] + 10, n)):
            if closes[i] > sh['price']:
                smc_setups.append({
                    "type": "CHoCH",
                    "bar_index": int(i),
                    "price": float(sh['price']),
                    "direction": "bullish"
                })
                break
    
    # Liquidity Sweeps - wick below swing low then close above it
    for sl in recent_swing_lows_idx:
        for i in range(sl['index'] + 1, min(sl['index'] + 10, n)):
            if lows[i] < sl['price'] and closes[i] > sl['price']:
                smc_setups.append({
                    "type": "SWEEP",
                    "bar_index": int(i),
                    "price": float(sl['price']),
                    "direction": "bullish"
                })
                break
    
    # Sweep above swing high
    for sh in recent_swing_highs_idx:
        for i in range(sh['index'] + 1, min(sh['index'] + 10, n)):
            if highs[i] > sh['price'] and closes[i] < sh['price']:
                smc_setups.append({
                    "type": "SWEEP",
                    "bar_index": int(i),
                    "price": float(sh['price']),
                    "direction": "bearish"
                })
                break
    
    result["smc_setups"] = smc_setups
    
    # --- ICT Setups ---
    ict_setups = []
    
    # OTE Zone (Fibonacci 62%-79% retracement of last significant swing)
    if len(recent_swing_highs_idx) > 0 and len(recent_swing_lows_idx) > 0:
        last_high = recent_swing_highs_idx[-1]
        last_low = recent_swing_lows_idx[-1]
        swing_range = last_high['price'] - last_low['price']
        if swing_range > 0 and abs(last_high['index'] - last_low['index']) >= 3:
            if last_high['index'] > last_low['index']:
                # Upswing: OTE retracement zone for long entry
                ote_high = last_high['price'] - 0.62 * swing_range
                ote_low = last_high['price'] - 0.79 * swing_range
                ict_setups.append({
                    "type": "OTE_ZONE",
                    "fib_high": round(float(ote_high), profile.get('price_decimals', 2)),
                    "fib_low": round(float(ote_low), profile.get('price_decimals', 2)),
                    "bar_start": int(last_low['index']),
                    "bar_end": int(last_high['index']),
                    "direction": "bullish"
                })
            else:
                # Downswing: OTE retracement zone for short entry
                ote_low = last_low['price'] + 0.62 * swing_range
                ote_high = last_low['price'] + 0.79 * swing_range
                ict_setups.append({
                    "type": "OTE_ZONE",
                    "fib_high": round(float(ote_high), profile.get('price_decimals', 2)),
                    "fib_low": round(float(ote_low), profile.get('price_decimals', 2)),
                    "bar_start": int(last_high['index']),
                    "bar_end": int(last_low['index']),
                    "direction": "bearish"
                })
    
    # Killzone bands (London 02:00-05:00 UTC, NY AM 07:00-10:00 UTC)
    kz_bars = []
    for i in range(visible_start, n):
        dt = datetime.fromtimestamp(rates[i]['time'])
        hour = dt.hour
        if 2 <= hour < 5:
            kz_bars.append({"bar_index": int(i), "session": "London"})
        elif 7 <= hour < 10:
            kz_bars.append({"bar_index": int(i), "session": "NY AM"})
    
    # Group consecutive killzone bars into ranges
    if kz_bars:
        current_kz = {"start_bar": kz_bars[0]['bar_index'], "end_bar": kz_bars[0]['bar_index'], "session": kz_bars[0]['session']}
        for kb in kz_bars[1:]:
            if kb['bar_index'] == current_kz['end_bar'] + 1 and kb['session'] == current_kz['session']:
                current_kz['end_bar'] = kb['bar_index']
            else:
                ict_setups.append({"type": "KILLZONE", **current_kz})
                current_kz = {"start_bar": kb['bar_index'], "end_bar": kb['bar_index'], "session": kb['session']}
        ict_setups.append({"type": "KILLZONE", **current_kz})
    
    result["ict_setups"] = ict_setups
    
    # --- Open Positions for chart overlay ---
    positions = []
    try:
        pos = mt5.positions_get(symbol=mt5_ticker)
        if pos:
            for p in pos:
                positions.append({
                    "entry": float(p.price_open),
                    "sl": float(p.sl),
                    "tp": float(p.tp),
                    "type": "LONG" if p.type == 0 else "SHORT",
                    "ticket": int(p.ticket),
                    "profit": float(p.profit)
                })
    except Exception:
        pass
    result["positions"] = positions
    
    return jsonify(result)

@app.route("/api/trade", methods=["POST"])
def execute_trade():
    """Manual execution of orders from dashboard."""
    connected = ensure_mt5()
    if not connected or mt5 is None:
        return jsonify({"success": False, "error": "MT5 not connected"})
        
    try:
        data = request.json
        symbol = data.get("symbol", "XAUUSD").upper()
        action_str = data.get("action", "").upper()
        volume = float(data.get("volume", 0.01))
        
        if action_str not in ["BUY", "SELL"] or volume <= 0:
            return jsonify({"success": False, "error": "Invalid parameters"})
            
        profile = config.get_symbol_profile(symbol)
        mt5_ticker = profile.get("mt5_ticker", config.SYMBOL_MT5)
        
        tick = mt5.symbol_info_tick(mt5_ticker)
        if tick is None:
            return jsonify({"success": False, "error": f"Failed to get tick for {mt5_ticker}"})
            
        order_type = mt5.ORDER_TYPE_BUY if action_str == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action_str == "BUY" else tick.bid
        
        sl = 0.0
        tp = 0.0
        default_sl_pips = profile.get("default_sl_pips", 50)
        pip_size = profile.get("pip_size", 0.01)
        if default_sl_pips:
            sl_dist = default_sl_pips * pip_size
            sl = price - sl_dist if action_str == "BUY" else price + sl_dist
            sl = round(sl, profile.get("price_decimals", 2))
            
        request_obj = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_ticker,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 999999,
            "comment": "Dashboard Manual",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        with config.MT5_LOCK:
            result = mt5.order_send(request_obj)
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            err = "Auto Trading Disabled" if result.retcode == 10027 else result.comment
            return jsonify({"success": False, "error": f"Retcode {result.retcode}: {err}"})
            
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("==================================================")
    print("STARTING INSTITUTIONAL REAL-TIME WEB DASHBOARD")
    print("Active Symbols:", list(config.SYMBOL_PROFILES.keys()))
    print("==================================================")
    print("Access the dashboard at: http://localhost:5000")
    print("Press Ctrl+C to stop the server.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
