import os
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

import config


def get_market_context_summary():
    """
    Gather complete live market state, open positions, recent trade history, and engine state
    for LLM reasoning and overview. Supports all configured symbols.
    """
    context = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "active_symbols": config.ACTIVE_SYMBOLS_MT5,
        "timeframe": config.TIMEFRAME_LTF,
        "engine_states": {},
        "open_positions": [],
        "recent_closed_trades": []
    }
    
    # 1. Load engine state from per-symbol monitor_state files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for sym_key in config.ACTIVE_SYMBOLS_MT5:
        state_file = os.path.join(base_dir, f"monitor_state_{sym_key}.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    context["engine_states"][sym_key] = json.load(f)
            except Exception as e:
                context["engine_states"][sym_key] = {"error": str(e)}
    # Fallback: legacy single-file monitor_state.json
    if not context["engine_states"]:
        legacy_file = os.path.join(base_dir, "monitor_state.json")
        if os.path.exists(legacy_file):
            try:
                with open(legacy_file, "r") as f:
                    context["engine_states"]["legacy"] = json.load(f)
            except Exception:
                pass
            
    # 2. Load open positions from MT5
    if mt5 is not None:
        try:
            if mt5.initialize():
                # Get positions for ALL symbols
                positions = mt5.positions_get()
                if positions:
                    for p in positions:
                        context["open_positions"].append({
                            "ticket": p.ticket,
                            "symbol": p.symbol,
                            "type": "BUY / LONG" if p.type == 0 else "SELL / SHORT",
                            "volume": p.volume,
                            "price_open": p.price_open,
                            "price_current": p.price_current,
                            "sl": p.sl,
                            "tp": p.tp,
                            "profit": p.profit,
                            "comment": p.comment
                        })
                        
                # 3. Recent closed trades for all symbols
                from_date = datetime.now() - timedelta(days=2)
                deals = mt5.history_deals_get(from_date, datetime.now())
                if deals:
                    symbol_deals = [d for d in deals if d.symbol != "" and d.entry in [1, 2]]
                    for d in symbol_deals[-5:]:  # last 5 closed trades
                        context["recent_closed_trades"].append({
                            "ticket": d.ticket,
                            "type": "BUY EXIT" if d.type == 0 else "SELL EXIT",
                            "volume": d.volume,
                            "price": d.price,
                            "profit": d.profit,
                            "time": datetime.fromtimestamp(d.time).strftime("%Y-%m-%d %H:%M:%S")
                        })
        except Exception:
            pass
            
    return context


def call_llm_api(messages, api_key=None, model_name=None, api_base_url=None, temperature=0.3):
    """
    Call any OpenAI-compatible LLM API (NVIDIA GPT-OSS-120B, GLM-5.2, GPT-4, etc.)
    using standard urllib without external dependencies.
    """
    if not api_key:
        api_key = getattr(config, "LLM_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
    if not model_name:
        model_name = getattr(config, "LLM_MODEL_NAME", "openai/gpt-oss-120b")
    if not api_base_url:
        api_base_url = getattr(config, "LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        
    if not api_key:
        return {
            "error": "API Key is missing in config.py or environment."
        }
        
    if not api_base_url.endswith("/chat/completions"):
        api_base_url = api_base_url.rstrip("/") + "/chat/completions"
            
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096
    }
    
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(api_base_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            if "choices" in res_data and len(res_data["choices"]) > 0:
                content = res_data["choices"][0]["message"]["content"]
                return {"success": True, "content": content, "raw": res_data}
            else:
                return {"error": "Unexpected API response structure.", "raw": res_data}
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        return {"error": f"HTTP Error {e.code}: {e.reason} - {err_msg}"}
    except Exception as e:
        return {"error": f"API Connection Error: {str(e)}"}


def analyze_market_and_positions(api_key=None, model_name=None, api_base_url=None):
    """
    Generate an institutional reasoning report AND automatically analyze and adjust all open position SL/TP
    levels in MT5 based on real-time quantitative rules (ATR, Breakeven, Trailing Stops, Fibonacci OTEs).
    """
    context = get_market_context_summary()
    
    # 1. ALWAYS run quantitative position adjustment on MT5 first
    quant_execution_results = auto_adjust_all_open_positions(verbose=True, force_recalculate=True)
    
    # 2. Call LLM for institutional narrative analysis
    system_prompt = (
        "You are ANTIGRAVITY AI QUANT, a senior institutional quantitative portfolio manager and risk analyst. "
        "You have access to live MT5 market data, open positions, closed trade history, and quantitative feature "
        "matrices (Fibonacci OTEs, MACD momentum, FLOD clusters, killzones).\n\n"
        f"Current Market & Portfolio Context:\n{json.dumps(context, indent=2)}\n\n"
        "Your task:\n"
        "1. Provide a comprehensive narrative analysis of active instruments (XAUUSD and EURUSD) market structure, HTF bias, and liquidity setups.\n"
        "2. Evaluate all open positions across all symbols, risk-reward ratios, and recent closed trade profitability.\n"
        "3. Provide your professional verdict on current open positions and risk exposure.\n\n"
        "Answer clearly, concisely, and with institutional authority using Markdown and LaTeX math formatting."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Generate institutional quantitative reasoning report and evaluate all open positions."}
    ]
    
    res = call_llm_api(messages, api_key=api_key, model_name=model_name, api_base_url=api_base_url, temperature=0.3)
    
    summary_text = res.get("content", "Quantitative analysis completed.") if res.get("success") else f"Quantitative Analysis Completed. (LLM Status: {res.get('error')})"
    
    return {
        "success": True,
        "summary": summary_text,
        "recommendations": [],
        "execution_results": quant_execution_results,
        "context_used": context
    }


def move_all_to_breakeven(verbose=True):
    """
    Agentic Action Handler:
    Moves Stop Loss of all open MT5 positions to Breakeven (entry price + small buffer to cover spread/commissions).
    For SHORT (SELL) positions: new_sl = price_open + 0.20 (or entry price if currently above entry).
    For LONG (BUY) positions: new_sl = price_open - 0.20 (or entry price if currently below entry).
    """
    if mt5 is None or not mt5.initialize():
        return ["[ERROR] Could not connect to MT5 for breakeven adjustment."]
        
    positions = mt5.positions_get()
    if not positions:
        return ["No open positions in MT5 to move to breakeven."]
        
    results = []
    for p in positions:
        symbol = p.symbol
        symbol_info = mt5.symbol_info(symbol)
        digits = symbol_info.digits if symbol_info else 2
        
        # Calculate breakeven SL with symbol-appropriate buffer
        try:
            profile = config.get_symbol_profile(symbol)
            buffer = profile["breakeven_buffer"]
        except (KeyError, Exception):
            buffer = 0.20 if ("XAU" in symbol or "GOLD" in symbol) else 0.0002
        if p.type == mt5.ORDER_TYPE_BUY:
            be_sl = round(p.price_open + buffer, digits)
            if p.price_current < p.price_open:
                be_sl = round(p.price_open, digits)
        else:  # SELL / SHORT
            be_sl = round(p.price_open + buffer, digits)
            if p.price_current > p.price_open:
                be_sl = round(p.price_open + buffer, digits)
                
        if abs(p.sl - be_sl) > 0.0001:
            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "symbol": symbol,
                "sl": be_sl,
                "tp": p.tp,
                "magic": 20260727,
                "comment": "AI Quant Breakeven SL"
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                msg = f"[AGENTIC BREAKEVEN] Ticket #{p.ticket} ({'BUY' if p.type==0 else 'SELL'} {symbol}): Moved Stop Loss to Breakeven @ ${be_sl:,.2f} (Entry: ${p.price_open:,.2f})"
                results.append(msg)
                if verbose: print(f"  {msg}")
            else:
                msg = f"[FAILED BREAKEVEN] Ticket #{p.ticket}: Could not update SL to ${be_sl:,.2f} (Retcode: {res.retcode if res else 'None'} - {res.comment if res else ''})"
                results.append(msg)
                if verbose: print(f"  {msg}")
        else:
            msg = f"[BREAKEVEN OPTIMAL] Ticket #{p.ticket} ({symbol}) is already at breakeven (${p.sl:,.2f})."
            results.append(msg)
            if verbose: print(f"  {msg}")
            
    return results


def auto_adjust_all_open_positions(verbose=True, force_recalculate=False):
    """
    Real-Time Quantitative SL/TP Manager:
    Scans all open MT5 positions and dynamically sets or adjusts SL/TP based on real-time market conditions
    (ATR volatility, profit protection breakeven locks, and trailing stops).
    """
    if mt5 is None:
        return ["[ERROR] MT5 module not installed."]
    if not mt5.initialize():
        return ["[ERROR] Could not initialize MT5 connection."]
        
    positions = mt5.positions_get()
    if not positions or len(positions) == 0:
        return ["No open positions currently active in MT5."]
        
    results = []
    for p in positions:
        symbol = p.symbol
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            continue
        digits = symbol_info.digits
        
        # Calculate recent ATR for volatility-based stops
        try:
            profile = config.get_symbol_profile(symbol)
            atr = profile["default_atr"]
        except (KeyError, Exception):
            atr = 4.50 if "XAU" in symbol or "GOLD" in symbol else 0.0050
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 20)
            if rates is not None and len(rates) >= 10:
                import numpy as np
                atr_calc = float(np.mean(rates['high'] - rates['low']))
                if atr_calc > 0:
                    atr = atr_calc
        except Exception:
            pass
            
        new_sl = float(p.sl)
        new_tp = float(p.tp)
        reason = ""
        
        # Determine optimal quantitative SL/TP
        if p.type == mt5.ORDER_TYPE_BUY:  # BUY / LONG
            optimal_sl = round(p.price_open - (1.5 * atr), digits)
            optimal_tp = round(p.price_open + (2.5 * atr), digits)
            
            if new_sl == 0.0 or force_recalculate:
                new_sl = optimal_sl
                reason += f"Quantitative SL set (-1.5 ATR @ ${atr:.2f}); "
            if new_tp == 0.0 or force_recalculate:
                new_tp = optimal_tp
                reason += f"Quantitative TP set (+2.5 ATR); "
                
            # Dynamic Breakeven & Trailing Stop protection (DISABLED per user request)
            # profit_points = p.price_current - p.price_open
            # if profit_points >= (1.0 * atr) or (profit_points >= 2.0 and "XAU" in symbol):
            #     be_sl = round(p.price_open + (0.1 * atr if atr < 2.0 else be_buffer), digits)
            #     if new_sl < be_sl:
            #         new_sl = be_sl
            #         reason += "Moved SL to Breakeven Lock (+0.20 pt); "
            # if profit_points >= (2.0 * atr):
            #     trail_sl = round(p.price_current - (1.0 * atr), digits)
            #     if new_sl < trail_sl:
            #         new_sl = trail_sl
            #         reason += "Trailing Stop locked (+1.0 ATR); "
                    
        elif p.type == mt5.ORDER_TYPE_SELL:  # SELL / SHORT
            optimal_sl = round(p.price_open + (1.5 * atr), digits)
            optimal_tp = round(p.price_open - (2.5 * atr), digits)
            
            if new_sl == 0.0 or force_recalculate:
                new_sl = optimal_sl
                reason += f"Quantitative SL set (+1.5 ATR @ ${atr:.2f}); "
            if new_tp == 0.0 or force_recalculate:
                new_tp = optimal_tp
                reason += f"Quantitative TP set (-2.5 ATR); "
                
            # Dynamic Breakeven & Trailing Stop protection (DISABLED per user request)
            # profit_points = p.price_open - p.price_current
            # if profit_points >= (1.0 * atr) or (profit_points >= 2.0 and "XAU" in symbol):
            #     be_sl = round(p.price_open - (0.1 * atr if atr < 2.0 else 0.20), digits)
            #     if new_sl == 0.0 or new_sl > be_sl:
            #         new_sl = be_sl
            #         reason += "Moved SL to Breakeven Lock (Protecting Profit); "
            # if profit_points >= (2.0 * atr):
            #     trail_sl = round(p.price_current + (1.0 * atr), digits)
            #     if new_sl == 0.0 or new_sl > trail_sl:
            #         new_sl = trail_sl
            #         reason += "Trailing Stop locked; "
                    
        # Check if an update is needed (Hysteresis to prevent rapid changing)
        min_delta = 0.15 * atr
        needs_update = False
        
        if p.sl == 0.0 or p.tp == 0.0:
            needs_update = True  # Initial SL/TP must be set immediately
        elif abs(new_sl - float(p.sl)) > min_delta or abs(new_tp - float(p.tp)) > min_delta:
            needs_update = True  # Only update if the change is significant (confirmation buffer)
            
        if needs_update:
            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "symbol": symbol,
                "sl": new_sl,
                "tp": new_tp,
                "magic": 20260727
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                msg = f"[AUTO-SL/TP] Ticket #{p.ticket} ({'BUY' if p.type==0 else 'SELL'} {symbol}): Updated SL -> ${new_sl:,.2f} | TP -> ${new_tp:,.2f} | {reason.strip(' ;')} (PnL: ${p.profit:+.2f})"
                results.append(msg)
                if verbose: print(f"  {msg}")
            else:
                msg = f"[FAILED] Ticket #{p.ticket}: Could not update SL/TP (Retcode: {res.retcode if res else 'None'} - {res.comment if res else ''})"
                results.append(msg)
                if verbose: print(f"  {msg}")
        else:
            msg = f"[OPTIMAL] Ticket #{p.ticket} ({symbol}): Current SL (${p.sl:,.2f}) and TP (${p.tp:,.2f}) are quantitatively optimal (PnL: ${p.profit:+.2f})."
            results.append(msg)
            if verbose: print(f"  {msg}")
            
    return results


def execute_agentic_actions(user_message, llm_response, context):
    """
    Parses agentic commands from the LLM or direct natural language instructions from the user
    and executes live actions on MT5.
    """
    exec_reports = []
    text_to_check = (user_message + " " + llm_response).lower()
    
    # Check for breakeven adjustment
    if "breakeven" in text_to_check or "break even" in text_to_check or "entry price" in text_to_check or "move_breakeven" in text_to_check or "sl to entry" in text_to_check or "stop to entry" in text_to_check or "move sl to breakeven" in text_to_check or "move stop to breakeven" in text_to_check or "move sl" in text_to_check or "move stop" in text_to_check:
        exec_reports.extend(move_all_to_breakeven(verbose=True))
    
    # Check for automatic SL/TP adjustment request
    elif "auto_adjust_sltp" in text_to_check or "set tp and sl automatically" in text_to_check or "set sl/tp" in text_to_check or "adjust my position" in text_to_check or "adjust position" in text_to_check or "set sl and tp" in text_to_check or "set tp and sl" in text_to_check or "protect profit" in text_to_check or "auto adjust" in text_to_check or "set sl" in text_to_check or "set tp" in text_to_check:
        exec_reports.extend(auto_adjust_all_open_positions(verbose=True))
        
    # Check for closing specific ticket
    import re
    close_matches = re.findall(r"close[\s_]*(?:position|trade|ticket)?[\s_#:=]*(\d{6,})", text_to_check)
    if close_matches and mt5 is not None and mt5.initialize():
        for t_str in set(close_matches):
            try:
                t_int = int(t_str)
                pos = mt5.positions_get(ticket=t_int)
                if pos:
                    p = pos[0]
                    tick = mt5.symbol_info_tick(p.symbol)
                    price_close = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
                    req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": p.ticket,
                        "symbol": p.symbol,
                        "volume": p.volume,
                        "type": mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                        "price": price_close,
                        "deviation": 20,
                        "magic": 20260727,
                        "comment": "AI Quant Chatbot Close"
                    }
                    res = mt5.order_send(req)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        exec_reports.append(f"[AGENTIC CLOSE] Closed Ticket #{t_int} ({p.symbol}) at market price ${price_close:,.2f} | Realized PnL: ${p.profit:+.2f}")
                    else:
                        exec_reports.append(f"[FAILED CLOSE] Could not close Ticket #{t_int} (Retcode: {res.retcode if res else 'None'})")
            except Exception as e:
                exec_reports.append(f"[ERROR] Error closing #{t_str}: {str(e)}")
                
    # Check for close all
    if "close all" in text_to_check or "close_all" in text_to_check:
        if mt5 is not None and mt5.initialize():
            positions = mt5.positions_get()
            if positions:
                for p in positions:
                    tick = mt5.symbol_info_tick(p.symbol)
                    price_close = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
                    req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": p.ticket,
                        "symbol": p.symbol,
                        "volume": p.volume,
                        "type": mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                        "price": price_close,
                        "deviation": 20,
                        "magic": 20260727,
                        "comment": "AI Quant Close All"
                    }
                    res = mt5.order_send(req)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        exec_reports.append(f"[AGENTIC CLOSE ALL] Closed Ticket #{p.ticket} ({p.symbol}) | Realized PnL: ${p.profit:+.2f}")
                        
    return exec_reports


def chat_with_quant(user_message, chat_history=[], api_key=None, model_name=None, api_base_url=None):
    """
    Interactive conversational endpoint for the dashboard chatbot with Agentic Action execution.
    """
    context = get_market_context_summary()
    
    system_prompt = (
        "You are ANTIGRAVITY AI QUANT (Agentic Model GLM-5.2 / GPT-OSS), an autonomous institutional quantitative coding and trading assistant "
        "with direct execution access to MT5.\n"
        "You have full visibility into the user's real-time MT5 trading system, quantitative features (Fibonacci OTEs, "
        "MACD momentum, FLOD clusters, killzones), open positions, and closed trade history.\n"
        f"Current System State & Open Positions:\n{json.dumps(context, indent=2)}\n\n"
        "CRITICAL AGENTIC EXECUTION RULES:\n"
        "1. When the user instructs you to perform an action on their open positions (such as moving SL to breakeven, setting SL/TP, protecting profits, closing positions), YOU MUST NOT just respond with Python code examples! You have live execution access! You MUST emit the appropriate ACTION TAG on its own line so the trading engine executes the command immediately in MT5:\n"
        "- To move all open positions Stop Loss to Breakeven (entry price + buffer): `[ACTION: MOVE_BREAKEVEN]`\n"
        "- To automatically calculate and apply optimal quantitative SL/TP to ALL open positions: `[ACTION: AUTO_ADJUST_SLTP]`\n"
        "- To close a specific position by ticket: `[ACTION: CLOSE_POSITION | ticket=12345]`\n"
        "- To close all open positions immediately: `[ACTION: CLOSE_ALL]`\n\n"
        "2. When explaining quantitative concepts, formulas, or price levels, use LaTeX math formatting (e.g., `$\\text{SL} = 4101.30$` or `$$\\Delta P = \\text{Entry} - \\text{Current}$$`) for clear institutional presentation.\n"
        "3. Answer clearly, concisely, and with institutional authority. Always emit the required action tag when an action is requested."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in chat_history[-6:]:  # Keep last 6 turns of history
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
    messages.append({"role": "user", "content": user_message})
    
    res = call_llm_api(messages, api_key=api_key, model_name=model_name, api_base_url=api_base_url, temperature=0.4)
    
    if res.get("success"):
        ai_text = res.get("content", "")
        # Execute agentic actions based on user intent and AI output
        exec_reports = execute_agentic_actions(user_message, ai_text, context)
        if exec_reports:
            report_block = "\n\n==================================================\n"
            report_block += "AGENTIC EXECUTION REPORT (LIVE MT5 ACCOUNT)\n"
            report_block += "==================================================\n"
            for rep in exec_reports:
                report_block += f"{rep}\n"
            res["content"] = ai_text + report_block
            
    return res


def validate_trade_signal(signal_dir, prob, meta_prob, current_price, sl, tp, features_dict, symbol="XAUUSD"):
    """
    Real-Time Institutional Consensus Loop:
    When the local XGBoost/LightGBM Primary Ensemble and Random Forest Meta-Model propose a trade signal,
    send all quantitative feature data to the NVIDIA LLM (gpt-oss-120b / GLM-5.2) for live validation!
    Returns JSON with approval status and optimized SL/TP coordinates.
    """
    system_prompt = (
        "You are ANTIGRAVITY INSTITUTIONAL RISK GATE, a real-time quantitative validation assistant running on NVIDIA GPU inference. "
        "Your task is to validate or reject a live trade signal proposed by local XGBoost / Random Forest meta-models based on real-time feature confluence.\n\n"
        "YOU MUST OUTPUT ONLY A VALID JSON OBJECT exactly in this structure (no extra markdown commentary outside JSON):\n"
        "```json\n"
        "{\n"
        '  "approved": true,\n'
        '  "confidence_score": 0.85,\n'
        '  "reason": "4H Fibonacci OTE zone alignment and VIX safe-haven sentiment confirm institutional long bias.",\n'
        '  "optimized_sl": 2345.50,\n'
        '  "optimized_tp": 2380.00\n'
        "}\n"
        "```\n"
        "If the setup has conflicting macro risk (e.g. VIX > 28 on a counter-trend trade or conflicting 4H MACD momentum), set `\"approved\": false`."
    )
    
    user_prompt = (
        f"Proposed Trade Signal for {symbol}:\n"
        f"  Direction: {'LONG (+1)' if signal_dir == 1 else 'SHORT (-1)' if signal_dir == -1 else 'FLAT (0)'}\n"
        f"  Primary Ensemble Probability: {prob:.3f}\n"
        f"  Secondary Meta-Model Confidence: {meta_prob:.3f}\n"
        f"  Entry Price: {current_price:,.6f} | Initial SL: {sl:,.6f} | Initial TP: {tp:,.6f}\n\n"
        f"Live Quantitative Feature Matrix:\n{json.dumps(features_dict, indent=2)}\n\n"
        f"Validate this institutional setup now."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    res = call_llm_api(messages, temperature=0.2)
    
    if "error" in res:
        return {"approved": True, "reason": f"LLM Validation fallback (API error: {res['error']})", "optimized_sl": sl, "optimized_tp": tp}
        
    content = res.get("content", "")
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "{" in content and "}" in content:
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            json_str = content[start_idx:end_idx+1]
        else:
            json_str = content.strip()
            
        parsed = json.loads(json_str)
        
        opt_sl = float(parsed.get("optimized_sl", sl))
        opt_tp = float(parsed.get("optimized_tp", tp))
        
        # Guard against LLM hallucinating SL/TP on the wrong side of price
        if signal_dir == 1.0 or signal_dir == 1:  # LONG
            if opt_sl >= current_price:
                opt_sl = sl
            if opt_tp <= current_price:
                opt_tp = tp
        elif signal_dir == -1.0 or signal_dir == -1:  # SHORT
            if opt_sl <= current_price:
                opt_sl = sl
            if opt_tp >= current_price:
                opt_tp = tp
                
        return {
            "approved": bool(parsed.get("approved", True)),
            "reason": str(parsed.get("reason", "LLM validated setup.")),
            "optimized_sl": opt_sl,
            "optimized_tp": opt_tp,
            "raw_response": content
        }
    except Exception as e:
        return {"approved": True, "reason": f"LLM Validation fallback (JSON parse note: {str(e)})", "optimized_sl": sl, "optimized_tp": tp}
