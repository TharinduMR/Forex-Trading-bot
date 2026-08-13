"""
Event-Driven Backtesting and Performance Evaluation Engine for Multi-Symbol Trading.
Supports Gold (XAUUSD / GC=F) and Forex pairs (EURUSD).
Simulates realistic trading conditions including spread, slippage, commission, ATR position sizing,
and reports institutional-grade risk-adjusted metrics (Sharpe, Sortino, Max Drawdown, Profit Factor).
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import config


class Backtester:
    def __init__(self, 
                 initial_capital=10000.0,
                 contract_size=100.0,      # Standard Gold contract = 100 oz; EURUSD = 100,000 units
                 spread_dollar=0.25,       # Typical Gold spread: $0.25 per oz; EURUSD: 0.00010 (1 pip)
                 slippage_dollar=0.05,     # Typical execution slippage
                 commission_per_lot=5.0,   # Round-turn commission per 1.0 standard lot
                 risk_per_trade_pct=0.01,  # 1% risk per trade
                 symbol_name="XAUUSD"):    # Display name for charts and reports
        self.initial_capital = initial_capital
        self.contract_size = contract_size
        self.spread_dollar = spread_dollar
        self.slippage_dollar = slippage_dollar
        self.commission_per_lot = commission_per_lot
        self.risk_per_trade_pct = risk_per_trade_pct
        self.symbol_name = symbol_name
        
    def run_simulation(self, df, preds, probs, use_regime_filter=True, meta_probs=None):
        """
        Run event-driven backtest across historical test dataframe.
        Supports Secondary AI Decision Maker (Meta-Labeling) via meta_probs.
        """
        print(f"\n==================================================")
        print(f"RUNNING EVENT-DRIVEN BACKTEST")
        print(f"==================================================")
        print(f"[Parameters] Initial Capital: ${self.initial_capital:,.2f} | Spread: ${self.spread_dollar} | Comm: ${self.commission_per_lot}/lot")
        if meta_probs is not None:
            print(f"[MetaLabeling] Secondary AI Decision Maker Enabled | Min Confidence: {config.META_CONFIDENCE_THRESHOLD*100:.1f}%")
        
        n = len(df)
        open_p = df['open'].values
        high_p = df['high'].values
        low_p = df['low'].values
        close_p = df['close'].values
        times = df.index
        
        # Volatility for position sizing and barriers
        from labeling import compute_volatility
        vol = compute_volatility(df['close']).values
        sigma_dollar = close_p * vol
        
        # Regime filter: check if market is in strong trend (e.g., ADX proxy or trend_direction != 0)
        trend_dir = df['trend_direction'].values if 'trend_direction' in df.columns else np.ones(n)
        
        equity = self.initial_capital
        equity_curve = [equity]
        trade_history = []
        
        # Active position state
        position = 0      # 1 for Long, -1 for Short, 0 for Flat
        entry_price = 0.0
        entry_time = None
        entry_bar = 0
        lot_size = 0.0
        tp_price = 0.0
        sl_price = 0.0
        risk_dollar = 0.0
        
        for i in range(n - 1):
            current_time = times[i]
            current_high = high_p[i]
            current_low = low_p[i]
            current_close = close_p[i]
            
            # 1. Manage existing open position
            if position != 0:
                bars_held = i - entry_bar
                exit_reason = None
                exit_price = 0.0
                
                if position == 1:  # LONG POSITION
                    # Check Stop Loss first (conservative) and account for gap-downs at Open
                    if current_low <= sl_price:
                        exit_reason = "STOP_LOSS"
                        fill_price = min(sl_price, open_p[i])
                        exit_price = fill_price - self.slippage_dollar
                    elif current_high >= tp_price:
                        exit_reason = "TAKE_PROFIT"
                        fill_price = max(tp_price, open_p[i])
                        exit_price = fill_price - self.slippage_dollar
                    elif bars_held >= config.MAX_HOLDING:
                        exit_reason = "TIME_STOP"
                        exit_price = current_close - self.slippage_dollar - (self.spread_dollar / 2.0)
                        
                    if exit_reason:
                        gross_pnl = (exit_price - entry_price) * (lot_size * self.contract_size)
                        comm = self.commission_per_lot * lot_size
                        net_pnl = gross_pnl - comm
                        equity += net_pnl
                        
                        trade_history.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'type': 'LONG',
                            'entry': entry_price,
                            'exit': exit_price,
                            'lots': lot_size,
                            'bars_held': bars_held,
                            'reason': exit_reason,
                            'pnl': net_pnl,
                            'r_multiple': net_pnl / (risk_dollar if risk_dollar > 0 else 1.0),
                            'regime': 'TREND' if trend_dir[entry_bar] != 0 else 'RANGE',
                            'equity': equity
                        })
                        position = 0
                        
                elif position == -1:  # SHORT POSITION
                    # Check Stop Loss first and account for gap-ups at Open
                    if current_high >= sl_price:
                        exit_reason = "STOP_LOSS"
                        fill_price = max(sl_price, open_p[i])
                        exit_price = fill_price + self.slippage_dollar
                    elif current_low <= tp_price:
                        exit_reason = "TAKE_PROFIT"
                        fill_price = min(tp_price, open_p[i])
                        exit_price = fill_price + self.slippage_dollar
                    elif bars_held >= config.MAX_HOLDING:
                        exit_reason = "TIME_STOP"
                        exit_price = current_close + self.slippage_dollar + (self.spread_dollar / 2.0)
                        
                    if exit_reason:
                        gross_pnl = (entry_price - exit_price) * (lot_size * self.contract_size)
                        comm = self.commission_per_lot * lot_size
                        net_pnl = gross_pnl - comm
                        equity += net_pnl
                        
                        trade_history.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'type': 'SHORT',
                            'entry': entry_price,
                            'exit': exit_price,
                            'lots': lot_size,
                            'bars_held': bars_held,
                            'reason': exit_reason,
                            'pnl': net_pnl,
                            'r_multiple': net_pnl / (risk_dollar if risk_dollar > 0 else 1.0),
                            'regime': 'TREND' if trend_dir[entry_bar] != 0 else 'RANGE',
                            'equity': equity
                        })
                        position = 0
                        
            # 2. Check for new entry signal if flat
            if position == 0 and i < n - config.MAX_HOLDING:
                signal = preds[i]
                prob = probs[i] if probs is not None else 1.0
                meta_prob = meta_probs[i] if meta_probs is not None else 1.0
                
                # Verify primary confidence threshold and regime filter
                if prob >= config.CONFIDENCE_THRESHOLD and signal in [1.0, -1.0]:
                    # Step 2: Secondary AI Decision Maker (Meta-Labeling) Risk Gate
                    if meta_prob < config.META_CONFIDENCE_THRESHOLD:
                        equity_curve.append(equity)
                        continue
                        
                    # Step 3: Top-Down HTF Directional Bias Gate (LTF must align with HTF)
                    if 'htf_directional_bias' in df.columns:
                        htf_bias_val = df['htf_directional_bias'].iloc[i]
                        if signal == 1.0 and htf_bias_val == -1.0:
                            equity_curve.append(equity)
                            continue
                        if signal == -1.0 and htf_bias_val == 1.0:
                            equity_curve.append(equity)
                            continue
                        
                    if use_regime_filter and 'choch_flag' in df.columns:
                        # Avoid taking trades immediately on the exact bar of a ChoCH reversal noise
                        if df['choch_flag'].iloc[i] == 1.0:
                            equity_curve.append(equity)
                            continue
                            
                    sigma = max(sigma_dollar[i], 0.50)
                    sl_dist = config.SL_MULT * sigma
                    tp_dist = config.TP_MULT * sigma
                    
                    # Dynamic ATR Position Sizing: Risk 1% of equity
                    max_risk_dollar = equity * self.risk_per_trade_pct
                    # Value per lot per oz move = contract_size (e.g. $100 per $1 move)
                    loss_per_lot = (sl_dist + self.spread_dollar + self.slippage_dollar) * self.contract_size + self.commission_per_lot
                    lot_size = max(0.01, round(max_risk_dollar / loss_per_lot, 2))
                    
                    # Dynamic meta position sizing: Scale up lot size if AI Decision Maker is >75% confident
                    if meta_prob >= 0.75:
                        lot_size *= 1.5
                    lot_size = min(round(lot_size, 2), 5.0)  # Cap at 5 lots for safety
                    
                    risk_dollar = loss_per_lot * lot_size
                    
                    if signal == 1.0:  # ENTER LONG
                        position = 1
                        entry_price = open_p[i + 1] + (self.spread_dollar / 2.0) + self.slippage_dollar
                        entry_time = times[i + 1]
                        entry_bar = i + 1
                        tp_price = entry_price + tp_dist
                        sl_price = entry_price - sl_dist
                    elif signal == -1.0:  # ENTER SHORT
                        position = -1
                        entry_price = open_p[i + 1] - (self.spread_dollar / 2.0) - self.slippage_dollar
                        entry_time = times[i + 1]
                        entry_bar = i + 1
                        tp_price = entry_price - tp_dist
                        sl_price = entry_price + sl_dist
                        
            equity_curve.append(equity)
            
        # Ensure equity curve matches dataframe length
        while len(equity_curve) < n:
            equity_curve.append(equity)
            
        df_trades = pd.DataFrame(trade_history)
        self._print_performance_report(equity_curve, df_trades, df.index)
        self._plot_equity_curve(equity_curve, df.index, df_trades)
        
        return equity_curve, df_trades
        
    def _print_performance_report(self, equity_curve, df_trades, dates):
        """Calculate and print institutional quantitative metrics."""
        print(f"\n--- INSTITUTIONAL PERFORMANCE REPORT ---")
        if df_trades.empty:
            print("[Warning] No trades executed during the backtest period!")
            return
            
        total_trades = len(df_trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] < 0]
        
        win_rate = len(wins) / total_trades * 100.0
        gross_profit = wins['pnl'].sum() if not wins.empty else 0.0
        gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        
        net_profit = equity_curve[-1] - self.initial_capital
        total_ret_pct = (net_profit / self.initial_capital) * 100.0
        
        # Annualized metrics
        days = (dates[-1] - dates[0]).days
        years = max(days / 365.25, 0.1)
        cagr = ((equity_curve[-1] / self.initial_capital) ** (1.0 / years) - 1.0) * 100.0
        
        # Drawdown calculation
        eq_s = pd.Series(equity_curve)
        peak = eq_s.cummax()
        dd_dollar = peak - eq_s
        dd_pct = (dd_dollar / peak) * 100.0
        max_dd_dollar = dd_dollar.max()
        max_dd_pct = dd_pct.max()
        
        calmar = cagr / max_dd_pct if max_dd_pct > 0 else cagr
        
        # Sharpe & Sortino (assuming 252 * (6.5 hours * 4 bars/hr) intraday periods per year)
        eq_rets = eq_s.pct_change().dropna()
        if eq_rets.std() > 0:
            bars_per_day = 96
            bars_per_year = 252 * bars_per_day
            sharpe = (eq_rets.mean() / eq_rets.std()) * np.sqrt(bars_per_year)
            downside_rets = eq_rets[eq_rets < 0]
            sortino = (eq_rets.mean() / downside_rets.std()) * np.sqrt(bars_per_year) if len(downside_rets) > 0 and downside_rets.std() > 0 else sharpe
        else:
            sharpe = 0.0
            sortino = 0.0
            
        avg_trade_pnl = df_trades['pnl'].mean()
        avg_r_mult = df_trades['r_multiple'].mean()
        
        print(f"Total Net Profit:     ${net_profit:,.2f} ({total_ret_pct:.2f}%) | CAGR: {cagr:.2f}%")
        print(f"Profit Factor:        {profit_factor:.2f} | Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
        print(f"Sharpe Ratio (Ann.):  {sharpe:.2f} | Sortino Ratio (Ann.): {sortino:.2f}")
        print(f"Max Drawdown:         ${max_dd_dollar:,.2f} ({max_dd_pct:.2f}%) | Calmar Ratio: {calmar:.2f}")
        print(f"Avg Trade Expectancy: ${avg_trade_pnl:,.2f} | Avg R-Multiple: {avg_r_mult:.2f}R")
        print(f"Avg Holding Period:   {df_trades['bars_held'].mean():.1f} bars ({df_trades['bars_held'].mean()*15:.0f} mins)")
        
        # Regime breakdown
        if 'regime' in df_trades.columns:
            print(f"\n--- Performance by Market Regime ---")
            for reg, group in df_trades.groupby('regime'):
                reg_wins = group[group['pnl'] > 0]
                reg_wr = len(reg_wins) / len(group) * 100.0 if len(group) > 0 else 0.0
                reg_pnl = group['pnl'].sum()
                print(f"  {reg:<6} Regime: {len(group)} trades | Win Rate: {reg_wr:.1f}% | Net PnL: ${reg_pnl:,.2f}")
        print(f"--------------------------------------------------\n")
        
    def _plot_equity_curve(self, equity_curve, dates, df_trades):
        """Plot and save institutional backtest performance chart."""
        try:
            plt.figure(figsize=(12, 7))
            
            # Subplot 1: Equity Curve
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(dates, equity_curve, label="Portfolio Equity ($)", color="#1f77b4", linewidth=2)
            ax1.set_title(f"{self.symbol_name} — Event-Driven Backtest Equity", fontsize=14, fontweight='bold')
            ax1.set_ylabel("Equity ($)", fontsize=11)
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper left")
            
            # Mark winning and losing trades
            if not df_trades.empty:
                wins = df_trades[df_trades['pnl'] > 0]
                losses = df_trades[df_trades['pnl'] < 0]
                ax1.scatter(wins['exit_time'], wins['equity'], color="green", marker="^", s=40, label="Win", zorder=5)
                ax1.scatter(losses['exit_time'], losses['equity'], color="red", marker="v", s=40, label="Loss", zorder=5)
                
            # Subplot 2: Drawdown (%)
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            eq_s = pd.Series(equity_curve, index=dates)
            peak = eq_s.cummax()
            dd_pct = ((eq_s - peak) / peak) * 100.0
            
            ax2.fill_between(dates, dd_pct, 0, color="darkred", alpha=0.4, label="Drawdown (%)")
            ax2.set_ylabel("Drawdown (%)", fontsize=11)
            ax2.set_xlabel("Date", fontsize=11)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="lower left")
            
            plt.tight_layout()
            plot_path = os.path.join(config.MODEL_DIR, "backtest_equity.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"[Chart] Saved backtest equity curve to: {plot_path}")
        except Exception as e:
            print(f"[Warning] Could not generate equity plot: {e}")

    def plot_trade_sanity_check(self, df, df_trades, n_bars=150):
        """
        Step 1 Visual Sanity Check: Plot candlestick price action, ICT zones (FVG/OB),
        and actual trade entries/exits to visually verify institutional edge.
        """
        try:
            if df_trades.empty or len(df) < n_bars:
                print("[Warning] Insufficient data or trades for visual sanity check chart.")
                return
                
            # Pick a window containing active trades
            last_trade_time = df_trades['exit_time'].iloc[-1]
            end_idx = df.index.get_loc(last_trade_time) if last_trade_time in df.index else len(df) - 1
            start_idx = max(0, end_idx - n_bars)
            
            sub_df = df.iloc[start_idx:end_idx+1]
            sub_dates = sub_df.index
            
            plt.figure(figsize=(14, 8))
            ax = plt.subplot(1, 1, 1)
            
            # Plot High/Low bars and Close
            for idx_val, row in sub_df.iterrows():
                color = "green" if row['close'] >= row['open'] else "red"
                ax.plot([idx_val, idx_val], [row['low'], row['high']], color=color, linewidth=1.2, alpha=0.8)
                ax.plot([idx_val, idx_val], [row['open'], row['close']], color=color, linewidth=3.5, alpha=0.9)
                
            # Plot ICT Fair Value Gaps if available
            if 'fvg_active' in sub_df.columns:
                bull_fvg = sub_df[sub_df['fvg_active'] == 1.0]
                bear_fvg = sub_df[sub_df['fvg_active'] == -1.0]
                ax.scatter(bull_fvg.index, bull_fvg['low'] * 0.9995, color="lime", marker="^", s=30, alpha=0.6, label="Bullish FVG")
                ax.scatter(bear_fvg.index, bear_fvg['high'] * 1.0005, color="magenta", marker="v", s=30, alpha=0.6, label="Bearish FVG")
                
            # Filter trades within this window
            sub_trades = df_trades[(df_trades['entry_time'] >= sub_dates[0]) & (df_trades['exit_time'] <= sub_dates[-1])]
            
            for _, tr in sub_trades.iterrows():
                ent_t = tr['entry_time']
                ext_t = tr['exit_time']
                ent_p = tr['entry']
                ext_p = tr['exit']
                pnl = tr['pnl']
                
                # Entry marker
                m_color = "blue" if tr['type'] == 'LONG' else "purple"
                m_marker = "^" if tr['type'] == 'LONG' else "v"
                ax.scatter(ent_t, ent_p, color=m_color, marker=m_marker, s=100, zorder=10, label=f"Entry {tr['type']}" if _ == 0 else "")
                
                # Exit marker
                ex_color = "darkgreen" if pnl > 0 else "darkred"
                ex_marker = "o" if pnl > 0 else "x"
                ax.scatter(ext_t, ext_p, color=ex_color, marker=ex_marker, s=80, zorder=10, label="Exit Win/Loss" if _ == 0 else "")
                
                # Connecting line
                ax.plot([ent_t, ext_t], [ent_p, ext_p], color=ex_color, linestyle="--", linewidth=1.5, alpha=0.8)
                
                # PnL annotation
                ax.annotate(f"${pnl:.0f}", (ext_t, ext_p), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold", color=ex_color)
                
            ax.set_title(f"{self.symbol_name} — Visual Sanity Check: Price Action, ICT Zones, Trade Execution", fontsize=14, fontweight="bold")
            ax.set_ylabel("Price ($)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper left")
            
            plt.tight_layout()
            plot_path = os.path.join(config.MODEL_DIR, "trade_sanity_check.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"[Chart] Saved visual sanity check chart to: {plot_path}")
        except Exception as e:
            print(f"[Warning] Could not generate visual sanity check plot: {e}")
