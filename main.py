"""
Master CLI Orchestrator for Real-Time Multi-Symbol Market Analyzing & Prediction Engine.
Supports Gold (XAUUSD / GC=F) and Forex pairs (EURUSD).
Usage:
  python main.py --mode train --source yfinance --symbol GC=F
  python main.py --mode train --source yfinance --symbol EURUSD=X
  python main.py --mode backtest --source yfinance --symbol GC=F
  python main.py --mode backtest --source yfinance --symbol EURUSD=X
  python main.py --mode live --source yfinance --symbol GC=F --live-iterations 3
  python main.py --mode live --source mt5 --symbol EURUSD --live-iterations 3
  python main.py --mode live-all  (runs live engines for ALL configured symbols)
  python main.py --mode all --source yfinance --live-iterations 2
"""
import argparse
import sys
import os
import joblib
import numpy as np
import pandas as pd
import config
from config import get_symbol_profile, get_model_paths, get_ticker_for_source, resolve_symbol
from data_loader import get_data_loader
from features import engineer_all_features, get_feature_column_names
from labeling import apply_triple_barrier, create_smc_regression_target, create_ict_regression_target
from train import train_pipeline, train_smc_model, train_ict_model
from backtester import Backtester
from live_engine import LivePredictionEngine


def run_training_mode(source, symbol, timeframe, use_optuna, n_trials):
    profile = get_symbol_profile(symbol)
    ticker = get_ticker_for_source(symbol, source)
    model_paths = get_model_paths(symbol, timeframe)
    
    print(f"\n[Main] Initializing Training Pipeline | Symbol: {profile['display_name']} | Ticker: {ticker} | Source: {source} | TF: {timeframe}")
    loader = get_data_loader(source)
    
    # 1. Download historical data
    df = loader.fetch_historical(ticker, timeframe)
    
    # Optional: fetch HTF data if available
    df_htf1 = None
    df_htf2 = None
    htf1_tf, htf2_tf = config.get_htf_timeframes(timeframe)
    try:
        df_htf1 = loader.fetch_historical(ticker, htf1_tf)
        df_htf2 = loader.fetch_historical(ticker, htf2_tf)
    except Exception as e:
        print(f"[Main] Note: HTF data loading skipped ({e})")
        
    # 2. Engineer Lookahead-Safe Features
    print(f"[Main] Engineering quantitative ICT and Order Flow features...")
    df_feat = engineer_all_features(df, df_htf1, df_htf2)
    
    # 3. Apply Triple-Barrier Target Labeling
    print(f"[Main] Applying López de Prado Triple-Barrier labeling (TP={config.TP_MULT}x, SL={config.SL_MULT}x, Time={config.MAX_HOLDING} bars)...")
    df_labeled = apply_triple_barrier(df_feat)
    
    # 4. Extract ML feature names
    feature_cols = get_feature_column_names(df_labeled)
    print(f"[Main] Extracted {len(feature_cols)} quantitative feature columns.")
    
    # 5. Execute Purged Walk-Forward Training & Ensemble Creation (per-symbol model paths)
    xgb_m, lgb_m, best_params = train_pipeline(df_labeled, feature_cols, target_col='target',
                                                 use_optuna=use_optuna, n_trials=n_trials, symbol=symbol, timeframe=timeframe)
    print(f"\n[Main] Training complete for {profile['display_name']} ({timeframe})! Models saved in: {config.MODEL_DIR}")
    return df_labeled, feature_cols


def run_smc_training_mode(source, symbol, timeframe):
    """Train the SMC Specialist Regression Model on event-filtered data."""
    profile = get_symbol_profile(symbol)
    ticker = get_ticker_for_source(symbol, source)
    
    print(f"\n[Main] Initializing SMC Event-Based Training Pipeline | Symbol: {profile['display_name']} | Ticker: {ticker} | Source: {source} | TF: {timeframe}")
    loader = get_data_loader(source)
    
    # 1. Download historical data
    df = loader.fetch_historical(ticker, timeframe)
    
    # Optional: fetch HTF data for confluence
    df_htf1 = None
    df_htf2 = None
    htf1_tf, htf2_tf = config.get_htf_timeframes(timeframe)
    try:
        df_htf1 = loader.fetch_historical(ticker, htf1_tf)
        df_htf2 = loader.fetch_historical(ticker, htf2_tf)
    except Exception as e:
        print(f"[Main] Note: HTF data loading skipped ({e})")
        
    # 2. Engineer Lookahead-Safe Features (includes detect_smc_event automatically)
    print(f"[Main] Engineering quantitative ICT features + SMC event detection...")
    df_feat = engineer_all_features(df, df_htf1, df_htf2)
    
    # 3. Print SMC Event Statistics
    if 'smc_trigger' in df_feat.columns:
        n_long = int((df_feat['smc_trigger'] == 1.0).sum())
        n_short = int((df_feat['smc_trigger'] == -1.0).sum())
        n_total = n_long + n_short
        print(f"\n[SMC Events] Detected {n_total} SMC confluence events in {len(df_feat)} bars ({n_total/max(len(df_feat),1)*100:.1f}%)")
        print(f"  Long setups:  {n_long}")
        print(f"  Short setups: {n_short}")
        
        # Session distribution analysis
        if n_total > 0 and 'kz_ny_am' in df_feat.columns:
            smc_events = df_feat[df_feat['smc_trigger'] != 0.0]
            ny_am_pct = smc_events['kz_ny_am'].mean() * 100
            london_pct = smc_events.get('kz_london', pd.Series(0.0)).mean() * 100
            print(f"  NY AM session: {ny_am_pct:.1f}% | London session: {london_pct:.1f}%")
    else:
        print(f"[Warning] smc_trigger column not generated. Check features.py detect_smc_event().")
        return
    
    # 4. Apply SMC Regression Target
    print(f"[Main] Applying SMC regression target (horizon={config.SMC_REGRESSION_HORIZON} bars, vol_window={config.SMC_VOL_WINDOW})...")
    df_labeled = create_smc_regression_target(df_feat)
    
    # 5. Extract ML feature names
    feature_cols = get_feature_column_names(df_labeled)
    print(f"[Main] Extracted {len(feature_cols)} feature columns for SMC model.")
    
    # 6. Train the SMC Specialist Model
    smc_model = train_smc_model(df_labeled, feature_cols, symbol=symbol, timeframe=timeframe)
    
    if smc_model is not None:
        print(f"\n[Main] SMC Training complete for {profile['display_name']} ({timeframe})!")
        print(f"[Main] Run '--mode live --symbol {symbol}' to use the SMC gate in live trading.")
    else:
        print(f"\n[Main] SMC Training failed for {profile['display_name']}. See errors above.")
    
    return df_labeled, feature_cols


def run_ict_training_mode(source, symbol, timeframe):
    """Train the ICT Specialist Regression Model on event-filtered data."""
    profile = get_symbol_profile(symbol)
    ticker = get_ticker_for_source(symbol, source)
    
    print(f"\n[Main] Initializing ICT Event-Based Training Pipeline | Symbol: {profile['display_name']} | Ticker: {ticker} | Source: {source} | TF: {timeframe}")
    loader = get_data_loader(source)
    
    # 1. Download historical data
    df = loader.fetch_historical(ticker, timeframe)
    
    # Optional: fetch HTF data for confluence
    df_htf1 = None
    df_htf2 = None
    htf1_tf, htf2_tf = config.get_htf_timeframes(timeframe)
    try:
        df_htf1 = loader.fetch_historical(ticker, htf1_tf)
        df_htf2 = loader.fetch_historical(ticker, htf2_tf)
    except Exception as e:
        print(f"[Main] Note: HTF data loading skipped ({e})")
        
    # 2. Engineer Lookahead-Safe Features (includes detect_ict_event automatically)
    print(f"[Main] Engineering quantitative ICT features + ICT event detection...")
    df_feat = engineer_all_features(df, df_htf1, df_htf2)
    
    # 3. Print ICT Event Statistics
    if 'ict_trigger' in df_feat.columns:
        n_long = int((df_feat['ict_trigger'] == 1.0).sum())
        n_short = int((df_feat['ict_trigger'] == -1.0).sum())
        n_total = n_long + n_short
        print(f"\n[ICT Events] Detected {n_total} ICT confluence events in {len(df_feat)} bars ({n_total/max(len(df_feat),1)*100:.1f}%)")
        print(f"  Long setups:  {n_long}")
        print(f"  Short setups: {n_short}")
        
        # Session distribution analysis
        if n_total > 0 and 'kz_ny_am' in df_feat.columns:
            ict_events = df_feat[df_feat['ict_trigger'] != 0.0]
            ny_am_pct = ict_events['kz_ny_am'].mean() * 100
            london_pct = ict_events.get('kz_london', pd.Series(0.0)).mean() * 100
            print(f"  NY AM session: {ny_am_pct:.1f}% | London session: {london_pct:.1f}%")
    else:
        print(f"[Warning] ict_trigger column not generated. Check features.py detect_ict_event().")
        return
    
    # 4. Apply ICT Regression Target
    print(f"[Main] Applying ICT regression target (horizon={config.ICT_REGRESSION_HORIZON} bars, vol_window={config.ICT_VOL_WINDOW})...")
    df_labeled = create_ict_regression_target(df_feat)
    
    # 5. Extract ML feature names
    feature_cols = get_feature_column_names(df_labeled)
    print(f"[Main] Extracted {len(feature_cols)} feature columns for ICT model.")
    
    # 6. Train the ICT Specialist Model
    ict_model = train_ict_model(df_labeled, feature_cols, symbol=symbol, timeframe=timeframe)
    
    if ict_model is not None:
        print(f"\n[Main] ICT Training complete for {profile['display_name']} ({timeframe})!")
        print(f"[Main] Run '--mode live --symbol {symbol}' to use the ICT gate in live trading.")
    else:
        print(f"\n[Main] ICT Training failed for {profile['display_name']}. See errors above.")
    
    return df_labeled, feature_cols


def run_backtest_mode(source, symbol, timeframe, spread_dollar=None):
    profile = get_symbol_profile(symbol)
    ticker = get_ticker_for_source(symbol, source)
    model_paths = get_model_paths(symbol, timeframe)
    
    if spread_dollar is None:
        spread_dollar = profile["spread_dollar"]
    
    print(f"\n[Main] Initializing Event-Driven Backtest | Symbol: {profile['display_name']} | Ticker: {ticker} | Source: {source} | TF: {timeframe} | Spread: {spread_dollar}")
    loader = get_data_loader(source)
    df = loader.fetch_historical(ticker, timeframe)
    
    df_htf1 = None
    df_htf2 = None
    htf1_tf, htf2_tf = config.get_htf_timeframes(timeframe)
    try:
        df_htf1 = loader.fetch_historical(ticker, htf1_tf)
        df_htf2 = loader.fetch_historical(ticker, htf2_tf)
    except Exception:
        pass
        
    df_feat = engineer_all_features(df, df_htf1, df_htf2)
    
    # Check if Out-of-Sample predictions exist (for honest walk-forward evaluation)
    oos_path = model_paths["oos_preds"]
    if os.path.exists(oos_path):
        print(f"[Main] Loaded Out-of-Sample Walk-Forward Predictions from {oos_path} (0% In-Sample Leakage)")
        oos_df = joblib.load(oos_path)
        # Align predictions to dataframe timestamps
        preds = pd.Series(np.nan, index=df_feat.index)
        probs = pd.Series(np.nan, index=df_feat.index)
        
        common_idx = df_feat.index.intersection(oos_df.index)
        preds.loc[common_idx] = oos_df.loc[common_idx, 'pred']
        probs.loc[common_idx] = oos_df.loc[common_idx, 'prob']
        
        preds = preds.fillna(0.0).values
        probs = probs.fillna(1.0).values
        
        if 'meta_prob' in oos_df.columns:
            meta_series = pd.Series(1.0, index=df_feat.index)
            meta_series.loc[common_idx] = oos_df.loc[common_idx, 'meta_prob']
            meta_probs = meta_series.fillna(1.0).values
        else:
            meta_probs = None
    else:
        print(f"[Main] OOS predictions not found. Falling back to saved model inference...")
        xgb_path = model_paths["xgb"]
        lgb_path = model_paths["lgb"]
        feat_path = model_paths["feature_names"]
        
        if not (os.path.exists(xgb_path) and os.path.exists(lgb_path) and os.path.exists(feat_path)):
            print(f"[Error] Trained models not found for {profile['display_name']}! Please run '--mode train --symbol {symbol}' first.")
            return
            
        xgb_model = joblib.load(xgb_path)
        lgb_model = joblib.load(lgb_path)
        feature_names = joblib.load(feat_path)
        
        for col in feature_names:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
                
        X = df_feat[feature_names].values
        prob_xgb = xgb_model.predict_proba(X)
        prob_lgb = lgb_model.predict_proba(X)
        prob_avg = (prob_xgb + prob_lgb) / 2.0
        
        preds = np.argmax(prob_avg, axis=1) - 1.0
        probs = np.max(prob_avg, axis=1)
        
        meta_path = model_paths["meta"]
        if os.path.exists(meta_path):
            meta_m = joblib.load(meta_path)
            meta_probs = meta_m.predict_proba(X)[:, 1]
        else:
            meta_probs = None
        
    bt = Backtester(
        initial_capital=10000.0,
        contract_size=profile["contract_size"],
        spread_dollar=spread_dollar,
        slippage_dollar=profile["slippage_dollar"],
        commission_per_lot=profile["commission_per_lot"],
        symbol_name=profile["display_name"]
    )
    equity_curve, df_trades = bt.run_simulation(df_feat, preds, probs, use_regime_filter=True, meta_probs=meta_probs)
    bt.plot_trade_sanity_check(df_feat, df_trades)
    return equity_curve, df_trades


def run_live_mode(source, symbol, timeframes, max_iterations=None):
    import threading
    import time
    
    if isinstance(timeframes, str):
        timeframes = [timeframes]
        
    profile = get_symbol_profile(symbol)
    ticker = get_ticker_for_source(symbol, source)
    print(f"\n[Main] Initializing Multi-Timeframe Live Inference Engine for {profile['display_name']}...")
    
    threads = []
    for tf in timeframes:
        def _run_engine(t=tf):
            try:
                engine = LivePredictionEngine(data_source=source, symbol=symbol, timeframe=t)
                engine.run_polling_loop(symbol=ticker, timeframe=t, max_iterations=max_iterations)
            except FileNotFoundError as e:
                print(f"[Main] Skipping Engine for {symbol} ({t}): {e}")
                # We write a dummy state so the dashboard doesn't hang on "WAITING FOR ENGINE..."
                dummy_state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"monitor_state_{symbol}_{t}.json")
                import json, datetime
                try:
                    with open(dummy_state_path, "w") as f:
                        json.dump({
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "close": 0.0,
                            "signal": "NO MODEL TRAINED",
                            "signal_code": 0.0,
                            "probabilities": [0.0, 1.0, 0.0],
                            "last_action": f"Model not found for {t}. Please run train mode first.",
                            "atr": 0.0
                        }, f)
                except Exception: pass
            except Exception as e:
                print(f"[Main] Engine for {symbol} ({t}) encountered error: {e}")
                
        thread = threading.Thread(target=_run_engine, name=f"LiveEngine-{symbol}-{tf}", daemon=True)
        threads.append(thread)
        thread.start()
        time.sleep(1) # Stagger MT5 connections
        
    for thread in threads:
        thread.join()


def run_live_all_mode(source, timeframe, max_iterations=None):
    """Run live engines for all configured symbols using threading."""
    import threading
    
    if source == "mt5":
        import MetaTrader5 as mt5
        with config.MT5_LOCK:
            if not mt5.initialize():
                print("[Main] FATAL: Could not initialize MT5.")
                return
                
    symbols_list = config.ACTIVE_SYMBOLS_MT5 if source == "mt5" else config.ACTIVE_SYMBOLS_YF
    print(f"\n[Main] Launching Multi-Symbol Live Engines for: {symbols_list}")
    
    threads = []
    for sym in symbols_list:
        profile = get_symbol_profile(sym)
        ticker = get_ticker_for_source(sym, source)
        
        def _run_engine(s=sym, t=ticker):
            try:
                engine = LivePredictionEngine(data_source=source, symbol=s)
                engine.run_polling_loop(symbol=t, timeframe=timeframe, max_iterations=max_iterations)
            except Exception as e:
                print(f"[Main] Engine for {s} encountered error: {e}")
        
        thread = threading.Thread(target=_run_engine, name=f"LiveEngine-{sym}", daemon=True)
        threads.append(thread)
        thread.start()
        print(f"[Main] Started live engine thread for {profile['display_name']}")
    
    # Wait for all threads
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n[Main] Multi-symbol live engines stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Multi-Symbol Prediction & Analysis Engine (XAUUSD, EURUSD)")
    parser.add_argument("--mode", type=str, choices=["train", "train-smc", "train-ict", "backtest", "live", "live-all", "all"], default="all", help="Execution mode")
    parser.add_argument("--source", type=str, choices=["yfinance", "mt5"], default=config.DEFAULT_DATA_SOURCE, help="Data source")
    parser.add_argument("--symbol", type=str, default=None, help="Ticker symbol (e.g. GC=F, EURUSD=X, XAUUSD, EURUSD)")
    parser.add_argument("--timeframes", nargs="+", default=[config.TIMEFRAME_LTF], help="List of timeframes (e.g., 15m 1h 4h)")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter optimization during training")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials if --optuna is enabled")
    parser.add_argument("--live-iterations", type=int, default=None, help="Max polling iterations in live mode (defaults to infinite/continuous)")
    parser.add_argument("--spread", type=float, default=None, help="Spread override (symbol-specific default used if not specified)")
    
    args = parser.parse_args()
    
    # Resolve symbol: default based on source if not specified
    symbol = args.symbol
    if symbol is None:
        symbol = config.SYMBOL_MT5 if args.source == "mt5" else config.SYMBOL_YF
    
    # Resolve to canonical key for profile lookup
    canonical = resolve_symbol(symbol)
        
    if args.mode == "live":
        print(f"\n[{canonical}] Initializing LIVE mode for timeframes: {args.timeframes}")
        run_live_mode(args.source, canonical, args.timeframes, max_iterations=args.live_iterations)
        return
        
    for tf in args.timeframes:
        print(f"\n[{canonical} | {tf}] Initializing mode: {args.mode}")
        if args.mode == "train":
            run_training_mode(args.source, canonical, tf, use_optuna=args.optuna, n_trials=args.trials)
        elif args.mode == "train-smc":
            run_smc_training_mode(args.source, canonical, tf)
        elif args.mode == "train-ict":
            run_ict_training_mode(args.source, canonical, tf)
        elif args.mode == "backtest":
            run_backtest_mode(args.source, canonical, tf, spread_dollar=args.spread)
        elif args.mode == "live-all":
            run_live_all_mode(args.source, tf, max_iterations=args.live_iterations)
        elif args.mode == "all":
            print(f"==================================================")
            print(f"RUNNING COMPLETE SYSTEM VERIFICATION FOR {canonical} ({tf})")
            print(f"==================================================")
            run_training_mode(args.source, canonical, tf, use_optuna=args.optuna, n_trials=args.trials)
            run_backtest_mode(args.source, canonical, tf, spread_dollar=args.spread)
            live_iters = 3 if args.live_iterations is None else args.live_iterations
            run_live_mode(args.source, canonical, [tf], max_iterations=live_iters)


if __name__ == "__main__":
    main()
