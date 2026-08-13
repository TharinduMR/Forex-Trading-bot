import MetaTrader5 as mt5
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_macro_data(start_date, end_date):
    """
    Fetches DXY, US10Y, and VIX data.
    Immune to future MT5 timestamps (e.g., year 2026 demo accounts).
    """
    # FIX: Prevent future date errors from MT5 demo accounts
    today_real = datetime.now().strftime('%Y-%m-%d')
    
    # Convert dates to strings if they aren't already
    start_str = str(start_date)
    end_str = str(end_date)
    
    # If the end_date from the dataframe is in the future, cap it at today's real date
    if end_str > today_real:
        end_str = today_real
        
    # If start_date is also in the future, default to 2 years back from today
    if start_str > today_real:
        start_str = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    tickers = {
        'DXY': 'DX-Y.NYB',      # US Dollar Index
        'US10Y': '^TNX',        # 10-Year Treasury Yield
        'VIX': '^VIX'           # Fear Index
    }
    
    try:
        # Download without progress bar to keep console clean
        macro_df = yf.download(list(tickers.values()), start=start_str, end=end_str, progress=False)
        
        # yfinance often returns a MultiIndex column DataFrame when downloading multiple tickers.
        # We need to extract just the 'Close' prices safely.
        if isinstance(macro_df.columns, pd.MultiIndex):
            if 'Close' in macro_df.columns.get_level_values(0):
                macro_df = macro_df['Close']
        elif 'Close' in macro_df.columns:
            macro_df = macro_df[['Close']]
            macro_df.columns = list(tickers.keys())
            
        # Safe rename mapping (yfinance uses the tickers as column names)
        rename_map = {v: k for k, v in tickers.items()}
        macro_df = macro_df.rename(columns=rename_map)
        
        # Ensure all expected columns exist
        for col in tickers.keys():
            if col not in macro_df.columns:
                macro_df[col] = np.nan
                
        return macro_df.ffill().dropna()
    except Exception as e:
        print(f"[MacroEngine] Failed to fetch macro data: {e}")
        return pd.DataFrame()


def calculate_macro_features(df, macro_df=None):
    """
    Merges macro data into the LTF (15m/1H) dataframe.
    Creates Risk-On/Risk-Off and Correlation features.
    """
    if macro_df is None or macro_df.empty:
        # Generate neutral bias if no data is available
        df['vix_regime'] = 0.0
        df['dxy_bias'] = 0.0
        df['us10y_bias'] = 0.0
        df['macro_bias'] = 0.0
        return df
        
    # Resample macro daily data to LTF and forward fill
    # Convert macro_df index to match tz if necessary
    try:
        if df.index.tz is not None:
            if macro_df.index.tz is None:
                # Localize to UTC first, then convert to match df
                macro_df.index = macro_df.index.tz_localize('UTC').tz_convert(df.index.tz)
            else:
                macro_df.index = macro_df.index.tz_convert(df.index.tz)
        elif macro_df.index.tz is not None:
            # If df has no tz, strip it from macro_df
            macro_df.index = macro_df.index.tz_localize(None)
    except Exception as e:
        print(f"[MacroEngine] Timezone alignment warning: {e}")
        
    macro_ltf = macro_df.reindex(df.index, method='ffill')
    
    # Feature 1: VIX Regime (Risk-On vs Risk-Off)
    macro_ltf['vix_regime'] = np.where(macro_ltf['VIX'] > 30, 1.0,   # Extreme Fear (Bullish for Gold)
                              np.where(macro_ltf['VIX'] < 15, -1.0, 0.0)) # Complacency (Bearish for Gold)
    
    # Feature 2: Macro Trend Alignment (DXY vs Gold)
    macro_ltf['dxy_returns'] = macro_ltf['DXY'].pct_change()
    macro_ltf['dxy_bias'] = np.where(macro_ltf['dxy_returns'] > 0, -1.0, 1.0) # DXY up = Gold down
    
    # Feature 3: US10Y Momentum (Interest Rates)
    macro_ltf['us10y_slope'] = macro_ltf['US10Y'].rolling(5).mean().diff()
    macro_ltf['us10y_bias'] = np.where(macro_ltf['us10y_slope'] > 0, -1.0, 1.0) # Yields up = Gold down
    
    # Combine into one Master Macro Bias Score
    macro_ltf['macro_bias'] = (
        macro_ltf['vix_regime'] * 1.5 + 
        macro_ltf['dxy_bias'] * 1.0 + 
        macro_ltf['us10y_bias'] * 1.0
    )
    
    # Merge into main dataframe
    for col in ['macro_bias', 'vix_regime', 'dxy_bias', 'us10y_bias']:
        df[col] = macro_ltf[col].fillna(0.0)
        
    return df