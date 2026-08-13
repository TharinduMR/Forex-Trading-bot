"""
Unified Data Loader for Gold (XAUUSD / GC=F) and Forex data.
Supports Yahoo Finance (yfinance) for offline prototyping/testing and MetaTrader 5 (MT5) for live broker feeds.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import config

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


class BaseDataLoader:
    """Abstract base class for data loaders."""
    def fetch_historical(self, symbol, timeframe, period=None):
        raise NotImplementedError
        
    def fetch_latest_buffer(self, symbol, timeframe, buffer_size=config.BUFFER_SIZE):
        raise NotImplementedError
        
    def _standardize_columns(self, df):
        """Ensure consistent column names: open, high, low, close, volume, tick_volume."""
        df.columns = [str(c).lower() for c in df.columns]
        
        # Handle renaming
        rename_map = {}
        for col in df.columns:
            if 'open' in col: rename_map[col] = 'open'
            elif 'high' in col: rename_map[col] = 'high'
            elif 'low' in col: rename_map[col] = 'low'
            elif 'close' in col: rename_map[col] = 'close'
            elif 'adj close' in col: continue
            elif 'volume' in col and 'tick' not in col: rename_map[col] = 'volume'
            elif 'tick' in col: rename_map[col] = 'tick_volume'
            
        df = df.rename(columns=rename_map)
        
        # Keep only essential columns
        required_cols = ['open', 'high', 'low', 'close']
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in downloaded data. Found: {df.columns.tolist()}")
                
        # Ensure volume and tick_volume exist
        if 'volume' not in df.columns and 'tick_volume' in df.columns:
            df['volume'] = df['tick_volume']
        elif 'volume' in df.columns and 'tick_volume' not in df.columns:
            df['tick_volume'] = df['volume']
        elif 'volume' not in df.columns and 'tick_volume' not in df.columns:
            df['volume'] = 0.0
            df['tick_volume'] = 0.0
            
        df = df[['open', 'high', 'low', 'close', 'volume', 'tick_volume']]
        
        # Ensure numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Remove any rows with NaN in OHLC
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Ensure index is timezone-naive datetime and sorted
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
            
        return df


class YFinanceDataLoader(BaseDataLoader):
    """Data loader using Yahoo Finance API."""
    def __init__(self):
        if yf is None:
            raise ImportError("yfinance library is not installed. Run: pip install yfinance")
            
    def fetch_historical(self, symbol=config.SYMBOL_YF, timeframe=config.TIMEFRAME_LTF, period=None):
        if period is None:
            if timeframe in ["1m", "2m", "5m", "15m", "30m", "90m"]:
                period = config.HISTORICAL_PERIOD_YF  # Max 60d for intraday in yfinance
            else:
                period = "730d"

        print(f"[YFinance] Downloading historical data for {symbol} ({timeframe}, period={period})...")
        ticker = yf.Ticker(symbol)

        attempts = []
        if timeframe in ["1m", "2m", "5m", "15m", "30m", "60m", "1h"]:
            attempts.append((period, None))
            attempts.append(("60d", None))
            attempts.append(("90d", None))
        else:
            attempts.append((period, None))

        for attempt_period, _ in attempts:
            try:
                df = ticker.history(period=attempt_period, interval=timeframe, auto_adjust=True)
                if df is None or df.empty:
                    df = yf.download(symbol, period=attempt_period, interval=timeframe, progress=False, auto_adjust=True, threads=False)
            except Exception:
                df = None

            if df is not None and not df.empty:
                break

        if df is None or df.empty:
            try:
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=90)
                df = yf.download(symbol, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), interval=timeframe, progress=False, auto_adjust=True, threads=False)
            except Exception:
                df = None

        if df is None or df.empty:
            raise RuntimeError(f"Failed to download data from Yahoo Finance for symbol={symbol}, interval={timeframe}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = self._standardize_columns(df)
        print(f"[YFinance] Successfully loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}.")
        return df
        
    def fetch_latest_buffer(self, symbol=config.SYMBOL_YF, timeframe=config.TIMEFRAME_LTF, buffer_size=config.BUFFER_SIZE):
        """Fetch the latest buffer of candles for real-time inference."""
        period = "60d" if timeframe in ["1m", "2m", "5m", "15m", "30m", "60m", "1h"] else "730d"
        df = self.fetch_historical(symbol, timeframe, period=period)
        if len(df) > buffer_size:
            df = df.iloc[-buffer_size:]
        return df


class MT5DataLoader(BaseDataLoader):
    """Data loader using MetaTrader 5 terminal integration."""
    def __init__(self):
        if mt5 is None:
            raise ImportError("MetaTrader5 package is not installed.")
        with config.MT5_LOCK:
            if not mt5.initialize():
                err = mt5.last_error()
                raise RuntimeError(f"MT5 initialize() failed with error code: {err}. Please ensure MT5 terminal is open.")
            
    def _map_timeframe(self, timeframe_str):
        mapping = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }
        if timeframe_str not in mapping:
            raise ValueError(f"Unsupported MT5 timeframe: '{timeframe_str}'. Valid: {list(mapping.keys())}")
        return mapping[timeframe_str]
        
    def fetch_historical(self, symbol=config.SYMBOL_MT5, timeframe=config.TIMEFRAME_LTF, period=None):
        tf_const = self._map_timeframe(timeframe)
        # Fetch up to 10,000 bars by default for historical training
        count = 5000 if period is None else 10000
        print(f"[MT5] Requesting {count} historical bars for {symbol} ({timeframe})...")
        
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            raise RuntimeError(f"MT5 copy_rates_from_pos failed for {symbol}: error {err}")
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        
        df = self._standardize_columns(df)
        print(f"[MT5] Successfully loaded {len(df)} bars.")
        return df
        
    def fetch_latest_buffer(self, symbol=config.SYMBOL_MT5, timeframe=config.TIMEFRAME_LTF, buffer_size=config.BUFFER_SIZE):
        tf_const = self._map_timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, buffer_size + 10)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            raise RuntimeError(f"MT5 failed to fetch live buffer for {symbol}: error {err}")
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        df = self._standardize_columns(df)
        if len(df) > buffer_size:
            df = df.iloc[-buffer_size:]
        return df
        
    def __del__(self):
        if mt5 is not None:
            mt5.shutdown()


def get_data_loader(source=config.DEFAULT_DATA_SOURCE):
    """Factory function to get the appropriate data loader."""
    source = source.lower()
    if source == "mt5":
        try:
            return MT5DataLoader()
        except Exception as e:
            warnings.warn(f"Failed to initialize MT5DataLoader ({e}). Falling back to YFinanceDataLoader.")
            return YFinanceDataLoader()
    elif source == "yfinance":
        return YFinanceDataLoader()
    else:
        raise ValueError(f"Unknown data source: '{source}'. Choose 'yfinance' or 'mt5'.")
