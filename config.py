"""
Configuration settings for the Real-Time Multi-Symbol Market Analyzing and Prediction Engine.
Supports Gold (XAUUSD / GC=F) and Forex pairs (EURUSD).
"""
import os
import threading

# Global lock for thread-safe MT5 access
MT5_LOCK = threading.Lock()

# --- SYMBOL & DATA SOURCE CONFIGURATION ---
DEFAULT_DATA_SOURCE = "mt5"       # Options: 'yfinance', 'mt5'
SYMBOL_YF = "GC=F"                # Gold Futures on Yahoo Finance
SYMBOL_MT5 = "XAUUSD"             # Gold on MetaTrader 5
SYMBOL_EURUSD_YF = "EURUSD=X"     # EURUSD on Yahoo Finance
SYMBOL_EURUSD_MT5 = "EURUSD"      # EURUSD on MetaTrader 5
FALLBACK_SYMBOLS_YF = ["EURUSD=X", "JPY=X", "^GSPC"]  # Additional macro/forex features if needed

# All tradeable symbols (used by live engine and monitor)
ACTIVE_SYMBOLS_MT5 = ["XAUUSD", "EURUSD"]
ACTIVE_SYMBOLS_YF = ["GC=F", "EURUSD=X"]

# --- TIMEFRAMES ---
# Yahoo Finance valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo
TIMEFRAME_LTF = "15m"             # Default primary execution timeframe if none specified

def get_htf_timeframes(base_tf):
    """
    Returns (HTF1, HTF2) for a given base timeframe.
    Ensures multi-timeframe confluence logic is dynamically sound.
    """
    mapping = {
        "1m": ("5m", "15m"),
        "2m": ("15m", "1h"),
        "5m": ("15m", "1h"),
        "15m": ("1h", "1d"),
        "30m": ("4h", "1d"),
        "60m": ("4h", "1d"),
        "1h": ("4h", "1d"),
        "90m": ("1d", "1wk"),
        "4h": ("1d", "1wk"),
        "1d": ("1wk", "1mo"),
    }
    return mapping.get(base_tf.lower(), ("1h", "1d"))

# --- DATA DOWNLOAD & BUFFER PARAMETERS ---
HISTORICAL_PERIOD_YF = "60d"      # Note: yfinance limits 15m intraday data to the last 60 days
BUFFER_SIZE = 250                 # Minimum candles needed for live rolling calculations

# --- TRIPLE-BARRIER LABELING PARAMETERS ---
VOLATILITY_SPAN = 50              # Span for exponential moving standard deviation of returns
TP_MULT = 2.0                     # Take Profit multiplier of rolling volatility
SL_MULT = 1.5                     # Stop Loss multiplier of rolling volatility
MAX_HOLDING = 16                  # Maximum holding bars (4 hours on 15m chart) before time stop

# --- QUANTITATIVE FEATURE ENGINEERING PARAMETERS ---
SWING_WINDOW = 2                  # Bars on each side required for fractal swing confirmation (2*k+1 = 5 bars)
OB_DISP_MULT = 1.5                # Rolling ATR multiplier to define displacement impulse for Order Blocks
MNSR_RANGES = 5                   # Number of dealing range midpoints for MNSR moving average
PD_WINDOW = 50                    # Lookback window for Premium / Discount equilibrium range
ATR_WINDOW = 14                   # Rolling ATR period

# --- SMC EVENT-BASED TRAINING PARAMETERS ---
SMC_LOOKBACK = 5                  # Bars to look back for sweep/CHoCH confluence window
SMC_REGRESSION_HORIZON = 16      # Forward bars for regression target (matches MAX_HOLDING)
SMC_ENTRY_THRESHOLD = 0.5        # Min |expected_return| magnitude to approve an SMC trade
SMC_VOL_WINDOW = 50              # Rolling window for volatility normalization in regression target
SMC_MIN_EVENTS = 30              # Minimum SMC events required to train (raise to 100+ with MT5 multi-year data)

# --- ICT EVENT-BASED TRAINING PARAMETERS ---
ICT_LOOKBACK = 5                  # Bars to look back for Judas Swing/CHoCH
ICT_REGRESSION_HORIZON = 24       # Forward return horizon (6 hours on 15m)
ICT_ENTRY_THRESHOLD = 0.75        # Min |expected_return| magnitude for entry
ICT_MIN_EVENTS = 15               # Minimum events to allow training (raise to 100+ with MT5 multi-year data)
ICT_VOL_WINDOW = 50              # Rolling window for volatility normalization in regression target

# --- MODEL TRAINING & CV PARAMETERS ---
N_SPLITS = 5                      # Number of Purged K-Fold splits
EMBARGO_PCT = 0.02                # Percentage of dataset to embargo after each test fold
OPTUNA_TRIALS = 30                # Number of hyperparameter optimization trials
HOLDOUT_PCT = 0.20                # Out-of-sample holdout set percentage (last 20% chronologically)
NESTED_CV_ENABLED = True          # Enable nested cross-validation to prevent hyperparameter leakage
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Legacy single-symbol paths (kept for backward compatibility)
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xauusd_xgb_model.joblib")
LGB_MODEL_PATH = os.path.join(MODEL_DIR, "xauusd_lgb_model.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
OOS_PREDS_PATH = os.path.join(MODEL_DIR, "oos_predictions.joblib")
META_MODEL_PATH = os.path.join(MODEL_DIR, "meta_decision_maker.joblib")

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- LIVE ENGINE & EXECUTION PARAMETERS ---
ENABLE_LIVE_TRADING = False       # Safety switch: False = Advisory/Dry-Run Mode (No real orders sent)
CONFIDENCE_THRESHOLD = 0.20       # Minimum directional confidence required to trigger a trade signal
META_CONFIDENCE_THRESHOLD = 0.10  # Secondary AI Decision Maker minimum confidence to approve trade
SIGNAL_SMOOTHING_WINDOW = 5       # EMA lookback window for ML prediction probabilities (increased from 3 for stability)
MIN_SIGNAL_EDGE = 0.01            # Minimum edge over flat before a directional signal is accepted
MIN_DIRECTIONAL_PROB = 0.24       # Minimum probability for the winning directional class before acting
MAX_SPREAD_POINTS = 35.0          # Maximum allowable spread in Gold points (e.g., $0.35 on $2300 Gold)
TRADE_VOLUME_LOTS = 0.01          # Micro-lot size for automated execution
POLL_INTERVAL_SECONDS = 5         # Real-time candle polling interval
MAX_DAILY_LOSS = 200.0            # Hard stop for daily loss before new entries are allowed
MAX_CONSECUTIVE_LOSSES = 3       # Stop new entries after repeated losing trades
MAX_DRAWDOWN_PCT = 0.10           # Stop new entries if portfolio equity drawdown exceeds 10%

# --- SIGNAL CONVICTION & HYSTERESIS PARAMETERS ---
# These prevent signal flickering (e.g. SELL -> FLAT -> SELL) by requiring strong evidence
# to ENTER a signal, and sustained weakness to EXIT it. Mimics a confident human trader.
CONVICTION_ENTRY_THRESHOLD = 0.55    # Directional probability must exceed this to ENTER a new signal
CONVICTION_HOLD_THRESHOLD = 0.35     # Probability must stay above this to MAINTAIN signal (exit below)
CONVICTION_ENTRY_EDGE = 0.10         # Entry direction must beat opposing direction by this margin

# --- INSTITUTIONAL LLM REASONING & NVIDIA API CONFIGURATION ---
LLM_API_KEY = os.environ.get("NVIDIA_NIM_API_KEY", "")
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL_NAME = "openai/gpt-oss-120b"
LLM_VALIDATION_ENABLED = True     # Validate all open positions & signals between local Meta models and NVIDIA LLM


# =====================================================================================
# PER-SYMBOL INSTRUMENT PROFILES
# =====================================================================================
# Each profile contains instrument-specific parameters for trading, backtesting,
# risk management, and model file paths.

SYMBOL_PROFILES = {
    # --- GOLD (XAUUSD) ---
    "XAUUSD": {
        "name": "XAUUSD",
        "display_name": "Gold (XAUUSD)",
        "yf_ticker": "GC=F",
        "mt5_ticker": "XAUUSD",
        "contract_size": 100.0,          # 1 standard lot = 100 troy oz → $100 per $1.00 move
        "spread_dollar": 0.25,           # Typical Gold spread: $0.25 per oz (25 points)
        "slippage_dollar": 0.05,         # Typical execution slippage: $0.05 per oz
        "commission_per_lot": 5.0,       # $5.00 round-turn commission per standard lot
        "max_spread_points": 35.0,       # Maximum allowable spread in points
        "trade_volume_lots": 0.01,       # Default micro-lot for automated execution
        "pip_scale": 0.01,               # 1 point = $0.01 price movement
        "breakeven_buffer": 0.20,        # $0.20 buffer above/below entry for breakeven SL
        "default_atr": 4.50,             # Fallback ATR if live calculation fails
        "price_decimals": 2,             # Display precision (e.g. $2,650.30)
        "price_format": "dollar",        # '$X,XXX.XX' formatting
        "model_prefix": "xauusd",        # File prefix for model artifacts
    },

    # --- EURUSD ---
    "EURUSD": {
        "name": "EURUSD",
        "display_name": "EUR/USD",
        "yf_ticker": "EURUSD=X",
        "mt5_ticker": "EURUSD",
        "contract_size": 100000.0,       # 1 standard lot = 100,000 base currency units
        "spread_dollar": 0.00010,        # Typical EURUSD spread: 1.0 pip = 0.00010
        "slippage_dollar": 0.00005,      # Typical execution slippage: 0.5 pip
        "commission_per_lot": 7.0,       # $7.00 round-turn commission per standard lot
        "max_spread_points": 30.0,       # Maximum allowable spread in points (3.0 pips = 30 points)
        "trade_volume_lots": 0.01,       # Default micro-lot for automated execution
        "pip_scale": 0.0001,             # 1 pip = 0.0001 price movement (4th decimal)
        "breakeven_buffer": 0.00020,     # 2 pips buffer above/below entry for breakeven SL
        "default_atr": 0.00045,          # Fallback ATR if live calculation fails (~4.5 pips per 15m)
        "price_decimals": 5,             # Display precision (e.g. 1.08450)
        "price_format": "decimal",       # 'X.XXXXX' formatting
        "model_prefix": "eurusd",        # File prefix for model artifacts
    },
}

# Aliases for common ticker names → canonical symbol key
_SYMBOL_ALIASES = {
    "XAUUSD": "XAUUSD",
    "GC=F": "XAUUSD",
    "GOLD": "XAUUSD",
    "EURUSD": "EURUSD",
    "EURUSD=X": "EURUSD",
}


def resolve_symbol(symbol_input):
    """Resolve any symbol string (ticker, alias) to the canonical profile key."""
    key = symbol_input.upper().strip()
    return _SYMBOL_ALIASES.get(key, key)


def get_symbol_profile(symbol_input):
    """
    Return the full instrument profile dict for a given symbol.
    Raises KeyError if the symbol is not configured.
    """
    key = resolve_symbol(symbol_input)
    if key not in SYMBOL_PROFILES:
        raise KeyError(f"No symbol profile configured for '{symbol_input}' (resolved to '{key}'). "
                       f"Available profiles: {list(SYMBOL_PROFILES.keys())}")
    return SYMBOL_PROFILES[key]


def get_model_paths(symbol_input, timeframe="15m"):
    """
    Return a dict of per-symbol and per-timeframe model file paths.
    E.g. for 'EURUSD' and '5m'   { 'xgb': 'models/eurusd_5m_xgb_model.joblib', ... }
    If the requested timeframe models do not exist, gracefully fallback to the 15m models.
    """
    profile = get_symbol_profile(symbol_input)
    prefix = profile["model_prefix"]
    tf = timeframe.lower()
    
    paths = {
        "xgb": os.path.join(MODEL_DIR, f"{prefix}_{tf}_xgb_model.joblib"),
        "lgb": os.path.join(MODEL_DIR, f"{prefix}_{tf}_lgb_model.joblib"),
        "feature_names": os.path.join(MODEL_DIR, f"{prefix}_{tf}_feature_names.joblib"),
        "oos_preds": os.path.join(MODEL_DIR, f"{prefix}_{tf}_oos_predictions.joblib"),
        "meta": os.path.join(MODEL_DIR, f"{prefix}_{tf}_meta_decision_maker.joblib"),
        "smc_reg": os.path.join(MODEL_DIR, f"{prefix}_{tf}_smc_regressor.joblib"),
        "smc_feature_names": os.path.join(MODEL_DIR, f"{prefix}_{tf}_smc_feature_names.joblib"),
        "ict_reg": os.path.join(MODEL_DIR, f"{prefix}_{tf}_ict_regressor.joblib"),
        "ict_feature_names": os.path.join(MODEL_DIR, f"{prefix}_{tf}_ict_feature_names.joblib"),
    }
    
    # Scale-Invariant Fallback: If no model is trained for this timeframe, use the 15m model.
    if not os.path.exists(paths["xgb"]) and tf != "15m":
        print(f"[{symbol_input}|{tf}] No specific model found. Falling back to scale-invariant 15m model.")
        fallback_tf = "15m"
        return {
            "xgb": os.path.join(MODEL_DIR, f"{prefix}_{fallback_tf}_xgb_model.joblib"),
            "lgb": os.path.join(MODEL_DIR, f"{prefix}_{fallback_tf}_lgb_model.joblib"),
            "feature_names": os.path.join(MODEL_DIR, f"{prefix}_{fallback_tf}_feature_names.joblib"),
            "oos_preds": os.path.join(MODEL_DIR, f"{prefix}_{fallback_tf}_oos_predictions.joblib"),
            "meta": os.path.join(MODEL_DIR, f"{prefix}_{fallback_tf}_meta_decision_maker.joblib"),
            "smc_reg": os.path.join(MODEL_DIR, f"{prefix}_{fallback_tf}_smc_regressor.joblib"),
            "smc_feature_names": os.path.join(MODEL_DIR, f"{prefix}_{fallback_tf}_smc_feature_names.joblib"),
            "ict_reg": os.path.join(MODEL_DIR, f"{prefix}_{fallback_tf}_ict_regressor.joblib"),
            "ict_feature_names": os.path.join(MODEL_DIR, f"{prefix}_{fallback_tf}_ict_feature_names.joblib"),
        }
        
    return paths


def get_ticker_for_source(symbol_input, source="mt5"):
    """Return the correct ticker string for the given data source."""
    profile = get_symbol_profile(symbol_input)
    if source.lower() == "yfinance":
        return profile["yf_ticker"]
    return profile["mt5_ticker"]
