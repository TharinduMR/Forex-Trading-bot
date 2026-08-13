"""
Target Labeling and Volatility Estimator for Gold (XAUUSD / GC=F) and Forex.
Implements Marcos López de Prado's Triple-Barrier Method (Take Profit, Stop Loss, and Time Stop)
with volatility scaling and sample weighting.
"""
import pandas as pd
import numpy as np
import config


def compute_volatility(close_series, span=config.VOLATILITY_SPAN):
    """
    Calculate exponentially weighted standard deviation of percentage returns.
    """
    returns = close_series.pct_change()
    vol = returns.ewm(span=span, min_periods=10).std()
    return vol.fillna(vol.mean()).fillna(0.01)


def apply_triple_barrier(df, tp_mult=config.TP_MULT, sl_mult=config.SL_MULT, max_holding=config.MAX_HOLDING):
    """
    Apply López de Prado's Triple-Barrier Method.
    
    Barriers:
    1. Upper barrier (Take Profit): entry + tp_mult * sigma_dollar
    2. Lower barrier (Stop Loss): entry - sl_mult * sigma_dollar
    3. Vertical barrier (Time Stop): max_holding bars forward
    
    Returns:
    - target: 1 (Long Win), -1 (Short Win), 0 (Time Stop / Ranging), NaN (Incomplete horizon at tail)
    - holding_bars: Number of bars held before exit
    - ret_horizon: Actual percentage return achieved at exit
    """
    n = len(df)
    close_p = df['close'].values
    high_p = df['high'].values
    low_p = df['low'].values
    
    vol = compute_volatility(df['close']).values
    sigma_dollar = close_p * vol
    
    labels = np.full(n, np.nan)
    holding_bars = np.full(n, np.nan)
    ret_horizon = np.full(n, np.nan)
    
    for i in range(n - max_holding):
        entry = close_p[i]
        sigma = max(sigma_dollar[i], 1e-4)
        
        tp = entry + tp_mult * sigma
        sl = entry - sl_mult * sigma
        
        exit_bar = i + max_holding
        label = 0.0
        exit_price = close_p[i + max_holding]
        
        for j in range(i + 1, i + max_holding + 1):
            hit_tp = high_p[j] >= tp
            hit_sl = low_p[j] <= sl
            
            if hit_tp and hit_sl:
                # Ambiguous intrabar touch (Phantom Win Scenario)
                # To be conservative, if both are hit in the same volatile whip, we strictly penalize it.
                # Labeling it 0.0 teaches the Primary Model that this is NOT a winning setup.
                label = 0.0
                exit_bar = j
                exit_price = close_p[j]
                break
            elif hit_tp:
                label = 1.0
                exit_bar = j
                exit_price = tp
                break
            elif hit_sl:
                label = -1.0
                exit_bar = j
                exit_price = sl
                break
            elif j == i + max_holding:
                label = 0.0
                exit_bar = j
                exit_price = close_p[j]
                
        labels[i] = label
        holding_bars[i] = exit_bar - i
        ret_horizon[i] = (exit_price - entry) / entry
        
    df['target'] = labels
    df['holding_bars'] = holding_bars
    df['ret_horizon'] = ret_horizon
    return df


def compute_sample_weights(df, target_col='target'):
    """
    Compute sample weights combining:
    1. Volatility weighting (up-weighting volatile regimes)
    2. Class frequency balancing
    3. Temporal weighting (up-weighting Tuesday/Wednesday 'Meat in the Middle' as per technicals.txt)
    """
    valid_idx = df[target_col].dropna().index
    if len(valid_idx) == 0:
        return pd.Series(1.0, index=df.index)
        
    vol = compute_volatility(df['close'])
    vol_weight = vol / vol.mean()
    
    # Class frequency weights
    targets = df.loc[valid_idx, target_col].values
    unique_classes, counts = np.unique(targets, return_counts=True)
    total_samples = len(targets)
    n_classes = len(unique_classes)
    
    class_weight_dict = {cls: total_samples / (n_classes * count) for cls, count in zip(unique_classes, counts)}
    class_weights = df[target_col].map(class_weight_dict).fillna(1.0)
    
    # Temporal weights
    if 'is_tues_wed' in df.columns:
        temp_weight = df['is_tues_wed'].replace({0.0: 1.0, 1.0: 1.5})
    else:
        temp_weight = pd.Series(1.0, index=df.index)
    
    # Time decay: recent samples weighted 2x more than oldest
    n_samples = len(df)
    time_decay = np.linspace(0.5, 1.0, n_samples)
    time_decay_series = pd.Series(time_decay, index=df.index)
    
    # SMC / ICT Concept Weights (Massively prioritize Orderblocks and FVGs)
    smc_weight = pd.Series(1.0, index=df.index)
    if 'ob_active' in df.columns and 'fvg_active' in df.columns:
        # If any SMC zone is active, multiply weight by 3.0
        smc_active = (df['ob_active'] != 0) | (df['fvg_active'] != 0) | (df.get('flod_cluster', 0) != 0)
        smc_weight = np.where(smc_active, 3.0, 1.0)
    
    combined_weights = vol_weight * class_weights * temp_weight * smc_weight * time_decay_series
    # Normalize weights to mean 1.0
    combined_weights = combined_weights / combined_weights.mean()
    return combined_weights


def create_smc_regression_target(df, horizon=None, vol_window=None):
    """
    Calculates the volatility-adjusted forward return for SMC Event-Based Training.
    
    Target: (close[t+h] - close[t]) / rolling_volatility
    
    Only matters on bars where 'smc_trigger' != 0. All non-event bars are masked
    to 0.0 so the model never learns from random market noise.
    
    For Short setups (smc_trigger == -1), the sign is flipped so that a positive
    target always means "the trade went in our favor".
    
    Args:
        df: DataFrame with 'close' and 'smc_trigger' columns.
        horizon: Number of forward bars for return calculation. Defaults to config.SMC_REGRESSION_HORIZON.
        vol_window: Rolling window for volatility normalization. Defaults to config.SMC_VOL_WINDOW.
    
    Returns:
        df with 'target_reg' column added.
    """
    if horizon is None:
        horizon = getattr(config, 'SMC_REGRESSION_HORIZON', 16)
    if vol_window is None:
        vol_window = getattr(config, 'SMC_VOL_WINDOW', 50)
    
    # Forward absolute return over the horizon
    fwd_return = df['close'].shift(-horizon) - df['close']
    
    # Rolling volatility (dollar-based) to normalize the target
    vol = df['close'].pct_change().rolling(vol_window, min_periods=10).std()
    vol_dollar = vol * df['close']
    vol_dollar = vol_dollar.replace(0, np.nan)
    
    # Volatility-adjusted forward return
    target_reg = (fwd_return / vol_dollar).fillna(0.0)
    
    # For short setups, flip the sign so positive = "trade went in our favor"
    smc_trigger = df.get('smc_trigger', pd.Series(0.0, index=df.index))
    target_reg = np.where(smc_trigger == -1.0, -target_reg, target_reg)
    
    # Mask non-event bars to 0.0 — the model should only learn from SMC events
    target_reg = np.where(smc_trigger == 0.0, 0.0, target_reg)
    
    # Clip extreme outliers to ±10 to prevent gradient explosion
    target_reg = np.clip(target_reg, -10.0, 10.0)
    
    df['target_reg'] = target_reg
    return df


def create_ict_regression_target(df, horizon=None, vol_window=None):
    """
    Calculates the volatility-adjusted forward return for ICT setups.
    
    Target: (close[t+h] - close[t]) / rolling_volatility
    
    Only matters on bars where 'ict_trigger' != 0. All non-event bars are masked
    to 0.0 so the model doesn't learn from random noise.
    
    For Short setups (ict_trigger == -1), the sign is flipped so that a positive
    target always means "the trade went in our favor".
    
    Args:
        df: DataFrame with 'close' and 'ict_trigger' columns.
        horizon: Number of forward bars for return calculation. Defaults to config.ICT_REGRESSION_HORIZON (24).
        vol_window: Rolling window for volatility normalization. Defaults to config.ICT_VOL_WINDOW (50).
    
    Returns:
        df with 'target_ict_reg' column added.
    """
    if horizon is None:
        horizon = getattr(config, 'ICT_REGRESSION_HORIZON', 24)
    if vol_window is None:
        vol_window = getattr(config, 'ICT_VOL_WINDOW', 50)
        
    # Forward absolute return over the horizon
    fwd_return = df['close'].shift(-horizon) - df['close']
    
    # Rolling volatility (dollar-based) to normalize the target
    vol = df['close'].pct_change().rolling(vol_window, min_periods=10).std()
    vol_dollar = vol * df['close']
    vol_dollar = vol_dollar.replace(0, np.nan)
    
    # Volatility-adjusted forward return
    target_ict_reg = (fwd_return / vol_dollar).fillna(0.0)
    
    # For short setups, flip the sign so positive = "trade went in our favor"
    ict_trigger = df.get('ict_trigger', pd.Series(0.0, index=df.index))
    target_ict_reg = np.where(ict_trigger == -1.0, -target_ict_reg, target_ict_reg)
    
    # Mask non-event bars to 0.0
    target_ict_reg = np.where(ict_trigger == 0.0, 0.0, target_ict_reg)
    
    # Clip extreme outliers
    target_ict_reg = np.clip(target_ict_reg, -10.0, 10.0)
    
    df['target_ict_reg'] = target_ict_reg
    return df
