"""
Lookahead-Safe Quantitative Feature Engineering Engine for Gold (XAUUSD / GC=F) and Forex.
Implements ICT concepts (FVG, Order Blocks, PD Zones, MNSR), Order Flow proxies (CVD),
and Market Structure (BOS/CHOCH) with strict 0% lookahead bias guarantees.
"""
import pandas as pd
import numpy as np
import config


def calculate_atr(df, window=config.ATR_WINDOW):
    """Calculate Average True Range (ATR)."""
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def get_confirmed_swings(df, k=config.SWING_WINDOW):
    """
    Lookahead-safe fractal swing detection.
    At timestamp t, a swing high is confirmed if high[t-k] was the maximum over [t-2*k, ..., t].
    This ensures we NEVER use future bars to label a swing at timestamp t.
    """
    high = df['high'].values
    low = df['low'].values
    n = len(df)
    
    confirmed_highs = np.full(n, np.nan)
    confirmed_lows = np.full(n, np.nan)
    
    window = 2 * k + 1
    for i in range(window - 1, n):
        # Candidate bar is at i - k
        candidate_idx = i - k
        # Window from i - 2*k to i
        window_highs = high[i - 2*k : i + 1]
        window_lows = low[i - 2*k : i + 1]
        
        if high[candidate_idx] == np.max(window_highs):
            confirmed_highs[i] = high[candidate_idx]
        if low[candidate_idx] == np.min(window_lows):
            confirmed_lows[i] = low[candidate_idx]
            
    # Convert to series and forward fill so at any timestamp t we know the last confirmed swing
    s_highs = pd.Series(confirmed_highs, index=df.index).ffill()
    s_lows = pd.Series(confirmed_lows, index=df.index).ffill()
    
    return s_highs, s_lows


def calculate_fvg(df, atr_series):
    """
    Lookahead-safe Fair Value Gap (FVG) detection and unmitigated state tracking.
    An FVG is confirmed at candle 3 close (timestamp t).
    """
    n = len(df)
    low = df['low'].values
    high = df['high'].values
    close = df['close'].values
    atr = atr_series.values
    
    bull_fvg_size = np.zeros(n)
    bear_fvg_size = np.zeros(n)
    
    dist_bull_fvg = np.zeros(n)
    dist_bear_fvg = np.zeros(n)
    fvg_active = np.zeros(n)
    
    # Maintain list of active unmitigated FVGs: (top_price, bottom_price)
    active_bull_fvgs = []
    active_bear_fvgs = []
    
    for i in range(2, n):
        # Check if candle i completes a Bullish FVG (between i-2 High and i Low)
        if low[i] > high[i-2] and close[i-2] < close[i]:
            gap = low[i] - high[i-2]
            bull_fvg_size[i] = gap
            active_bull_fvgs.append((low[i], high[i-2]))
            
        # Check if candle i completes a Bearish FVG (between i High and i-2 Low)
        if high[i] < low[i-2] and close[i-2] > close[i]:
            gap = low[i-2] - high[i]
            bear_fvg_size[i] = gap
            active_bear_fvgs.append((low[i-2], high[i]))
            
        # Mitigate Bullish FVGs if price drops below gap midpoint
        active_bull_fvgs = [fvg for fvg in active_bull_fvgs if close[i] > (fvg[1] + (fvg[0] - fvg[1]) * 0.5)]
        # Mitigate Bearish FVGs if price rises above gap midpoint
        active_bear_fvgs = [fvg for fvg in active_bear_fvgs if close[i] < (fvg[0] - (fvg[0] - fvg[1]) * 0.5)]
        
        # Calculate distance to nearest unmitigated FVG normalized by ATR
        current_close = close[i]
        current_atr = max(atr[i], 1e-5)
        
        if active_bull_fvgs:
            # Nearest bull FVG top is the one closest below current close
            nearest_bull = max([fvg[0] for fvg in active_bull_fvgs])
            dist_bull_fvg[i] = (current_close - nearest_bull) / current_atr
        else:
            dist_bull_fvg[i] = 0.0
            
        if active_bear_fvgs:
            # Nearest bear FVG bottom is the one closest above current close
            nearest_bear = min([fvg[0] for fvg in active_bear_fvgs])
            dist_bear_fvg[i] = (nearest_bear - current_close) / current_atr
        else:
            dist_bear_fvg[i] = 0.0
            
        if active_bull_fvgs and not active_bear_fvgs:
            fvg_active[i] = 1.0
        elif active_bear_fvgs and not active_bull_fvgs:
            fvg_active[i] = -1.0
        elif active_bull_fvgs and active_bear_fvgs:
            fvg_active[i] = 1.0 if abs(dist_bull_fvg[i]) < abs(dist_bear_fvg[i]) else -1.0
        else:
            fvg_active[i] = 0.0
            
    df['bull_fvg_size'] = bull_fvg_size
    df['bear_fvg_size'] = bear_fvg_size
    df['dist_bull_fvg'] = dist_bull_fvg
    df['dist_bear_fvg'] = dist_bear_fvg
    df['fvg_active'] = fvg_active
    return df


def calculate_order_blocks(df, atr_series, disp_mult=config.OB_DISP_MULT, lookback=10):
    """
    Lookahead-safe Order Block (OB) detection and mitigation tracking.
    An OB is triggered when a strong displacement impulse occurs. The OB is defined as the
    last opposite-colored candle before the displacement.
    """
    n = len(df)
    open_p = df['open'].values
    close_p = df['close'].values
    high_p = df['high'].values
    low_p = df['low'].values
    atr = atr_series.values
    
    dist_bull_ob = np.zeros(n)
    dist_bear_ob = np.zeros(n)
    ob_active = np.zeros(n)
    
    active_bull_obs = []  # (high_boundary, low_boundary)
    active_bear_obs = []  # (high_boundary, low_boundary)
    
    for i in range(lookback, n):
        displacement = abs(close_p[i] - open_p[i]) > (disp_mult * atr[i-1])
        
        if displacement:
            if close_p[i] > open_p[i]:  # Bullish impulse
                # Look back over last 'lookback' bars for the last bearish candle
                for j in range(i - 1, max(-1, i - lookback - 1), -1):
                    if close_p[j] < open_p[j]:
                        active_bull_obs.append((high_p[j], low_p[j]))
                        break
            elif close_p[i] < open_p[i]:  # Bearish impulse
                # Look back for last bullish candle
                for j in range(i - 1, max(-1, i - lookback - 1), -1):
                    if close_p[j] > open_p[j]:
                        active_bear_obs.append((high_p[j], low_p[j]))
                        break
                        
        # Mitigate Bullish OBs if price drops below the OB low boundary
        active_bull_obs = [ob for ob in active_bull_obs if low_p[i] >= ob[1]]
        # Mitigate Bearish OBs if price rises above the OB high boundary
        active_bear_obs = [ob for ob in active_bear_obs if high_p[i] <= ob[0]]
        
        current_close = close_p[i]
        current_atr = max(atr[i], 1e-5)
        
        if active_bull_obs:
            nearest_bull_high = max([ob[0] for ob in active_bull_obs])
            dist_bull_ob[i] = (current_close - nearest_bull_high) / current_atr
        else:
            dist_bull_ob[i] = 0.0
            
        if active_bear_obs:
            nearest_bear_low = min([ob[1] for ob in active_bear_obs])
            dist_bear_ob[i] = (nearest_bear_low - current_close) / current_atr
        else:
            dist_bear_ob[i] = 0.0
            
        if active_bull_obs and not active_bear_obs:
            ob_active[i] = 1.0
        elif active_bear_obs and not active_bull_obs:
            ob_active[i] = -1.0
        elif active_bull_obs and active_bear_obs:
            ob_active[i] = 1.0 if abs(dist_bull_ob[i]) < abs(dist_bear_ob[i]) else -1.0
        else:
            ob_active[i] = 0.0
            
    df['dist_bull_ob'] = dist_bull_ob
    df['dist_bear_ob'] = dist_bear_ob
    df['ob_active'] = ob_active
    return df


def calculate_pd_zones(df, atr_series, window=config.PD_WINDOW):
    """
    Premium & Discount Zones & Fibonacci Retracements based on rolling swing equilibrium.
    Tracks 50% equilibrium, 61.8% OTE (Optimal Trade Entry), and 78.6% Deep OTE.
    """
    rolling_high = df['high'].rolling(window=window, min_periods=10).max()
    rolling_low = df['low'].rolling(window=window, min_periods=10).min()
    
    range_size = (rolling_high - rolling_low).replace(0, 1e-5)
    equilibrium = (rolling_high + rolling_low) / 2.0
    
    # Fibonacci levels inside the dealing range
    fib_618_discount = rolling_low + 0.382 * range_size  # 61.8% retracement down from swing high
    fib_786_discount = rolling_low + 0.214 * range_size  # 78.6% deep discount
    fib_618_premium = rolling_low + 0.618 * range_size   # 61.8% retracement up from swing low
    fib_786_premium = rolling_low + 0.786 * range_size   # 78.6% deep premium
    
    atr = atr_series.replace(0, np.nan)
    
    df['dist_to_equilibrium'] = ((df['close'] - equilibrium) / atr).fillna(0.0)
    df['dist_to_fib_618_bull'] = ((df['close'] - fib_618_discount) / atr).fillna(0.0)
    df['dist_to_fib_618_bear'] = ((fib_618_premium - df['close']) / atr).fillna(0.0)
    
    # Classify Zone: +1 Premium (above eq), -1 Discount (below eq), 0 Equilibrium
    df['pd_zone'] = np.where(df['close'] > equilibrium, 1.0, np.where(df['close'] < equilibrium, -1.0, 0.0))
    
    # Fibonacci OTE Zone Classification
    # +2: Deep Premium (>= 78.6%), +1: Premium OTE (61.8% to 78.6%), 0: Mid-range, -1: Discount OTE (61.8% to 78.6%), -2: Deep Discount (>= 78.6%)
    df['fib_ote_zone'] = np.where(df['close'] >= fib_786_premium, 2.0,
                         np.where((df['close'] >= fib_618_premium) & (df['close'] < fib_786_premium), 1.0,
                         np.where(df['close'] <= fib_786_discount, -2.0,
                         np.where((df['close'] <= fib_618_discount) & (df['close'] > fib_786_discount), -1.0, 0.0))))
    return df


def calculate_mnsr(df, atr_series, n_ranges=config.MNSR_RANGES):
    """
    Mean of Structural Range (MNSR).
    Calculates moving average of the midpoints of confirmed fractal dealing ranges.
    """
    s_highs, s_lows = get_confirmed_swings(df, k=config.SWING_WINDOW)
    range_midpoint = (s_highs + s_lows) / 2.0
    
    mnsr_val = range_midpoint.rolling(window=n_ranges, min_periods=1).mean()
    atr = atr_series.replace(0, np.nan)
    
    df['mnsr_val'] = mnsr_val
    df['dist_to_mnsr'] = (df['close'] - mnsr_val) / atr
    df['dist_to_mnsr'] = df['dist_to_mnsr'].fillna(0.0)
    return df


def calculate_order_flow(df):
    """
    Order Flow & Volume proxy.
    Uses Intraday Intensity / Close Location Value (CLV) to approximate buying vs selling delta.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume'].replace(0, 1e-5)
    
    # CLV ranges from -1 (closed on low) to +1 (closed on high)
    range_hl = (high - low).replace(0, 1e-5)
    clv = ((close - low) - (high - close)) / range_hl
    
    approx_delta = clv * volume
    cvd = approx_delta.rolling(window=50, min_periods=1).sum()
    
    df['delta_per_candle'] = approx_delta
    df['cvd_trend'] = (cvd - cvd.rolling(window=14, min_periods=1).mean()) / (volume.rolling(14, min_periods=1).mean() + 1e-5)
    df['volume_imbalance'] = approx_delta / volume
    return df


def calculate_market_structure(df):
    """
    Market Structure: Break of Structure (BOS) and Change of Character (CHOCH).
    """
    s_highs, s_lows = get_confirmed_swings(df, k=config.SWING_WINDOW)
    close = df['close']
    
    # Shift confirmed swings by 1 so we compare against previous structure
    prev_high = s_highs.shift(1)
    prev_low = s_lows.shift(1)
    
    bos_bull = (close > prev_high) & (close.shift(1) <= prev_high)
    bos_bear = (close < prev_low) & (close.shift(1) >= prev_low)
    
    df['bos'] = np.where(bos_bull, 1.0, np.where(bos_bear, -1.0, 0.0))
    
    # Track cumulative trend direction from BOS
    trend = np.zeros(len(df))
    current_trend = 0.0
    choch_flag = np.zeros(len(df))
    choch_direction = np.zeros(len(df))  # +1 bullish CHoCH, -1 bearish CHoCH
    
    bos_vals = df['bos'].values
    for i in range(len(df)):
        if bos_vals[i] == 1.0:
            if current_trend == -1.0:
                choch_flag[i] = 1.0  # Trend reversed from bear to bull
                choch_direction[i] = 1.0  # Bullish CHoCH
            current_trend = 1.0
        elif bos_vals[i] == -1.0:
            if current_trend == 1.0:
                choch_flag[i] = 1.0  # Trend reversed from bull to bear
                choch_direction[i] = -1.0  # Bearish CHoCH
            current_trend = -1.0
        trend[i] = current_trend
        
    df['trend_direction'] = trend
    df['choch_flag'] = choch_flag
    df['choch_direction'] = choch_direction
    return df


def add_technical_features(df):
    """Add standard momentum/volatility features (RSI, Returns, Volatility)."""
    # Returns
    df['returns'] = df['close'].pct_change().fillna(0.0)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0.0)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=14, min_periods=1).std().fillna(0.0)
    
    # RSI (14) & Stochastic RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss.replace(0, 1e-5))
    rsi_val = 100 - (100 / (1 + rs))
    df['rsi'] = (rsi_val / 100.0).fillna(0.5)
    
    rsi_min = rsi_val.rolling(14, min_periods=1).min()
    rsi_max = rsi_val.rolling(14, min_periods=1).max()
    stoch_k = (rsi_val - rsi_min) / (rsi_max - rsi_min + 1e-5)
    df['stoch_rsi_k'] = stoch_k.ewm(span=3).mean().fillna(0.5)
    df['stoch_rsi_d'] = stoch_k.rolling(3, min_periods=1).mean().fillna(0.5)
    
    # MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    price_scale = df['close'].replace(0, 1e-5)
    df['macd_line'] = (macd_line / price_scale).fillna(0.0)
    df['macd_signal'] = (signal_line / price_scale).fillna(0.0)
    df['macd_hist'] = (macd_hist / price_scale).fillna(0.0)
    df['macd_momentum'] = np.where(macd_hist > macd_hist.shift(1), 1.0, -1.0)
    
    # Bollinger Bands
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    std_20 = df['close'].rolling(window=20, min_periods=1).std().fillna(0.0)
    df['bb_upper'] = df['sma_20'] + (2 * std_20)
    df['bb_lower'] = df['sma_20'] - (2 * std_20)
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['sma_20']).fillna(0.0)
    
    # Lag Features (Historical Returns)
    for k in range(1, 6):
        df[f'return_lag_{k}'] = df['returns'].shift(k).fillna(0.0)
        
    return df


def calculate_algorithmic_clock(df):
    """
    Section 2: Algorithmic Clock and Temporal Normalization (NY Local Time).
    Tracks London Killzone, NY AM/PM Killzones, London Close, and Tuesday/Wednesday weighting.
    """
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                ny_times = df.index.tz_convert('America/New_York')
            else:
                ny_times = df.index.tz_localize('UTC').tz_convert('America/New_York')
            hours = ny_times.hour
            minutes = ny_times.minute
            day_of_week = ny_times.dayofweek
        elif 'time' in df.columns:
            dt_col = pd.to_datetime(df['time'])
            hours = dt_col.dt.hour
            minutes = dt_col.dt.minute
            day_of_week = dt_col.dt.dayofweek
        else:
            hours = np.zeros(len(df))
            minutes = np.zeros(len(df))
            day_of_week = np.zeros(len(df))
    except Exception:
        hours = np.zeros(len(df))
        minutes = np.zeros(len(df))
        day_of_week = np.zeros(len(df))

    time_float = hours + minutes / 60.0
    
    # 1. London Killzone (02:00 - 05:00 NY)
    df['kz_london'] = ((time_float >= 2.0) & (time_float <= 5.0)).astype(float)
    # 2. NY AM Killzone (08:30 - 11:00 NY)
    df['kz_ny_am'] = ((time_float >= 8.5) & (time_float <= 11.0)).astype(float)
    # 3. NY PM Killzone (13:30 - 16:00 NY)
    df['kz_ny_pm'] = ((time_float >= 13.5) & (time_float <= 16.0)).astype(float)
    # 4. London Close Exit Window (10:00 - 12:00 NY)
    df['kz_london_close'] = ((time_float >= 10.0) & (time_float <= 12.0)).astype(float)
    # 5. Tuesday / Wednesday "Meat in the Middle" (Day 1 = Tuesday, Day 2 = Wednesday)
    df['is_tues_wed'] = ((day_of_week == 1) | (day_of_week == 2)).astype(float)
    return df


def calculate_pd_clusters(df):
    """
    Section 3: PD Array Matrix as Boolean States & Clusters.
    - FLOD (First Line of Defense) Cluster: Active FVG overlapping with Order Block.
    - BPR (Balanced Price Range): BISI and SIBI overlapping zone near current price.
    """
    # FLOD Cluster (+1 for bullish alignment, -1 for bearish alignment)
    fvg_act = df.get('fvg_active', pd.Series(0.0, index=df.index))
    ob_act = df.get('ob_active', pd.Series(0.0, index=df.index))
    
    flod = np.where((fvg_act > 0) & (ob_act > 0), 1.0,
           np.where((fvg_act < 0) & (ob_act < 0), -1.0, 0.0))
    df['flod_cluster'] = flod
    
    # BPR Cluster (Balanced Price Range where both bull and bear FVGs exist close to price)
    dist_bull = df.get('dist_bull_fvg', pd.Series(0.0, index=df.index)).abs()
    dist_bear = df.get('dist_bear_fvg', pd.Series(0.0, index=df.index)).abs()
    
    bpr = np.where((dist_bull < 2.0) & (dist_bear < 2.0) & (fvg_act != 0), 1.0, 0.0)
    df['bpr_cluster'] = bpr * np.sign(fvg_act.replace(0, 1))
    return df


def calculate_liquidity_pools(df, atr_series):
    """
    Section 3: Liquidity Pools (Discrete swing points where middle wick is local extrema).
    Calculates normalized distance to confirmed Buy-Side (swing highs) and Sell-Side (swing lows) liquidity.
    """
    s_highs, s_lows = get_confirmed_swings(df, k=config.SWING_WINDOW)
    atr = atr_series.replace(0, np.nan)
    
    # Shift confirmed swings by 1 to guarantee 0% lookahead bias
    df['dist_buy_liquidity'] = ((s_highs.shift(1) - df['close']) / atr).fillna(0.0)
    df['dist_sell_liquidity'] = ((df['close'] - s_lows.shift(1)) / atr).fillna(0.0)
    return df


from numba import jit

@jit(nopython=True)
def _calculate_vp_core(high, low, close, vol, lookback):
    n = len(close)
    poc = np.zeros(n)
    vah = np.zeros(n)
    val = np.zeros(n)
    typ_price = (high + low + close) / 3.0
    
    for i in range(n):
        if i < 10:
            poc[i] = close[i]
            vah[i] = high[i]
            val[i] = low[i]
            continue
            
        start_idx = max(0, i - lookback + 1)
        w_prices = typ_price[start_idx:i+1]
        w_vols = vol[start_idx:i+1]
        
        min_p = np.min(w_prices)
        max_p = np.max(w_prices)
        if max_p == min_p:
            poc[i] = close[i]
            vah[i] = high[i]
            val[i] = low[i]
            continue
            
        bins = np.linspace(min_p, max_p, 20)
        bin_indices = np.searchsorted(bins, w_prices) - 1
        # Numba np.clip equivalent for arrays requires a loop, but we can do it with np.minimum/maximum
        bin_indices = np.maximum(0, np.minimum(18, bin_indices))
        
        bin_vols = np.bincount(bin_indices, weights=w_vols, minlength=19)
        max_bin = np.argmax(bin_vols)
        poc[i] = (bins[max_bin] + bins[max_bin + 1]) / 2.0
        
        total_vol = np.sum(bin_vols)
        target_vol = total_vol * 0.70
        curr_vol = bin_vols[max_bin]
        left_bin = max_bin
        right_bin = max_bin
        
        while curr_vol < target_vol and (left_bin > 0 or right_bin < 18):
            left_v = bin_vols[left_bin - 1] if left_bin > 0 else 0
            right_v = bin_vols[right_bin + 1] if right_bin < 18 else 0
            if left_v >= right_v and left_bin > 0:
                left_bin -= 1
                curr_vol += left_v
            elif right_bin < 18:
                right_bin += 1
                curr_vol += right_v
            else:
                break
                
        val[i] = bins[left_bin]
        vah[i] = bins[right_bin + 1] if right_bin + 1 < len(bins) else bins[right_bin]
        
    return poc, vah, val


def calculate_volume_profile(df, atr_series, lookback=96):
    """
    Lookahead-safe Volume Profile features (POC, VAH, VAL).
    Calculates rolling Point of Control and 70% Value Area bounds over the last 'lookback' candles.
    Optimized with Numba.
    """
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol = df['volume'].values
    atr = atr_series.replace(0, np.nan).values
    
    poc, vah, val = _calculate_vp_core(high, low, close, vol, lookback)
    
    df['poc_val'] = poc
    df['vah_val'] = vah
    df['val_val'] = val
    df['dist_to_poc'] = ((df['close'] - poc) / atr_series.replace(0, np.nan)).fillna(0.0)
    df['dist_to_vah'] = ((df['close'] - vah) / atr_series.replace(0, np.nan)).fillna(0.0)
    df['dist_to_val'] = ((df['close'] - val) / atr_series.replace(0, np.nan)).fillna(0.0)
    df['in_value_area'] = np.where((df['close'] >= val) & (df['close'] <= vah), 1.0, 0.0)
    return df


def calculate_htf_directional_bias(df):
    """
    Top-Down Institutional Directional Bias.
    HTF (1h/4h/Daily + Macro + Volume Profile) determines the direction (Long vs Short).
    LTF (15m) only triggers entries in the direction of the HTF bias.
    """
    bias_score = np.zeros(len(df))
    
    # 1. 1-Hour Market Structure & Trend
    if 'htf1_trend_direction' in df.columns:
        bias_score += df['htf1_trend_direction'].values * 1.5
    elif 'trend_direction' in df.columns:
        bias_score += df['trend_direction'].values * 0.5
        
    # 2. 4-Hour Market Structure & Trend
    if 'htf2_trend_direction' in df.columns:
        bias_score += df['htf2_trend_direction'].values * 2.0
        
    # 3. Macro Bias (DXY & Geopolitical Fear/VIX)
    if 'macro_bias' in df.columns:
        bias_score += df['macro_bias'].values * 1.5
        
    # 4. Volume Profile acceptance (if close > POC, bullish acceptance)
    if 'dist_to_poc' in df.columns:
        bias_score += np.where(df['dist_to_poc'].values > 0, 0.5, -0.5)
        
    # Assign directional bias: +1.0 for Bullish (Long Only), -1.0 for Bearish (Short Only), 0.0 for Neutral
    df['htf_directional_bias'] = np.where(bias_score >= 1.0, 1.0, np.where(bias_score <= -1.0, -1.0, 0.0))
    return df


def merge_timeframes(ltf_df, htf_df, prefix="htf_"):
    """
    Lookahead-safe multi-timeframe merging.
    Crucial Rule: htf_df must be shifted by 1 BEFORE forward-filling onto ltf_df
    to guarantee that forming higher-timeframe candles are never leaked.
    """
    if htf_df is None or htf_df.empty:
        return ltf_df
        
    # Extract key HTF features
    htf_cols = [c for c in htf_df.columns if c in [
        'pd_zone', 'ob_active', 'fvg_active', 'trend_direction', 'rsi', 'dist_to_equilibrium',
        'flod_cluster', 'bpr_cluster', 'kz_ny_am', 'dist_to_poc', 'in_value_area',
        'fib_ote_zone', 'macd_hist', 'macd_momentum'
    ]]
    
    htf_subset = htf_df[htf_cols].copy()
    htf_subset = htf_subset.add_prefix(prefix)
    
    # SHIFT BY 1 TO PREVENT FUTURE LEAK
    htf_shifted = htf_subset.shift(1)
    
    # Reindex onto LTF datetime index and forward fill
    merged = ltf_df.join(htf_shifted, how='left')
    merged = merged.ffill().fillna(0.0)
    return merged


def detect_liquidity_sweeps(df, s_highs, s_lows):
    """
    Detects when price wicks below a confirmed swing low (Sell-Side Liquidity Sweep)
    or wicks above a confirmed swing high (Buy-Side Liquidity Sweep) but closes back inside.
    This is the exact moment whales trigger retail stop losses.
    """
    prev_s_low = s_lows.shift(1)
    prev_s_high = s_highs.shift(1)
    
    # Sell-Side Sweep (Bullish Reversal trigger): Low breaks swing low, but Close is above it.
    sell_sweep = (df['low'] < prev_s_low) & (df['close'] > prev_s_low)
    
    # Buy-Side Sweep (Bearish Reversal trigger): High breaks swing high, but Close is below it.
    buy_sweep = (df['high'] > prev_s_high) & (df['close'] < prev_s_high)
    
    # +1.0 for Bullish Sweep (Ready to Buy), -1.0 for Bearish Sweep (Ready to Sell)
    df['liquidity_sweep'] = np.where(sell_sweep, 1.0, 
                            np.where(buy_sweep, -1.0, 0.0))
    return df


def calculate_daily_poc_magnet(df):
    """
    Calculates Yesterday's Point of Control (POC) and measures distance to it.
    The market acts like a magnet to this level.
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
        
    # Group by day
    df['date'] = df.index.date
    
    # Function to find POC (price level with max volume) for a day
    def get_poc(group):
        if group.empty: return np.nan
        # Bin prices into 50 levels
        bins = np.linspace(group['low'].min(), group['high'].max(), 50)
        # Fix: pd.Series doesn't work well with np.digitize if it has NaNs, but assuming clean data
        indices = np.searchsorted(bins, group['close'].values)
        indices = np.clip(indices, 0, len(bins) - 1)
        vols = np.bincount(indices, weights=group['volume'].values)
        if len(vols) == 0: return np.nan
        poc_bin = np.argmax(vols)
        return bins[min(poc_bin, len(bins)-1)]
        
    # Calculate POC for each day, then shift by 1 to get YESTERDAY'S POC
    daily_poc = df.groupby('date').apply(get_poc).shift(1)
    df['yesterday_poc'] = df['date'].map(daily_poc)
    
    # Calculate distance to Yesterday's POC normalized by ATR
    atr = calculate_atr(df)
    df['dist_to_daily_poc'] = ((df['close'] - df['yesterday_poc']) / atr).fillna(0.0)
    
    df.drop(columns=['date'], inplace=True)
    return df


def normalize_session_volume(df):
    try:
        hour = pd.DatetimeIndex(df.index).hour
    except Exception:
        hour = np.zeros(len(df))
    df['vol_zscore'] = df.groupby(hour)['volume'].transform(
        lambda x: (x - x.rolling(20, min_periods=1).mean()) / (x.rolling(20, min_periods=1).std() + 1e-5)
    )
    return df


def calculate_volume_anomaly(df):
    vol = df['volume'].replace(0, np.nan)
    vol_ma = vol.rolling(window=96, min_periods=20).mean()
    vol_std = vol.rolling(window=96, min_periods=20).std()
    df['volume_zscore'] = ((vol - vol_ma) / (vol_std + 1e-5)).fillna(0.0)
    df['volume_spike'] = (df['volume_zscore'] > 2.0).astype(float)
    spike_direction = np.where(df['close'] > df['open'], 1.0,
                      np.where(df['close'] < df['open'], -1.0, 0.0))
    df['directed_volume_spike'] = df['volume_spike'] * spike_direction
    return df


def detect_smc_event(df, lookback=None):
    """
    Flags the exact bars where the SMC Playbook conditions align.
    UPDATED: Relaxed rules to catch more valid setups (Sweep OR OB Tap + CHoCH).
    """
    if lookback is None:
        lookback = getattr(config, 'SMC_LOOKBACK', 8)  # Increased default to 8
    
    # 1. HTF Bias & Premium/Discount (Relaxed: Removed strict requirement)
    # We still calculate it, but won't force it to block the trade entirely.
    htf_pd = df.get('htf1_pd_zone', pd.Series(0.0, index=df.index))
    
    # 2. Liquidity Sweep OR Direct OB Tap
    liq_sweep = df.get('liquidity_sweep', pd.Series(0.0, index=df.index))
    bull_sweep_occurred = (liq_sweep == 1.0).astype(float).rolling(window=lookback, min_periods=1).max() > 0
    bear_sweep_occurred = (liq_sweep == -1.0).astype(float).rolling(window=lookback, min_periods=1).max() > 0
    
    # 3. LTF CHoCH (directional, within lookback window)
    choch_dir = df.get('choch_direction', pd.Series(0.0, index=df.index))
    bull_choch = (choch_dir == 1.0).astype(float).rolling(window=lookback, min_periods=1).max() > 0
    bear_choch = (choch_dir == -1.0).astype(float).rolling(window=lookback, min_periods=1).max() > 0
    
    # 4. FVG / OB Active at current price (Displacement)
    fvg_act = df.get('fvg_active', pd.Series(0.0, index=df.index))
    ob_act = df.get('ob_active', pd.Series(0.0, index=df.index))
    bull_pd_array_active = (fvg_act == 1.0) | (ob_act == 1.0)
    bear_pd_array_active = (fvg_act == -1.0) | (ob_act == -1.0)
    
    # --- SMC LONG EVENT: Bull Sweep/Tap + Bull CHoCH + Bull FVG/OB ---
    # Removed HTF Discount requirement to allow AI to learn context instead of hard-blocking
    smc_long = (bull_sweep_occurred & bull_choch & bull_pd_array_active).astype(float)
    
    # --- SMC SHORT EVENT: Bear Sweep/Tap + Bear CHoCH + Bear FVG/OB ---
    smc_short = (bear_sweep_occurred & bear_choch & bear_pd_array_active).astype(float)
    
    df['smc_long_event'] = smc_long
    df['smc_short_event'] = smc_short
    
    # Unified Trigger: 1 for Long Setup, -1 for Short Setup, 0 for No Setup
    df['smc_trigger'] = np.where(smc_long == 1.0, 1.0, 
                        np.where(smc_short == 1.0, -1.0, 0.0))
    return df


def detect_ict_event(df, lookback=None):
    """
    Flags the exact bars where the ICT Playbook conditions align.
    1. HTF Bias established.
    2. Judas Swing (Liquidity sweep against bias).
    3. Displacement (FVG in direction of bias).
    4. OTE Confluence (Price inside 62%-79% Fib of the displacement leg).
    5. Killzone Time (London or NY).
    """
    if lookback is None:
        lookback = getattr(config, 'ICT_LOOKBACK', 5)
        
    # --- 1. Time Killzones ---
    in_killzone = (df.get('kz_london', 0) == 1.0) | (df.get('kz_ny_am', 0) == 1.0)
    
    # --- 2. HTF Bias ---
    bull_bias = df.get('htf_directional_bias', 0.0) == 1.0
    bear_bias = df.get('htf_directional_bias', 0.0) == -1.0
    
    # If htf_directional_bias isn't populated (e.g. intermediate step), fall back to htf1_trend_direction
    if not bull_bias.any() and not bear_bias.any():
        bull_bias = df.get('htf1_trend_direction', 0.0) == 1.0
        bear_bias = df.get('htf1_trend_direction', 0.0) == -1.0
        
    # --- 3. Judas Swing (Sweep against bias) ---
    # If Bull bias, we want a recent Sell-Side Sweep (liquidity_sweep == 1.0)
    bull_judas = (df.get('liquidity_sweep', 0) == 1.0).astype(float).rolling(window=lookback, min_periods=1).max() > 0
    # If Bear bias, we want a recent Buy-Side Sweep (liquidity_sweep == -1.0)
    bear_judas = (df.get('liquidity_sweep', 0) == -1.0).astype(float).rolling(window=lookback, min_periods=1).max() > 0
    
    # --- 4. Displacement & FVG (In direction of bias) ---
    bull_displacement = df.get('fvg_active', 0) == 1.0
    bear_displacement = df.get('fvg_active', 0) == -1.0
    
    # --- 5. OTE (Optimal Trade Entry) Confluence ---
    # -1.0 / -2.0 means Discount OTE / Deep Discount -> Good for Longs
    # +1.0 / +2.0 means Premium OTE / Deep Premium -> Good for Shorts
    bull_ote = df.get('fib_ote_zone', 0) <= -1.0
    bear_ote = df.get('fib_ote_zone', 0) >= 1.0
    
    # --- ICT LONG EVENT: Killzone + Bull Bias + Judas Sweep + Bull FVG + OTE ---
    df['ict_long_event'] = (in_killzone & bull_bias & bull_judas & bull_displacement & bull_ote).astype(float)
    
    # --- ICT SHORT EVENT: Killzone + Bear Bias + Bear Judas + Bear FVG + OTE ---
    df['ict_short_event'] = (in_killzone & bear_bias & bear_judas & bear_displacement & bear_ote).astype(float)
    
    # Unified Trigger: 1 for Long, -1 for Short, 0 for No Setup
    df['ict_trigger'] = np.where(df['ict_long_event'] == 1.0, 1.0, 
                        np.where(df['ict_short_event'] == 1.0, -1.0, 0.0))
                        
    return df


def engineer_all_features(df, htf1_df=None, htf2_df=None, is_htf=False):
    """
    Master pipeline to generate all quantitative features for a dataframe.
    """
    df = df.copy()
    atr = calculate_atr(df)
    
    df = add_technical_features(df)
    df = calculate_fvg(df, atr)
    df = calculate_order_blocks(df, atr)
    df = calculate_pd_zones(df, atr)
    df = calculate_mnsr(df, atr)
    df = calculate_order_flow(df)
    df = calculate_market_structure(df)
    df = calculate_algorithmic_clock(df)
    df = calculate_pd_clusters(df)
    df = calculate_liquidity_pools(df, atr)
    df = calculate_volume_profile(df, atr)
    df = normalize_session_volume(df)
    df = calculate_volume_anomaly(df)
    
    s_highs, s_lows = get_confirmed_swings(df)
    df = detect_liquidity_sweeps(df, s_highs, s_lows)
    df = calculate_daily_poc_magnet(df)
    
    if not is_htf:
        try:
            from macro_sentiment import fetch_macro_data, calculate_macro_features
            start_date = df.index[0].date()
            end_date = df.index[-1].date()
            macro_df = fetch_macro_data(str(start_date), str(end_date))
            df = calculate_macro_features(df, macro_df=macro_df)
        except Exception as e:
            print(f"[Features] Macro data fetch failed: {e}. Using neutral bias.")
            for col in ['vix_regime', 'dxy_bias', 'us10y_bias', 'macro_bias']:
                if col not in df.columns:
                    df[col] = 0.0
            
    # If HTF dataframes are provided, process them and merge safely
    if htf1_df is not None and not htf1_df.empty:
        htf1_proc = engineer_all_features(htf1_df, is_htf=True)
        df = merge_timeframes(df, htf1_proc, prefix="htf1_")
        
    if htf2_df is not None and not htf2_df.empty:
        htf2_proc = engineer_all_features(htf2_df, is_htf=True)
        df = merge_timeframes(df, htf2_proc, prefix="htf2_")
        
    if not is_htf:
        df = calculate_htf_directional_bias(df)
        # SMC & ICT Event Detection (requires HTF bias to be merged first)
        df = detect_smc_event(df)
        df = detect_ict_event(df)
        
    # Final cleanup of any lingering infinite or NaN values
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.fillna(0.0)
    
    return df


def get_feature_column_names(df):
    """Return the list of columns that are ML features (excluding OHLCV)."""
    exclude = [
        'open', 'high', 'low', 'close', 'volume', 'tick_volume', 'time', 'target', 'ret_horizon', 'holding_bars',
        'poc_val', 'vah_val', 'val_val', 'mnsr_val', 'yesterday_poc',
        'smc_long_event', 'smc_short_event', 'smc_trigger', 'target_reg',
        'ict_long_event', 'ict_short_event', 'ict_trigger', 'target_ict_reg'
    ]
    return [c for c in df.columns if c not in exclude]
