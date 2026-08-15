"""
Order Block & High Probability Setup Detector
Identifies institutional order blocks, validates them, and calculates trade setups.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class SetupType(Enum):
    BULLISH_OB = "Bullish Order Block"
    BEARISH_OB = "Bearish Order Block"
    MSB_RETEST = "MSB Retest"
    BREAKOUT_PULSE = "Breakout Pulse"

@dataclass
class TradeSetup:
    setup_type: SetupType
    symbol: str
    timeframe: str
    direction: int  # 1 for Buy, -1 for Sell
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0.0 to 1.0
    rationale: str
    order_block_start: float
    order_block_end: float
    timestamp: pd.Timestamp
    status: str = "PENDING"  # PENDING, ACTIVE, FILLED, CANCELLED

class OrderBlockEngine:
    def __init__(self, min_imbalance_ratio: float = 2.0, min_volume_ratio: float = 1.5):
        self.min_imbalance_ratio = min_imbalance_ratio
        self.min_volume_ratio = min_volume_ratio
        
    def detect_order_blocks(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[TradeSetup]:
        """
        Detects valid Order Blocks based on:
        1. Strong move away (Imbalance/FVG)
        2. Volume spike at origin
        3. Liquidity sweep prior to move (optional but preferred)
        """
        if len(df) < 50:
            return []
            
        setups = []
        df = df.copy()
        
        # Calculate Imbalance (FVG)
        df['fvg_lower'] = df['high'].shift(2)
        df['fvg_upper'] = df['low'].shift(2)
        df['fvg_size'] = df['fvg_upper'] - df['fvg_lower']
        
        # Calculate Average True Range for dynamic thresholds
        df['atr'] = self._calculate_atr(df, period=14)
        
        # Identify Strong Candles (Displacement)
        df['body_size'] = abs(df['close'] - df['open'])
        df['avg_body'] = df['body_size'].rolling(20).mean()
        df['is_displacement'] = df['body_size'] > (df['avg_body'] * 1.5)
        
        # Volume Check
        df['vol_avg'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_avg']
        
        for i in range(10, len(df) - 5):
            # Bullish OB Detection
            if df['is_displacement'].iloc[i] and df['close'].iloc[i] > df['open'].iloc[i]:
                if self._validate_bullish_ob(df, i):
                    setup = self._create_bullish_setup(df, i, symbol, timeframe)
                    if setup:
                        setups.append(setup)
            
            # Bearish OB Detection
            elif df['is_displacement'].iloc[i] and df['close'].iloc[i] < df['open'].iloc[i]:
                if self._validate_bearish_ob(df, i):
                    setup = self._create_bearish_setup(df, i, symbol, timeframe)
                    if setup:
                        setups.append(setup)
                        
        return setups

    def _validate_bullish_ob(self, df: pd.DataFrame, idx: int) -> bool:
        """Validate Bullish Order Block criteria"""
        current = df.iloc[idx]
        
        # 1. Must have FVG immediately after
        fvg_exists = (df['low'].iloc[idx+1] > df['high'].iloc[idx-1]) or \
                     (df['low'].iloc[idx+2] > df['high'].iloc[idx-1])
        
        # 2. Volume Spike
        volume_ok = current['vol_ratio'] > self.min_volume_ratio
        
        # 3. Price hasn't deeply violated the block yet (optional for fresh blocks)
        # For re-entry, we check if price is returning to the block
        
        return fvg_exists and volume_ok

    def _validate_bearish_ob(self, df: pd.DataFrame, idx: int) -> bool:
        """Validate Bearish Order Block criteria"""
        current = df.iloc[idx]
        
        # 1. Must have FVG immediately after
        fvg_exists = (df['high'].iloc[idx+1] < df['low'].iloc[idx-1]) or \
                     (df['high'].iloc[idx+2] < df['low'].iloc[idx-1])
        
        # 2. Volume Spike
        volume_ok = current['vol_ratio'] > self.min_volume_ratio
        
        return fvg_exists and volume_ok

    def _create_bullish_setup(self, df: pd.DataFrame, idx: int, symbol: str, timeframe: str) -> Optional[TradeSetup]:
        candle = df.iloc[idx]
        ob_low = candle['low']
        ob_high = candle['close'] # Top of body for entry
        
        # Entry: 50% mean threshold of the OB
        entry_price = (ob_low + ob_high) / 2.0
        
        # Stop Loss: Below the low of the OB minus buffer
        buffer = candle['atr'] * 0.5
        stop_loss = ob_low - buffer
        
        # Take Profit: 1:2 or 1:3 RR, or next liquidity pool (recent high)
        risk = entry_price - stop_loss
        take_profit = entry_price + (risk * 2.5)
        
        # Confidence Scoring
        confidence = 0.5
        if candle['vol_ratio'] > 2.0: confidence += 0.2
        if (candle['close'] - candle['open']) > candle['avg_body'] * 2: confidence += 0.2
        if df['fvg_size'].iloc[idx+1] > 0: confidence += 0.1
        
        return TradeSetup(
            setup_type=SetupType.BULLISH_OB,
            symbol=symbol,
            timeframe=timeframe,
            direction=1,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=min(confidence, 0.95),
            rationale=f"Bullish OB detected with {candle['vol_ratio']:.2f}x volume. FVG confirmed.",
            order_block_start=ob_low,
            order_block_end=ob_high,
            timestamp=candle.name,
            status="PENDING"
        )

    def _create_bearish_setup(self, df: pd.DataFrame, idx: int, symbol: str, timeframe: str) -> Optional[TradeSetup]:
        candle = df.iloc[idx]
        ob_high = candle['high']
        ob_low = candle['close'] # Bottom of body for entry
        
        # Entry: 50% mean threshold
        entry_price = (ob_low + ob_high) / 2.0
        
        # Stop Loss: Above the high plus buffer
        buffer = candle['atr'] * 0.5
        stop_loss = ob_high + buffer
        
        # Take Profit
        risk = stop_loss - entry_price
        take_profit = entry_price - (risk * 2.5)
        
        # Confidence Scoring
        confidence = 0.5
        if candle['vol_ratio'] > 2.0: confidence += 0.2
        if (candle['open'] - candle['close']) > candle['avg_body'] * 2: confidence += 0.2
        if df['fvg_size'].iloc[idx+1] > 0: confidence += 0.1
        
        return TradeSetup(
            setup_type=SetupType.BEARISH_OB,
            symbol=symbol,
            timeframe=timeframe,
            direction=-1,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=min(confidence, 0.95),
            rationale=f"Bearish OB detected with {candle['vol_ratio']:.2f}x volume. FVG confirmed.",
            order_block_start=ob_high,
            order_block_end=ob_low,
            timestamp=candle.name,
            status="PENDING"
        )

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def find_active_setups(self, current_price: float, setups: List[TradeSetup], tolerance_pips: float = 0.0005) -> List[TradeSetup]:
        """Filter setups that are currently within entry range"""
        active = []
        for setup in setups:
            if setup.status != "PENDING":
                continue
                
            diff = abs(current_price - setup.entry_price)
            # Allow entry if price is within tolerance or has crossed into the zone
            if setup.direction == 1: # Buy
                if current_price <= setup.entry_price + tolerance_pips and current_price >= setup.stop_loss:
                    active.append(setup)
            else: # Sell
                if current_price >= setup.entry_price - tolerance_pips and current_price <= setup.stop_loss:
                    active.append(setup)
                    
        return active
