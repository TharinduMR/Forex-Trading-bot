import pandas as pd
from data_loader import get_data_loader
from features import engineer_all_features

loader = get_data_loader("mt5") # or yfinance
df = loader.fetch_latest_buffer("XAUUSD", "15m", buffer_size=250)
df = engineer_all_features(df)

# Look at the latest closed candle's Volume Profile data
latest_candle = df.iloc[-2]
print(f"Current Close: {latest_candle['close']}")
print(f"Point of Control (POC): {latest_candle['poc_val']}")
print(f"Value Area High (VAH): {latest_candle['vah_val']}")
print(f"Value Area Low (VAL): {latest_candle['val_val']}")
print(f"Distance to POC (ATR normalized): {latest_candle['dist_to_poc']:.2f}")
print(f"In Value Area? {latest_candle['in_value_area']}")
print(f"Yesterday's POC: {latest_candle['yesterday_poc']}")