import numpy as np
import pandas as pd


def generate_market_metrics(df):
    """
    Generates market-analysis metrics from available OHLCV data.

    Note:
    This is demonstration/analytical data and is NOT live
    broker market-depth data.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    # Make sure required columns exist
    required = ["Close", "Volume"]

    for col in required:
        if col not in data.columns:
            data[col] = 0

    close = pd.to_numeric(data["Close"], errors="coerce")
    volume = pd.to_numeric(data["Volume"], errors="coerce").fillna(0)

    # Approximate market metrics for analytical demonstration
    data["Average_LTP_20"] = close.rolling(20).mean()
    data["Average_LTP_60"] = close.rolling(60).mean()

    data["Quantity_5M"] = volume.rolling(5).sum()
    data["Quantity_20M"] = volume.rolling(20).sum()
    data["Quantity_60M"] = volume.rolling(60).sum()

    # Demonstration market-depth values
    data["Bid_Price"] = close * 0.999
    data["Ask_Price"] = close * 1.001

    data["Bid_Quantity"] = volume.rolling(5).mean() * 1.05
    data["Ask_Quantity"] = volume.rolling(5).mean() * 1.02

    return data