import yfinance as yf
import pandas as pd

stocks = [
    "ITC.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS"
]

results = []

for symbol in stocks:

    try:
        data = yf.download(
            symbol,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            continue

        # Clean MultiIndex
        if data.columns.nlevels > 1:
            data.columns = data.columns.get_level_values(0)

        data = data.dropna(subset=["Close"])

        if len(data) < 120:
            print(f"Not enough data for {symbol}")
            continue

        # SMMA 20
        data["SMMA_20"] = data["Close"].ewm(
            alpha=1/20,
            adjust=False
        ).mean()

        # SMMA 120
        data["SMMA_120"] = data["Close"].ewm(
            alpha=1/120,
            adjust=False
        ).mean()

        # Latest values
        latest = data.iloc[-1]

        smma20 = float(latest["SMMA_20"])
        smma120 = float(latest["SMMA_120"])
        ltp = float(latest["Close"])

        # Current trend
        if smma20 > smma120:
            trend = "BULLISH"
        elif smma20 < smma120:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        results.append({
            "Stock": symbol,
            "LTP": round(ltp, 2),
            "SMMA_20": round(smma20, 2),
            "SMMA_120": round(smma120, 2),
            "Trend": trend
        })

    except Exception as e:
        print(f"Error with {symbol}: {e}")


result_df = pd.DataFrame(results)

print("\nMULTI-STOCK SMMA ANALYSIS")
print("=" * 80)

if result_df.empty:
    print("No data available.")
else:
    print(result_df.to_string(index=False))