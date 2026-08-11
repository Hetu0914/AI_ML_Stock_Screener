import yfinance as yf
import pandas as pd

stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "SBIN.NS",
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
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            continue

        if data.columns.nlevels > 1:
            data.columns = data.columns.get_level_values(0)

        data = data.dropna(subset=["Close"])

        if data.empty:
            continue

        latest = data.iloc[-1]

        ltp = float(latest["Close"])
        volume = int(latest["Volume"])

        # Requirement: LTP between ₹30 and ₹500
        if 30 <= ltp <= 500:

            results.append({
                "Stock": symbol,
                "LTP": round(ltp, 2),
                "Volume": volume
            })

    except Exception as e:
        print(f"Skipping {symbol}: {e}")


result_df = pd.DataFrame(results)

print("\nSTOCK SCREENING RESULT")
print("=" * 60)

if result_df.empty:
    print("No stocks found.")
else:
    print(result_df.to_string(index=False))