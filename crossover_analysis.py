import yfinance as yf
import pandas as pd

stocks = [
    "ITC.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS"
]

all_signals = []

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

        # BUY crossover
        data["BUY_SIGNAL"] = (
            (data["SMMA_20"] > data["SMMA_120"]) &
            (data["SMMA_20"].shift(1) <= data["SMMA_120"].shift(1))
        )

        # SELL crossover
        data["SELL_SIGNAL"] = (
            (data["SMMA_20"] < data["SMMA_120"]) &
            (data["SMMA_20"].shift(1) >= data["SMMA_120"].shift(1))
        )

        # Get crossover rows
        signals = data[
            data["BUY_SIGNAL"] | data["SELL_SIGNAL"]
        ].copy()

        for date, row in signals.iterrows():

            if row["BUY_SIGNAL"]:
                signal = "BUY"
            else:
                signal = "SELL"

            all_signals.append({
                "Stock": symbol,
                "Date": date.strftime("%Y-%m-%d"),
                "Signal": signal,
                "Close": round(float(row["Close"]), 2),
                "SMMA_20": round(float(row["SMMA_20"]), 2),
                "SMMA_120": round(float(row["SMMA_120"]), 2)
            })

    except Exception as e:
        print(f"Error with {symbol}: {e}")


# Create final DataFrame
signals_df = pd.DataFrame(all_signals)

print("\nALL DETECTED CROSSOVERS")
print("=" * 90)

if signals_df.empty:
    print("No crossover signals found.")
else:
    print(signals_df.to_string(index=False))

    print("\nTotal crossovers:", len(signals_df))