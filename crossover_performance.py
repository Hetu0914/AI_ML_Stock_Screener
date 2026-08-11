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

        # SMMA
        data["SMMA_20"] = data["Close"].ewm(
            alpha=1/20,
            adjust=False
        ).mean()

        data["SMMA_120"] = data["Close"].ewm(
            alpha=1/120,
            adjust=False
        ).mean()

        # Signals
        data["BUY_SIGNAL"] = (
            (data["SMMA_20"] > data["SMMA_120"]) &
            (data["SMMA_20"].shift(1) <= data["SMMA_120"].shift(1))
        )

        data["SELL_SIGNAL"] = (
            (data["SMMA_20"] < data["SMMA_120"]) &
            (data["SMMA_20"].shift(1) >= data["SMMA_120"].shift(1))
        )

        # Price after 5 trading days
        data["FUTURE_CLOSE_5D"] = data["Close"].shift(-5)

        # Find signals
        signals = data[
            data["BUY_SIGNAL"] | data["SELL_SIGNAL"]
        ].copy()

        for date, row in signals.iterrows():

            entry_price = float(row["Close"])
            future_price = row["FUTURE_CLOSE_5D"]

            # Skip signals without 5-day future data
            if pd.isna(future_price):
                continue

            future_price = float(future_price)

            if row["BUY_SIGNAL"]:

                signal = "BUY"

                return_pct = (
                    (future_price - entry_price)
                    / entry_price
                ) * 100

            else:

                signal = "SELL"

                return_pct = (
                    (entry_price - future_price)
                    / entry_price
                ) * 100

            profitable = return_pct > 0

            all_signals.append({
                "Stock": symbol,
                "Date": date.strftime("%Y-%m-%d"),
                "Signal": signal,
                "Entry_Price": round(entry_price, 2),
                "Future_Price_5D": round(future_price, 2),
                "Return_5D_%": round(return_pct, 2),
                "Profitable": profitable
            })

    except Exception as e:
        print(f"Error with {symbol}: {e}")


# Create DataFrame
performance_df = pd.DataFrame(all_signals)

print("\nCROSSOVER PERFORMANCE")
print("=" * 100)

if performance_df.empty:

    print("No crossover data found.")

else:

    print(
        performance_df.to_string(index=False)
    )

    print("\nTotal signals:", len(performance_df))

    print(
        "Profitable signals:",
        performance_df["Profitable"].sum()
    )

    print(
        "Failed signals:",
        (~performance_df["Profitable"]).sum()
    )

    success_rate = (
        performance_df["Profitable"].mean() * 100
    )

    print(
        f"Overall success rate: {success_rate:.2f}%"
    )