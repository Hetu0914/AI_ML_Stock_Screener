import yfinance as yf
import pandas as pd

stocks = [
    "ITC.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS"
]

all_data = []

for symbol in stocks:

    try:
        print(f"\nDownloading {symbol}...")

        data = yf.download(
            symbol,
            period="10y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            print(f"No data for {symbol}")
            continue

        # Clean columns
        if data.columns.nlevels > 1:
            data.columns = data.columns.get_level_values(0)

        data = data.dropna(subset=["Close"]).copy()

        if len(data) < 120:
            print(f"Not enough data for {symbol}")
            continue

        # =====================================================
        # SMMA
        # =====================================================

        data["SMMA_20"] = data["Close"].ewm(
            alpha=1 / 20,
            adjust=False
        ).mean()

        data["SMMA_120"] = data["Close"].ewm(
            alpha=1 / 120,
            adjust=False
        ).mean()

        data["SMMA_DIFF"] = (
            data["SMMA_20"] - data["SMMA_120"]
        )

        # =====================================================
        # MOMENTUM
        # =====================================================

        data["MOMENTUM_5D"] = (
            data["Close"].pct_change(5) * 100
        )

        data["MOMENTUM_10D"] = (
            data["Close"].pct_change(10) * 100
        )

        # =====================================================
        # VOLATILITY
        # =====================================================

        data["VOLATILITY_20D"] = (
            data["Close"]
            .pct_change()
            .rolling(20)
            .std() * 100
        )

        # =====================================================
        # BUY / SELL CROSSOVER
        # =====================================================

        data["BUY_SIGNAL"] = (
            (data["SMMA_20"] > data["SMMA_120"]) &
            (
                data["SMMA_20"].shift(1)
                <= data["SMMA_120"].shift(1)
            )
        )

        data["SELL_SIGNAL"] = (
            (data["SMMA_20"] < data["SMMA_120"]) &
            (
                data["SMMA_20"].shift(1)
                >= data["SMMA_120"].shift(1)
            )
        )

        # =====================================================
        # FUTURE 5-DAY PRICE
        # =====================================================

        data["FUTURE_CLOSE_5D"] = (
            data["Close"].shift(-5)
        )

        # =====================================================
        # KEEP ONLY CROSSOVER DAYS
        # =====================================================

        signals = data[
            data["BUY_SIGNAL"] |
            data["SELL_SIGNAL"]
        ].copy()

        # Remove rows with missing ML features
        signals = signals.dropna(
            subset=[
                "MOMENTUM_5D",
                "MOMENTUM_10D",
                "VOLATILITY_20D",
                "FUTURE_CLOSE_5D"
            ]
        )

        # =====================================================
        # CREATE TRAINING ROWS
        # =====================================================

        for date, row in signals.iterrows():

            if row["BUY_SIGNAL"]:
                signal = 1

                return_pct = (
                    (
                        row["FUTURE_CLOSE_5D"]
                        - row["Close"]
                    )
                    / row["Close"]
                ) * 100

            else:
                signal = -1

                return_pct = (
                    (
                        row["Close"]
                        - row["FUTURE_CLOSE_5D"]
                    )
                    / row["Close"]
                ) * 100

            profitable = int(return_pct > 0)

            all_data.append({

                "Stock": symbol,

                "Date": date.strftime("%Y-%m-%d"),

                "Signal": signal,

                "Close": float(row["Close"]),

                "Volume": float(row["Volume"]),

                "SMMA_20": float(row["SMMA_20"]),

                "SMMA_120": float(row["SMMA_120"]),

                "SMMA_DIFF": float(row["SMMA_DIFF"]),

                "MOMENTUM_5D": float(
                    row["MOMENTUM_5D"]
                ),

                "MOMENTUM_10D": float(
                    row["MOMENTUM_10D"]
                ),

                "VOLATILITY_20D": float(
                    row["VOLATILITY_20D"]
                ),

                "RETURN_5D": float(
                    return_pct
                ),

                "Profitable": profitable
            })

        print(
            f"{symbol}: {len(signals)} crossover signals"
        )

    except Exception as e:

        print(
            f"Error with {symbol}: {e}"
        )


# =========================================================
# CREATE FINAL DATASET
# =========================================================

df = pd.DataFrame(all_data)

print("\n")
print("=" * 100)
print("ML DATASET")
print("=" * 100)

if df.empty:

    print("No crossover data available.")

else:

    print(
        df.to_string(index=False)
    )

    print("\nDataset shape:")
    print(df.shape)

    print("\nTarget distribution:")
    print(df["Profitable"].value_counts())

    print("\nSignal distribution:")
    print(df["Signal"].value_counts())

    df.to_csv(
        "ml_training_data.csv",
        index=False
    )

    print(
        "\nSaved as: ml_training_data.csv"
    )