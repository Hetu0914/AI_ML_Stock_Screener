import pandas as pd
import numpy as np
import yfinance as yf
import joblib


# ==========================================
# FINAL AI ML STOCK SCREENER
# ==========================================

print("\n" + "=" * 70)
print("FINAL AI ML STOCK SCREENER")
print("=" * 70)

# Load trained model
model = joblib.load("stock_model.pkl")

print("\nModel loaded successfully.")

# Stocks used in our assignment
stocks = [
    "ITC.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS"
]

results = []

# ==========================================
# Analyze each stock
# ==========================================

for stock in stocks:

    print(f"\nDownloading {stock}...")

    data = yf.download(
        stock,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        print(f"No data available for {stock}")
        continue

    # Handle yfinance MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # ======================================
    # Calculate features
    # ======================================

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

    data["MOMENTUM_5D"] = (
        data["Close"].pct_change(5) * 100
    )

    data["MOMENTUM_10D"] = (
        data["Close"].pct_change(10) * 100
    )

    data["VOLATILITY_20D"] = (
        data["Close"].pct_change()
        .rolling(20)
        .std() * 100
    )

    data = data.dropna()

    if data.empty:
        print(f"Not enough data for {stock}")
        continue

    # Latest market data
    latest = data.iloc[-1]

    # ======================================
    # Prepare ML features
    # ======================================

    features = pd.DataFrame([{
        "Close": float(latest["Close"]),
        "Volume": float(latest["Volume"]),
        "SMMA_20": float(latest["SMMA_20"]),
        "SMMA_120": float(latest["SMMA_120"]),
        "SMMA_DIFF": float(latest["SMMA_DIFF"]),
        "MOMENTUM_5D": float(latest["MOMENTUM_5D"]),
        "MOMENTUM_10D": float(latest["MOMENTUM_10D"]),
        "VOLATILITY_20D": float(latest["VOLATILITY_20D"]),
        "Signal": 0
    }])

    # ======================================
    # ML prediction
    # ======================================

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    probability = probabilities[1] * 100

    # ======================================
    # Final decision
    # ======================================

    if prediction == 1 and probability >= 70:
        decision = "BUY"

    elif prediction == 1:
        decision = "WATCH"

    else:
        decision = "NO TRADE"

    results.append({
        "Stock": stock,
        "Date": data.index[-1].strftime("%Y-%m-%d"),
        "Close": round(float(latest["Close"]), 2),
        "Prediction": int(prediction),
        "Probability_%": round(probability, 2),
        "Decision": decision
    })


# ==========================================
# Display results
# ==========================================

print("\n" + "=" * 70)
print("FINAL SCREENING RESULTS")
print("=" * 70)

if results:

    result_df = pd.DataFrame(results)

    print(result_df.to_string(index=False))

    # ======================================
    # High confidence results
    # ======================================

    high_confidence = result_df[
        (result_df["Prediction"] == 1) &
        (result_df["Probability_%"] >= 70)
    ]

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE OPPORTUNITIES")
    print("=" * 70)

    if not high_confidence.empty:
        print(high_confidence.to_string(index=False))
    else:
        print("No high-confidence opportunities found.")

    # ======================================
    # Save results
    # ======================================

    result_df.to_csv(
        "final_screening_results.csv",
        index=False
    )

    print("\nSaved as: final_screening_results.csv")

else:

    print("No results generated.")