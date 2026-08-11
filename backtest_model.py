import pandas as pd
import joblib

from sklearn.metrics import accuracy_score


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("ml_training_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


print("\nML BACKTEST")
print("=" * 60)

print("\nDataset:")
print(df.shape)


# =========================================================
# FEATURES
# =========================================================

features = [
    "Close",
    "Volume",
    "SMMA_20",
    "SMMA_120",
    "SMMA_DIFF",
    "MOMENTUM_5D",
    "MOMENTUM_10D",
    "VOLATILITY_20D",
    "Signal"
]


# =========================================================
# TIME-BASED TEST SET
# =========================================================

split_index = int(len(df) * 0.80)

test_data = df.iloc[split_index:].copy()

X_test = test_data[features]


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("stock_model.pkl")


# =========================================================
# ML PREDICTIONS
# =========================================================

test_data["Predicted"] = model.predict(X_test)


# =========================================================
# PREDICTION PROBABILITY
# =========================================================

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(X_test)

    test_data["Probability"] = probabilities.max(axis=1)

else:

    test_data["Probability"] = 0.0


# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    test_data["Profitable"],
    test_data["Predicted"]
)

print("\nModel Accuracy:")
print(round(accuracy, 4))


# =========================================================
# MODEL TRADES
# =========================================================

model_trades = test_data[
    test_data["Predicted"] == 1
].copy()


print("\nPredicted profitable trades:")
print(len(model_trades))


# =========================================================
# ACTUAL RESULTS
# =========================================================

if len(model_trades) > 0:

    winning_trades = (
        model_trades["Profitable"] == 1
    ).sum()

    losing_trades = (
        model_trades["Profitable"] == 0
    ).sum()

    win_rate = (
        winning_trades /
        len(model_trades)
    ) * 100

    average_return = (
        model_trades["RETURN_5D"].mean()
    )

    total_return = (
        model_trades["RETURN_5D"].sum()
    )

    print("\nModel Backtest Results")
    print("-" * 40)

    print(
        "Winning trades:",
        winning_trades
    )

    print(
        "Losing trades:",
        losing_trades
    )

    print(
        "Win rate:",
        round(win_rate, 2),
        "%"
    )

    print(
        "Average 5D return:",
        round(average_return, 2),
        "%"
    )

    print(
        "Total 5D return:",
        round(total_return, 2),
        "%"
    )

else:

    print(
        "\nNo profitable trades were predicted."
    )


# =========================================================
# SIMPLE CROSSOVER STRATEGY
# =========================================================

crossover_winners = (
    test_data["Profitable"] == 1
).sum()

crossover_losers = (
    test_data["Profitable"] == 0
).sum()

crossover_win_rate = (
    crossover_winners /
    len(test_data)
) * 100

crossover_average_return = (
    test_data["RETURN_5D"].mean()
)

crossover_total_return = (
    test_data["RETURN_5D"].sum()
)


print("\n" + "=" * 60)
print("SIMPLE CROSSOVER STRATEGY")
print("=" * 60)

print(
    "\nTotal trades:",
    len(test_data)
)

print(
    "Winning trades:",
    crossover_winners
)

print(
    "Losing trades:",
    crossover_losers
)

print(
    "Win rate:",
    round(crossover_win_rate, 2),
    "%"
)

print(
    "Average 5D return:",
    round(crossover_average_return, 2),
    "%"
)

print(
    "Total 5D return:",
    round(crossover_total_return, 2),
    "%"
)


# =========================================================
# HIGH CONFIDENCE TRADES
# =========================================================

high_confidence = test_data[
    (test_data["Predicted"] == 1) &
    (test_data["Probability"] >= 0.70)
].copy()


print("\n" + "=" * 60)
print("HIGH-CONFIDENCE ML TRADES")
print("=" * 60)

print(
    "\nProbability >= 70%"
)

print(
    "Number of trades:",
    len(high_confidence)
)


if len(high_confidence) > 0:

    print(
        "\nHigh-confidence trades:"
    )

    print(
        high_confidence[
            [
                "Stock",
                "Date",
                "Signal",
                "RETURN_5D",
                "Profitable",
                "Probability"
            ]
        ].to_string(index=False)
    )

    high_conf_win_rate = (
        high_confidence["Profitable"].mean()
        * 100
    )

    high_conf_return = (
        high_confidence["RETURN_5D"].mean()
    )

    print(
        "\nHigh-confidence win rate:",
        round(high_conf_win_rate, 2),
        "%"
    )

    print(
        "High-confidence average return:",
        round(high_conf_return, 2),
        "%"
    )

else:

    print(
        "\nNo high-confidence trades."
    )


# =========================================================
# SAVE BACKTEST RESULTS
# =========================================================

test_data.to_csv(
    "backtest_results.csv",
    index=False
)

print(
    "\nSaved as: backtest_results.csv"
)