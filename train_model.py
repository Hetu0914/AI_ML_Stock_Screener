import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("ml_training_data.csv")

print("\nML MODEL TRAINING")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

# Sort by date
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

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

X = df[features]
y = df["Profitable"]

# =========================================================
# TIME-BASED TRAIN / TEST SPLIT
# =========================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining period:")
print(
    df["Date"].iloc[0].date(),
    "to",
    df["Date"].iloc[split_index - 1].date()
)

print("\nTesting period:")
print(
    df["Date"].iloc[split_index].date(),
    "to",
    df["Date"].iloc[-1].date()
)

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# RANDOM FOREST RESULTS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# =========================================================
# BASELINE 1: ALWAYS PREDICT 0
# =========================================================

baseline_0 = [0] * len(y_test)

baseline_0_accuracy = accuracy_score(
    y_test,
    baseline_0
)

print("\nBaseline 1 - Always Predict 0:")
print(
    round(baseline_0_accuracy, 4)
)

# =========================================================
# BASELINE 2: ALWAYS PREDICT 1
# =========================================================

baseline_1 = [1] * len(y_test)

baseline_1_accuracy = accuracy_score(
    y_test,
    baseline_1
)

print("\nBaseline 2 - Always Predict 1:")
print(
    round(baseline_1_accuracy, 4)
)

# =========================================================
# MODEL COMPARISON
# =========================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": [
        "Always Predict 0",
        "Always Predict 1",
        "Random Forest"
    ],
    "Accuracy": [
        baseline_0_accuracy,
        baseline_1_accuracy,
        accuracy
    ]
})

print(
    comparison.to_string(index=False)
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(
    importance.to_string(index=False)
)

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "stock_model.pkl"
)

print("\nModel saved as: stock_model.pkl")