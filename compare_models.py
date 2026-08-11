import pandas as pd

# Load backtest results
df = pd.read_csv("backtest_results.csv")

print("\n## ML MODEL")

total_trades = len(df)
winning_trades = (df["Profitable"] == 1).sum()
losing_trades = (df["Profitable"] == 0).sum()

win_rate = winning_trades / total_trades * 100
avg_return = df["RETURN_5D"].mean()
total_return = df["RETURN_5D"].sum()

print(f"Total test trades:       {total_trades}")
print(f"Winning trades:          {winning_trades}")
print(f"Losing trades:           {losing_trades}")
print(f"Win rate:                {win_rate:.2f}%")
print(f"Average 5D return:       {avg_return:.2f}%")
print(f"Total 5D return:         {total_return:.2f}%")


# --------------------------------------------------
# HIGH CONFIDENCE
# --------------------------------------------------

high_conf = df[df["Probability"] >= 0.70].copy()

print("\n## ML HIGH-CONFIDENCE (>=70%)")

if len(high_conf) > 0:

    hc_trades = len(high_conf)
    hc_wins = (high_conf["Profitable"] == 1).sum()

    hc_win_rate = hc_wins / hc_trades * 100
    hc_avg_return = high_conf["RETURN_5D"].mean()
    hc_total_return = high_conf["RETURN_5D"].sum()

    print(f"Trades:                  {hc_trades}")
    print(f"Win rate:                {hc_win_rate:.2f}%")
    print(f"Average 5D return:       {hc_avg_return:.2f}%")
    print(f"Total 5D return:         {hc_total_return:.2f}%")

else:

    hc_trades = 0
    hc_win_rate = 0
    hc_avg_return = 0
    hc_total_return = 0

    print("No high-confidence trades.")


# --------------------------------------------------
# FINAL COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame({

    "Strategy": [
        "Random Forest ML",
        "ML High Confidence >=70%"
    ],

    "Trades": [
        total_trades,
        hc_trades
    ],

    "Win Rate %": [
        win_rate,
        hc_win_rate
    ],

    "Average Return %": [
        avg_return,
        hc_avg_return
    ],

    "Total Return %": [
        total_return,
        hc_total_return
    ]
})


print("\n## FINAL COMPARISON\n")
print(comparison.to_string(index=False))


comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nSaved as: model_comparison.csv")