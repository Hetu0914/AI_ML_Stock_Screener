import yfinance as yf

# Download 2 years of daily Reliance data
stock = yf.download(
    "RELIANCE.NS",
    period="2y",
    interval="1d",
    auto_adjust=False
)

# Remove extra ticker level if present
if hasattr(stock.columns, "get_level_values"):
    if stock.columns.nlevels > 1:
        stock.columns = stock.columns.get_level_values(0)

# Remove rows with missing Close prices
stock = stock.dropna(subset=["Close"])

# Calculate SMMA 20
stock["SMMA_20"] = stock["Close"].ewm(
    alpha=1/20,
    adjust=False
).mean()

# Calculate SMMA 120
stock["SMMA_120"] = stock["Close"].ewm(
    alpha=1/120,
    adjust=False
).mean()

# Detect BUY crossover
stock["BUY_SIGNAL"] = (
    (stock["SMMA_20"] > stock["SMMA_120"]) &
    (stock["SMMA_20"].shift(1) <= stock["SMMA_120"].shift(1))
)

# Detect SELL crossover
stock["SELL_SIGNAL"] = (
    (stock["SMMA_20"] < stock["SMMA_120"]) &
    (stock["SMMA_20"].shift(1) >= stock["SMMA_120"].shift(1))
)

# Future price after 5 trading days
stock["FUTURE_CLOSE_5D"] = stock["Close"].shift(-5)

# BUY return
stock["BUY_RETURN_5D"] = (
    stock["FUTURE_CLOSE_5D"] - stock["Close"]
) / stock["Close"]

# SELL return
stock["SELL_RETURN_5D"] = (
    stock["Close"] - stock["FUTURE_CLOSE_5D"]
) / stock["Close"]

# Profitability
stock["BUY_PROFITABLE"] = stock["BUY_RETURN_5D"] > 0
stock["SELL_PROFITABLE"] = stock["SELL_RETURN_5D"] > 0

# Get crossover rows
signals = stock[
    stock["BUY_SIGNAL"] | stock["SELL_SIGNAL"]
]

print("\nNumber of rows:", len(stock))
print("Number of crossovers:", len(signals))

print("\nCROSSOVER PERFORMANCE")
print("=" * 80)

print(
    signals[
        [
            "Close",
            "SMMA_20",
            "SMMA_120",
            "BUY_SIGNAL",
            "SELL_SIGNAL",
            "FUTURE_CLOSE_5D",
            "BUY_RETURN_5D",
            "SELL_RETURN_5D",
            "BUY_PROFITABLE",
            "SELL_PROFITABLE"
        ]
    ]
)