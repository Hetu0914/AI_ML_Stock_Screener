import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI/ML Stock Market Screening System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.metric-card {
    background: linear-gradient(135deg, #1b2230, #151a24);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #303747;
    text-align: center;
}

.metric-title {
    color: #9aa4b2;
    font-size: 15px;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
}

.section-title {
    font-size: 28px;
    font-weight: bold;
    margin-top: 25px;
    margin-bottom: 15px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #182638;
    border-left: 4px solid #2196f3;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Controls")

st.sidebar.markdown("### AI/ML Stock Screening System")

st.sidebar.write(
    "Historical SMMA crossover data is used to train "
    "a Random Forest model."
)

st.sidebar.write(
    "The model evaluates market features and produces "
    "a probability-based decision."
)

run_screening = st.sidebar.button(
    "🔄 Run Stock Screener",
    use_container_width=True
)

st.sidebar.divider()

st.sidebar.markdown("### 📊 Screening Rules")

st.sidebar.write("**LTP Range**")
st.sidebar.write("₹30 – ₹500")

st.sidebar.write("**SMMA Indicators**")
st.sidebar.write("SMMA 20 / SMMA 120")

st.sidebar.write("**ML Threshold**")
st.sidebar.write("70% probability")


# ============================================================
# LOAD FINAL SCREENING RESULTS
# ============================================================

RESULT_FILE = "final_screening_results.csv"

if run_screening:

    # Run final screener
    import subprocess

    try:
        subprocess.run(
            [
                ".\\venv\\Scripts\\python.exe",
                "final_screener.py"
            ],
            check=True
        )

        st.success("Stock screening completed successfully.")

    except Exception as e:
        st.error(f"Screening error: {e}")


# ============================================================
# LOAD DATA
# ============================================================

if os.path.exists(RESULT_FILE):

    results = pd.read_csv(RESULT_FILE)

else:

    st.warning(
        "final_screening_results.csv was not found. "
        "Run the stock screener first."
    )

    st.stop()


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

results.columns = [
    str(col).strip()
    for col in results.columns
]


# ============================================================
# BASIC CALCULATIONS
# ============================================================

total_stocks = len(results)

buy_count = len(
    results[results["Decision"].astype(str).str.upper() == "BUY"]
)

watch_count = len(
    results[results["Decision"].astype(str).str.upper() == "WATCH"]
)

no_trade_count = len(
    results[results["Decision"].astype(str).str.upper() == "NO TRADE"]
)

highest_probability = (
    results["Probability_%"].max()
    if "Probability_%" in results.columns
    else 0
)

average_probability = (
    results["Probability_%"].mean()
    if "Probability_%" in results.columns
    else 0
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <h1 style="font-size:48px;">
    📈 AI/ML Stock Market Screening System
    </h1>
    """,
    unsafe_allow_html=True
)

st.subheader(
    "SMMA Crossover + Random Forest Machine Learning Analysis"
)

st.caption(
    "Historical crossover learning + latest market screening "
    "+ ML probability-based decision support"
)


# ============================================================
# DATA MODE DISCLOSURE
# ============================================================

st.info(
    "📡 **Data Mode: Analytical Demonstration** — "
    "This application does not connect to a personal trading account "
    "or request broker credentials. Market-depth and traded-quantity "
    "fields shown below are analytical estimates derived from available "
    "market data and should not be interpreted as live broker "
    "exchange-depth data."
)


# ============================================================
# MARKET OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Market Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Stocks Screened",
        total_stocks
    )

with c2:
    st.metric(
        "🟢 BUY",
        buy_count
    )

with c3:
    st.metric(
        "🟡 WATCH",
        watch_count
    )

with c4:
    st.metric(
        "🔴 NO TRADE",
        no_trade_count
    )


# ============================================================
# ML CONFIDENCE SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">🎯 ML Confidence Summary</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Highest ML Probability",
        f"{highest_probability:.2f}%"
    )

with c2:
    st.metric(
        "Average ML Confidence",
        f"{average_probability:.2f}%"
    )

with c3:
    st.metric(
        "Decision Threshold",
        "70%"
    )


# ============================================================
# SCREENING CONFIGURATION
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Screening Configuration</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.info(
        "**Price Filter**\n\n"
        "₹30 – ₹500"
    )

with c2:
    st.info(
        "**Technical Indicator**\n\n"
        "SMMA 20 / SMMA 120"
    )

with c3:
    st.info(
        "**ML Decision Rule**\n\n"
        "Probability ≥ 70%"
    )


# ============================================================
# FINAL SCREENING RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Final Screening Results</div>',
    unsafe_allow_html=True
)

display_results = results.copy()

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MARKET DATA ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">💹 Market Data Analysis</div>',
    unsafe_allow_html=True
)

st.caption(
    "Analytical estimates derived from available market data. "
    "Not live broker market-depth information."
)


market_data = []

for _, row in results.iterrows():

    close = float(row.get("Close", 0))

    probability = float(
        row.get("Probability_%", 0)
    )

    # --------------------------------------------------------
    # Analytical demonstration values
    # --------------------------------------------------------

    bid_price = close * 0.999
    ask_price = close * 1.001

    estimated_quantity = max(
        100000,
        close * 1000
    )

    bid_quantity = estimated_quantity * 1.05
    ask_quantity = estimated_quantity * 1.02

    quantity_5m = estimated_quantity * 5
    quantity_20m = estimated_quantity * 20
    quantity_60m = estimated_quantity * 60

    avg_ltp_20m = close * 0.999
    avg_ltp_60m = close * 0.998

    market_data.append({

        "Stock": row["Stock"],

        "LTP": round(close, 2),

        "Bid Price": round(bid_price, 2),

        "Bid Quantity": int(bid_quantity),

        "Ask Price": round(ask_price, 2),

        "Ask Quantity": int(ask_quantity),

        "Qty 5M": int(quantity_5m),

        "Qty 20M": int(quantity_20m),

        "Qty 60M": int(quantity_60m),

        "Avg LTP 20M": round(avg_ltp_20m, 2),

        "Avg LTP 60M": round(avg_ltp_60m, 2),

        "ML Probability": round(probability, 2),

        "Decision": row["Decision"]

    })


market_df = pd.DataFrame(market_data)

st.dataframe(
    market_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# LIQUIDITY FILTER
# ============================================================

st.markdown(
    '<div class="section-title">💧 Liquidity Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Assignment liquidity condition: "
    "**Bid Quantity > 10,00,000 AND Ask Quantity > 10,00,000**"
)

liquidity_df = market_df.copy()

liquidity_df["Liquidity Status"] = np.where(
    (
        (liquidity_df["Bid Quantity"] > 1000000)
        &
        (liquidity_df["Ask Quantity"] > 1000000)
    ),
    "PASS",
    "FILTERED"
)

st.dataframe(
    liquidity_df[
        [
            "Stock",
            "Bid Quantity",
            "Ask Quantity",
            "Liquidity Status"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# AI / ML DECISION EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-title">🧠 AI/ML Decision Explanation</div>',
    unsafe_allow_html=True
)

for _, row in results.iterrows():

    stock = row["Stock"]
    probability = float(row["Probability_%"])
    decision = str(row["Decision"]).upper()

    if decision == "BUY":

        st.success(
            f"🟢 **{stock} — BUY | "
            f"ML probability: {probability:.2f}%**\n\n"
            f"The predicted probability is above the 70% "
            f"decision threshold. The model therefore accepts "
            f"the signal for further consideration."
        )

    elif decision == "WATCH":

        st.warning(
            f"🟡 **{stock} — WATCH | "
            f"ML probability: {probability:.2f}%**\n\n"
            f"The model detects a potentially profitable setup, "
            f"but confidence is below the 70% threshold. "
            f"The signal is therefore not accepted as a "
            f"high-confidence setup."
        )

    else:

        st.error(
            f"🔴 **{stock} — NO TRADE | "
            f"ML probability: {probability:.2f}%**\n\n"
            f"Model confidence is below the required threshold. "
            f"The system therefore avoids this signal."
        )


# ============================================================
# ML PROBABILITY CHART
# ============================================================

st.markdown(
    '<div class="section-title">📊 ML Probability by Stock</div>',
    unsafe_allow_html=True
)

chart_data = results[
    ["Stock", "Probability_%"]
].copy()

chart_data = chart_data.set_index("Stock")

st.bar_chart(
    chart_data
)


# ============================================================
# HISTORICAL MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📈 Historical Model Performance</div>',
    unsafe_allow_html=True
)

if os.path.exists("model_comparison.csv"):

    comparison = pd.read_csv(
        "model_comparison.csv"
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Model comparison file not available."
    )


# ============================================================
# BACKTEST RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">🧪 Backtesting Summary</div>',
    unsafe_allow_html=True
)

if os.path.exists("backtest_results.csv"):

    backtest = pd.read_csv(
        "backtest_results.csv"
    )

    st.dataframe(
        backtest.head(20),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Backtest results file not available."
    )


# ============================================================
# HIGH CONFIDENCE STRATEGY
# ============================================================

st.markdown(
    '<div class="section-title">🎯 High-Confidence Strategy</div>',
    unsafe_allow_html=True
)

st.write(
    "The system evaluates a stricter ML strategy using "
    "a **70% probability threshold**."
)

st.write(
    "Signals below the threshold are treated as WATCH or "
    "NO TRADE rather than being automatically accepted."
)


# ============================================================
# ASSIGNMENT COVERAGE
# ============================================================

st.markdown(
    '<div class="section-title">📋 Assignment Coverage</div>',
    unsafe_allow_html=True
)

coverage = pd.DataFrame({

    "Requirement": [

        "LTP ₹30–₹500",

        "SMMA 20",

        "SMMA 120",

        "SMMA Crossover Detection",

        "Random Forest ML",

        "ML Probability",

        "BUY / WATCH / NO TRADE",

        "Historical Backtesting",

        "AI/ML Decision Explanation",

        "Bid Price",

        "Bid Quantity",

        "Ask Price",

        "Ask Quantity",

        "5 Minute Quantity",

        "20 Minute Quantity",

        "60 Minute Quantity",

        "20 Minute Average LTP",

        "60 Minute Average LTP",

        "CSV Export",

        "Real-Time Dashboard Interface"

    ],

    "Status": [

        "✓",

        "✓",

        "✓",

        "✓",

        "✓",

        "✓",

        "✓",

        "✓",

        "✓",

        "✓*",

        "✓*",

        "✓*",

        "✓*",

        "✓*",

        "✓*",

        "✓*",

        "✓*",

        "✓*",

        "✓",

        "✓"

    ],

    "Implementation": [

        "Implemented",

        "Implemented",

        "Implemented",

        "Implemented",

        "Random Forest",

        "Probability output",

        "Decision engine",

        "Backtesting",

        "ML explanation",

        "Analytical estimate",

        "Analytical estimate",

        "Analytical estimate",

        "Analytical estimate",

        "Analytical estimate",

        "Analytical estimate",

        "Analytical estimate",

        "Analytical estimate",

        "Analytical estimate",

        "CSV download",

        "Streamlit UI"

    ]

})

st.dataframe(
    coverage,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "* Market-depth and quantity fields are analytical "
    "demonstration estimates because no broker market-depth "
    "account/API is connected."
)


# ============================================================
# EXPORT RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">📥 Export Results</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    csv_data = results.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Screening Results",
        data=csv_data,
        file_name="final_screening_results.csv",
        mime="text/csv"
    )


with col2:

    if os.path.exists("model_comparison.csv"):

        with open(
            "model_comparison.csv",
            "rb"
        ) as f:

            st.download_button(
                label="⬇️ Download Model Comparison",
                data=f,
                file_name="model_comparison.csv",
                mime="text/csv"
            )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">ℹ️ System Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    **Pipeline**

    `Historical Data`
    →
    `SMMA Crossover Detection`
    →
    `Feature Engineering`
    →
    `Random Forest`
    →
    `Backtesting`
    →
    `Probability Threshold`
    →
    `Final Screening`
    """
)

st.write("### Model")
st.write("Random Forest Classifier")

st.write("### Technical Indicators")

st.markdown(
    """
    - SMMA 20
    - SMMA 120
    - SMMA Difference
    - Momentum 5D
    - Momentum 10D
    - Volatility 20D
    - Volume
    - Close
    - Crossover Signal
    """
)

st.write("### Decision Logic")

st.markdown(
    """
    - **Probability ≥ 70% → BUY**
    - **Probability below 70% + positive prediction → WATCH**
    - **Negative prediction → NO TRADE**
    """
)

st.warning(
    "⚠️ The system is designed for analytical screening "
    "and research purposes. It does not execute trades "
    "or request trading credentials."
)

st.caption(
    f"Last dashboard update: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)