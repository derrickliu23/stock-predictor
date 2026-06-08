import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.fetcher import load_stock, fetch_stock
from src.features import add_features
from src.predictor import predict_next_day, evaluate

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("Stock price predictor")

# --- Sidebar ---
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker symbol", value="AAPL").upper()
refresh = st.sidebar.button("Refresh data")

if refresh:
    fetch_stock(ticker)
    st.sidebar.success(f"Refreshed {ticker} data")

# --- Load and predict ---
try:
    result = predict_next_day(ticker)
    metrics = evaluate(ticker)
except FileNotFoundError:
    st.error(f"No trained model found for {ticker}. Run: python train.py {ticker}")
    st.stop()

# --- Metric cards ---
col1, col2, col3, col4 = st.columns(4)

col1.metric("Last close", f"${result['last_close']}")
col2.metric(
    "Predicted next",
    f"${result['predicted_next']}",
    delta=f"{result['change_pct']:+.2f}%"
)
col3.metric("MAPE", f"{metrics['mape']}%")
col4.metric("MAE", f"${metrics['mae']}")

st.divider()

# --- Chart ---
st.subheader("Predicted vs actual — test set")

df = load_stock(ticker)
df = add_features(df)

actuals = metrics["actuals"]
preds   = metrics["predictions"]

dates = df.index[-len(actuals):]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates, actuals, label="Actual", color="#1D9E75", linewidth=1.5)
ax.plot(dates, preds,   label="Predicted", color="#7F77DD", linewidth=1.5, linestyle="--")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

st.divider()

# --- Raw data ---
with st.expander("View raw data"):
    st.dataframe(df[["Close", "ma_7", "ma_21", "rsi", "volume_ratio"]].tail(30))