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
from src.backtester import run_backtest

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("Stock price predictor")

# --- Sidebar ---
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker symbol", value="AAPL").upper()
starting_cash = st.sidebar.number_input("Starting cash ($)", value=10000, step=1000)
refresh = st.sidebar.button("Refresh data")

if refresh:
    fetch_stock(ticker)
    st.sidebar.success(f"Refreshed {ticker} data")

# --- Load ---
try:
    result  = predict_next_day(ticker)
    metrics = evaluate(ticker)
except FileNotFoundError:
    st.error(f"No trained model found for {ticker}. Run: python train.py {ticker}")
    st.stop()

# --- Tabs ---
tab1, tab2 = st.tabs(["Prediction", "Backtest"])

# ── Tab 1: Prediction ──────────────────────────────────────────
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last close",     f"${result['last_close']}")
    col2.metric("Predicted next", f"${result['predicted_next']}",
                delta=f"{result['change_pct']:+.2f}%")
    col3.metric("MAPE",           f"{metrics['mape']}%")
    col4.metric("MAE",            f"${metrics['mae']}")

    st.divider()
    st.subheader("Predicted vs actual — test set")

    df     = load_stock(ticker)
    df     = add_features(df)
    actuals = metrics["actuals"]
    preds   = metrics["predictions"]
    dates   = df.index[-len(actuals):]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, actuals, label="Actual",    color="#1D9E75", linewidth=1.5)
    ax.plot(dates, preds,   label="Predicted", color="#7F77DD", linewidth=1.5, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("View raw data"):
        st.dataframe(df[["Close", "ma_7", "ma_21", "rsi", "volume_ratio"]].tail(30))

# ── Tab 2: Backtest ────────────────────────────────────────────
with tab2:
    st.subheader("Trading simulation")

    with st.spinner("Running backtest..."):
        bt = run_backtest(ticker, starting_cash=float(starting_cash))

    # summary cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final portfolio",   f"${bt['final_value']:,.2f}",
                delta=f"{bt['model_return']:+.2f}%")
    col2.metric("Buy & hold value",  f"${bt['buy_hold_value']:,.2f}",
                delta=f"{bt['buy_hold_return']:+.2f}%")
    col3.metric("Total trades",      bt["n_trades"])
    col4.metric("Starting cash",     f"${bt['starting_cash']:,.0f}")

    st.divider()

    # portfolio vs buy & hold chart
    st.subheader("Portfolio value vs buy & hold")

    df_bt = load_stock(ticker)
    df_bt = add_features(df_bt)

    # buy and hold portfolio line
    split        = int(len(df_bt) * 0.8)
    test_closes  = df_bt["Close"].values[-len(bt["portfolio"]):]
    shares_bh    = int(starting_cash // test_closes[0])
    leftover     = starting_cash - shares_bh * test_closes[0]
    buy_hold_series = shares_bh * test_closes + leftover

    dates = bt["dates"][:len(bt["portfolio"])]

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(dates, bt["portfolio"],                   label="Model strategy", color="#7F77DD", linewidth=1.5)
    ax2.plot(dates, buy_hold_series[:len(bt["portfolio"])], label="Buy & hold",     color="#1D9E75", linewidth=1.5, linestyle="--")
    ax2.axhline(y=starting_cash, color="#888", linewidth=0.8, linestyle=":")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax2.set_ylabel("Portfolio value ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    # buy/sell markers on price chart
    st.subheader("Buy / sell signals on price")

    signals = bt["signals"]
    prices  = df_bt["Close"].values[-len(signals):]
    sig_dates = dates[:len(signals)]

    buy_dates  = [sig_dates[i] for i, s in enumerate(signals) if s == "BUY"]
    buy_prices = [prices[i]    for i, s in enumerate(signals) if s == "BUY"]