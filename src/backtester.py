import numpy as np
import pandas as pd
import torch
import pickle
from src.features import FEATURE_COLS, add_features
from src.fetcher import load_stock
from src.model import WINDOW_SIZE, load

def run_backtest(ticker: str, starting_cash: float = 10000.0) -> dict:
    model, scaler, target_scaler = load(ticker)
    model.eval()

    df = load_stock(ticker)
    df = add_features(df)

    features = df[FEATURE_COLS].values
    closes   = df["Close"].values
    dates    = df.index

    features_scaled = scaler.transform(features)

    # build all windows
    X = []
    for i in range(WINDOW_SIZE, len(features_scaled)):
        X.append(features_scaled[i - WINDOW_SIZE:i])
    X = torch.tensor(np.array(X, dtype=np.float32))

    with torch.no_grad():
        preds_scaled = model(X).numpy()

    preds_pct = target_scaler.inverse_transform(preds_scaled).flatten()

    # use test set only — never backtest on training data
    split = int(len(X) * 0.8)
    X_dates  = dates[WINDOW_SIZE:]
    test_dates   = X_dates[split:]
    test_closes  = closes[WINDOW_SIZE:][split:]
    test_preds   = preds_pct[split:]

    # simulate trading
    cash        = starting_cash
    shares      = 0
    portfolio   = []
    signals     = []
    trade_log   = []

    for i in range(len(test_closes)):
        price     = test_closes[i]
        pred_pct  = test_preds[i]
        signal = "HOLD"
        if pred_pct > 0.0005:
            signal = "BUY"
        elif pred_pct < -0.003:
            signal = "SELL"
        signals.append(signal)

        if signal == "BUY" and cash >= price:
            # buy as many shares as we can afford
            shares_to_buy = int(cash // price)
            shares += shares_to_buy
            cash   -= shares_to_buy * price
            trade_log.append({
                "date": test_dates[i],
                "action": "BUY",
                "price": round(float(price), 2),
                "shares": shares_to_buy
            })

        elif signal == "SELL" and shares > 0:
            # sell all shares
            cash  += shares * price
            trade_log.append({
                "date": test_dates[i],
                "action": "SELL",
                "price": round(float(price), 2),
                "shares": shares
            })
            shares = 0

        portfolio_value = cash + shares * price
        portfolio.append(portfolio_value)

    # final liquidation
    final_price = test_closes[-1]
    final_value = cash + shares * final_price

    # buy and hold benchmark
    shares_bh       = int(starting_cash // test_closes[0])
    buy_hold_value  = shares_bh * final_price + (starting_cash - shares_bh * test_closes[0])
    buy_hold_return = ((buy_hold_value - starting_cash) / starting_cash) * 100

    model_return = ((final_value - starting_cash) / starting_cash) * 100

    return {
        "dates":          test_dates,
        "portfolio":      portfolio,
        "signals":        signals,
        "trade_log":      pd.DataFrame(trade_log),
        "final_value":    round(final_value, 2),
        "model_return":   round(model_return, 2),
        "buy_hold_value": round(buy_hold_value, 2),
        "buy_hold_return":round(buy_hold_return, 2),
        "n_trades":       len(trade_log),
        "starting_cash":  starting_cash
    }