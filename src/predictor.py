import numpy as np
import torch
import pickle
from src.features import FEATURE_COLS, add_features
from src.fetcher import load_stock
from src.model import WINDOW_SIZE, load

def predict_next_day(ticker: str) -> dict:
    model, scaler, target_scaler = load(ticker)
    model.eval()

    df = load_stock(ticker)
    df = add_features(df)

    features = df[FEATURE_COLS].values[-WINDOW_SIZE:]
    features_scaled = scaler.transform(features)

    X = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(X).numpy()

    pred_pct_change = target_scaler.inverse_transform(pred_scaled)[0][0]
    last_price = df["Close"].iloc[-1]
    pred_price = last_price * (1 + pred_pct_change)
    change = pred_price - last_price
    change_pct = pred_pct_change * 100

    return {
        "ticker": ticker,
        "last_close": round(float(last_price), 2),
        "predicted_next": round(float(pred_price), 2),
        "change": round(float(change), 2),
        "change_pct": round(float(change_pct), 2),
        "last_date": str(df.index[-1].date()),
    }
    
def evaluate(ticker: str) -> dict:
    model, scaler, target_scaler = load(ticker)
    model.eval()

    df = load_stock(ticker)
    df = add_features(df)

    features = df[FEATURE_COLS].values
    target = df["target"].values

    features_scaled = scaler.transform(features)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(features_scaled)):
        X.append(features_scaled[i - WINDOW_SIZE:i])
        y.append(target[i])

    X = torch.tensor(np.array(X, dtype=np.float32))
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_test = X[split:]
    y_test = y[split:]

    with torch.no_grad():
        preds_scaled = model(X_test).numpy()

    preds_pct = target_scaler.inverse_transform(preds_scaled).flatten()

    # convert % change predictions back to prices for interpretable metrics
    closes = df["Close"].values[WINDOW_SIZE + split:]
    pred_prices = closes * (1 + preds_pct)
    actual_prices = closes * (1 + y_test)

    mae  = np.mean(np.abs(pred_prices - actual_prices))
    rmse = np.sqrt(np.mean((pred_prices - actual_prices) ** 2))
    mape = np.mean(np.abs((pred_prices - actual_prices) / actual_prices)) * 100

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
        "predictions": pred_prices,
        "actuals": actual_prices
    }