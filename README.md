# Stock Price Predictor

Predicts next-day stock prices using an LSTM neural network trained on historical price and volume data. Includes a Streamlit dashboard for visualizing predictions against actual prices.

## How it works

1. **Fetch** — pulls up to 10 years of historical OHLCV data from Yahoo Finance
2. **Engineer features** — computes moving averages, RSI, momentum, and volatility indicators
3. **Train** — trains a two-layer LSTM on 30-day sliding windows, predicting next-day % price change
4. **Predict** — loads the trained model and outputs a next-day price prediction with accuracy metrics
5. **Visualize** — Streamlit dashboard shows the prediction, key metrics, and predicted vs actual chart

## Setup

**1. Clone the repo**
```bash
git clone git@github.com:your-username/stock-predictor.git
cd stock-predictor
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

**Train a model for any ticker:**
```bash
python train.py AAPL
python train.py MSFT
python train.py TSLA
```

**Get a next-day prediction:**
```bash
python train.py AAPL predict
```

Output:
```
AAPL — 2026-06-05
Last close:      $307.34
Predicted next:  $307.14
Expected change: -0.20 (-0.06%)
```

**Evaluate model accuracy on the test set:**
```bash
python train.py AAPL evaluate
```

Output:
```
Test set evaluation:
MAE:  $2.73   (avg dollar error)
RMSE: $3.93   (penalizes big misses)
MAPE: 1.17%   (avg % error)
```

**Launch the dashboard:**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Project structure

```
stock-predictor/
├── data/                  # cached stock CSVs (auto-generated, not committed)
├── models/                # saved model weights and scalers (not committed)
├── src/
│   ├── fetcher.py         # pulls and caches data from Yahoo Finance
│   ├── features.py        # engineers technical indicators
│   ├── model.py           # LSTM architecture, training loop, save/load
│   ├── predictor.py       # next-day prediction and test set evaluation
│   └── evaluate.py        # accuracy metrics
├── app.py                 # Streamlit dashboard
├── train.py               # CLI for training, predicting, and evaluating
└── requirements.txt
```

## Features engineered

| Feature | Description |
|---|---|
| `ma_7 / ma_21 / ma_50` | Moving averages over 1 week, 1 month, 2 months |
| `roc_7 / roc_21` | Rate of change (momentum) over 7 and 21 days |
| `volatility_7 / volatility_21` | Rolling standard deviation of price |
| `volume_ratio` | Today's volume vs 7-day average — spike detector |
| `rsi` | Relative Strength Index (14-day) — overbought/oversold signal |

## Model architecture

- Two stacked LSTM layers (32 → 16 units) with dropout
- Trained on 30-day sliding windows
- Predicts next-day **% price change** (not raw price) to avoid trend bias
- Early stopping with patience of 10 epochs
- Trained with Adam optimizer, MSE loss

## Results (AAPL, 10 years of data)

| Metric | Value |
|---|---|
| MAE | $2.73 |
| RMSE | $3.93 |
| MAPE | 1.17% |

## Stack

- [PyTorch](https://pytorch.org) — LSTM model
- [yfinance](https://github.com/ranaroussi/yfinance) — free stock data
- [scikit-learn](https://scikit-learn.org) — data preprocessing
- [Streamlit](https://streamlit.io) — dashboard
- [pandas](https://pandas.pydata.org) / [numpy](https://numpy.org) — data wrangling
- [matplotlib](https://matplotlib.org) — charts

## Disclaimer

This project is for educational purposes only. Predictions are based solely on historical price and volume data and should not be used to make real investment decisions.