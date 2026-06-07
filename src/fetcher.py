import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_stock(ticker: str, period: str = "5y") -> pd.DataFrame:
    print(f"Fetching {ticker} ({period})...")
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.dropna(inplace=True)

    # save to disk so we don't re-fetch every run
    Path("data").mkdir(exist_ok=True)
    df.to_csv(f"data/{ticker}.csv")
    print(f"Saved {len(df)} rows to data/{ticker}.csv")
    return df

def load_stock(ticker: str) -> pd.DataFrame:
    path = f"data/{ticker}.csv"
    if Path(path).exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return fetch_stock(ticker)