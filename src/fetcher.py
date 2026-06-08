import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_stock(ticker: str, period: str = "10y") -> pd.DataFrame:
    print(f"Fetching {ticker} ({period})...")
    df = yf.download(ticker, period=period, auto_adjust=True)

    # flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)

    Path("data").mkdir(exist_ok=True)
    df.to_csv(f"data/{ticker}.csv")
    print(f"Saved {len(df)} rows to data/{ticker}.csv")
    return df

def load_stock(ticker: str) -> pd.DataFrame:
    path = f"data/{ticker}.csv"
    if Path(path).exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return fetch_stock(ticker)