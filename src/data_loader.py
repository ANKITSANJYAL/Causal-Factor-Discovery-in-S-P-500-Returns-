import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_sp500_data(tickers, start="2019-01-01", end=None):
    print("⚠️ Running NEW fetch_sp500_data (per-ticker loop)")

    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    all_data = []
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df = df.reset_index()
        df["Ticker"] = ticker
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)
    print("Final shape after stacking:", df_all.shape)
    print("Columns:", df_all.columns.tolist())
    return df_all

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "AMZN"]
    df = fetch_sp500_data(tickers)
    df.to_csv("data/sp500_data.csv", index=False)
