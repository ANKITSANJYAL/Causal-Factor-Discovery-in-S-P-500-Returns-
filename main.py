from src.data_loader import fetch_sp500_data
from src.feature_engineer import engineer_features
from src.dml_model import run_dml
import pandas as pd
from datetime import datetime
import os

def pipeline():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tickers = ["AAPL", "MSFT", "JPM", "XOM", "AMZN"]
    end = datetime.today().strftime('%Y-%m-%d')

    print(f"📈 Pulling stock data from 2019-01-01 to {end}...")
    raw = fetch_sp500_data(tickers, start="2019-01-01", end=end)
    raw.to_csv("data/sp500_data.csv", index=False)

    print("Raw data shape:", raw.shape)
    print("Raw data columns:", raw.columns.tolist())

    print("🛠️ Engineering features...")
    df = engineer_features(raw)
    df.to_csv("data/sp500_data_features.csv", index=False)

    print("📊 Running Double Machine Learning for each ticker:")
    for ticker in tickers:
        print(f"--- {ticker} ---")
        run_dml(data_path="data/sp500_data_features.csv", ticker=ticker)

    print("✅ Pipeline complete.")

if __name__ == "__main__":
    pipeline()
