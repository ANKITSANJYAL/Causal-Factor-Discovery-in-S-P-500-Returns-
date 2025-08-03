import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Step 1: Flatten multi-index columns ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]

    if 'Date_' in df.columns[0]:
        df = df.rename(columns={df.columns[0]: 'Date'})

    # --- Step 2: Melt wide table into long format ---
    tickers = list(set(col.split('_')[1] for col in df.columns if '_' in col and col.split('_')[1] != 'Date'))
    print(f"Detected tickers: {tickers}")

    long_dfs = []
    for ticker in tickers:
        try:
            cols = [f'{col}_{ticker}' for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
            if not all(c in df.columns for c in cols):
                print(f"⚠️ Skipping {ticker} — some columns missing")
                continue

            sub = df[["Date"] + cols].copy()
            sub.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            sub["Ticker"] = ticker

            # 🚨 Drop rows where Close is NaN (these are leftover artifacts)
            sub = sub.dropna(subset=["Close"])

            long_dfs.append(sub)
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            continue

    df = pd.concat(long_dfs, ignore_index=True)
    df = df.sort_values(['Ticker', 'Date'])

    print("📊 Sample reshaped data:")
    print(df.head(10))
    print("📊 Close column null ratio:", df["Close"].isna().mean())
    print("📊 Volume column null ratio:", df["Volume"].isna().mean())

    # --- Step 3: Ensure numerics ---
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # --- Step 4: Feature engineering ---
    df['Return'] = df.groupby("Ticker")["Close"].pct_change(fill_method=None)
    df['NextReturn'] = df.groupby("Ticker")["Return"].shift(-1)
    df["Momentum_3"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change(3, fill_method=None))
    df["Volatility_3"] = df.groupby("Ticker")["Return"].transform(lambda x: x.rolling(3).std())
    df["MA_3"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(3).mean())
    df["MA_5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5).mean())
    df["VolumeLog"] = df["Volume"].apply(lambda x: np.log1p(x))

    # --- Step 5: Drop rows with missing features ---
    required = ["Return", "NextReturn", "Momentum_3", "Volatility_3", "MA_3", "MA_5", "VolumeLog"]

    print("\n🧪 Missing values per required feature:")
    print(df[required].isna().mean().sort_values(ascending=False))

    df["complete_row"] = df[required].notna().all(axis=1)
    print("✅ Valid rows with all required features:", df["complete_row"].sum(), "/", len(df))

    df[~df["complete_row"]].to_csv("data/debug_incomplete_rows.csv", index=False)

    df = df[df["complete_row"]].drop(columns=["complete_row"])
    df["Target"] = df["NextReturn"]

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/sp500_data.csv")  # Your wide-format file
    df = engineer_features(df)
    df.to_csv("data/sp500_data_features.csv", index=False)
