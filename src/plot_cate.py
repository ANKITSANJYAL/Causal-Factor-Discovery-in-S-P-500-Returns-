import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cate(data_path, model_path, ticker):
    # Load model
    model = joblib.load(model_path)
    
    # Load data and filter
    df = pd.read_csv(data_path)
    df = df[df["Ticker"] == ticker].copy()
    df = df.sort_values("Date")

    # Align feature names with your model
    X = df[["Volatility_3", "MA_3", "MA_5", "VolumeLog"]].values

    # Use model to estimate CATE
    cate = model.effect(X)

    # Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(cate, bins=30, kde=True, color="skyblue")
    plt.title(f"CATE Distribution for {ticker}")
    plt.xlabel("Estimated Treatment Effect (Momentum → Return)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/cate_{ticker}.png")
    plt.show()

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "AMZN"]
    for ticker in tickers:
        plot_cate(
            data_path="data/sp500_data_features.csv",
            model_path=f"results/dml_model_{ticker}.pkl",
            ticker=ticker
        )
