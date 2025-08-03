from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import pandas as pd
import joblib
import os

def run_dml(data_path="data/sp500_data_features.csv", ticker="AAPL"):
    df = pd.read_csv(data_path)
    df = df[df["Ticker"] == ticker].copy()
    df = df.sort_values("Date")

    # Select treatment, target, and controls
    Y = df["Target"].values
    X = df[["Volatility_3", "MA_3", "MA_5", "VolumeLog"]].values 
    T = df["Momentum_3"].values


    # Chronological split: first 80% train, last 20% test
    split_idx = int(0.8 * len(df))
    X_train, X_test = X[:split_idx], X[split_idx:]
    T_train, T_test = T[:split_idx], T[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # Define DML model
    model = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
        model_t=LassoCV(),
        discrete_treatment=False,
        random_state=42
    )
    model.fit(Y_train, T_train, X=X_train)

    # Compute causal effects
    cate = model.effect(X_test)
    ate = model.ate(X_test)

    # Save model
    os.makedirs("results", exist_ok=True)
    joblib.dump(model, f"results/dml_model_{ticker}.pkl")

    # Log output
    with open("training_log.txt", "a") as log:
        log.write(f"Ticker: {ticker} | ATE: {ate:.6f} | N Test: {len(X_test)}\n")

    return model, X_test, cate

if __name__ == "__main__":
    for ticker in ["AAPL", "MSFT", "JPM", "XOM", "AMZN"]:
        print(f"Running DML for {ticker}...")
        run_dml(ticker=ticker)
