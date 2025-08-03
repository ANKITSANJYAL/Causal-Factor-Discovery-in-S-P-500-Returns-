import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def run_lstm(X_train, X_test, y_train, y_test):
    # reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[EarlyStopping(patience=5)], verbose=0)

    y_pred = model.predict(X_test).flatten()
    return evaluate(y_test, y_pred), y_pred

def run_models(df, ticker):
    df = df[df["Ticker"] == ticker].copy()
    df = df.sort_values("Date")

    features = ["Momentum_3", "Volatility_3", "MA_3", "MA_5", "VolumeLog"]
    X = df[features].values
    y = df["Target"].values

    # time series split (chronological)
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results.append({
        "Ticker": ticker,
        "Model": "LinearRegression",
        **evaluate(y_test, y_pred_lr)
    })

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results.append({
        "Ticker": ticker,
        "Model": "RandomForest",
        **evaluate(y_test, y_pred_rf)
    })

    # LSTM
    lstm_result, _ = run_lstm(X_train_scaled, X_test_scaled, y_train, y_test)
    results.append({
        "Ticker": ticker,
        "Model": "LSTM",
        **lstm_result
    })

    return results

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv("data/sp500_data_features.csv")

    tickers = df["Ticker"].unique().tolist()
    all_results = []

    for ticker in tickers:
        print(f"🔍 Running predictive models for {ticker}")
        results = run_models(df, ticker)
        all_results.extend(results)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results/predictive_metrics.csv", index=False)
    print("✅ Saved results to results/predictive_metrics.csv")
