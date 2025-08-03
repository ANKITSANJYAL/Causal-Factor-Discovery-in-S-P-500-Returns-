import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────── #
#         Evaluation Metrics       #
# ──────────────────────────────── #
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

# ──────────────────────────────── #
#         LSTM Architectures       #
# ──────────────────────────────── #
def build_lstm_model(input_shape, version="base"):
    model = Sequential()

    if version == "base":
        model.add(LSTM(64, input_shape=input_shape))
    elif version == "deep":
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(32))
    elif version == "wide":
        model.add(LSTM(128, input_shape=input_shape))
    elif version == "deep_wide":
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(64))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# ──────────────────────────────── #
#       Train and Record Loss      #
# ──────────────────────────────── #
def train_lstm(X_train, X_test, y_train, y_test, version, epochs=100):
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), version)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )

    y_pred = model.predict(X_test).flatten()
    metrics = evaluate(y_test, y_pred)

    return metrics, history

# ──────────────────────────────── #
#        Plot Loss Curves          #
# ──────────────────────────────── #
def plot_loss(history, version, output_dir):
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{version.upper()} - Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"loss_curve_{version}.png"))
    plt.close()

# ──────────────────────────────── #
#            Main Logic            #
# ──────────────────────────────── #
def run_ablation(df, ticker, output_dir):
    df = df[df["Ticker"] == ticker].copy()
    df = df.sort_values("Date")

    features = ["Momentum_3", "Volatility_3", "MA_3", "MA_5", "VolumeLog"]
    X = df[features].values
    y = df["Target"].values

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    versions = ["base", "deep", "wide", "deep_wide"]
    all_metrics = []

    for version in versions:
        print(f"📊 Training {version.upper()} model...")
        metrics, history = train_lstm(X_train_scaled, X_test_scaled, y_train, y_test, version)
        plot_loss(history, version, output_dir)
        all_metrics.append({"Model": version, **metrics})

    return all_metrics

if __name__ == "__main__":
    os.makedirs("results/ablation", exist_ok=True)

    df = pd.read_csv("data/sp500_data_features.csv")
    ticker = "AAPL"  # Change this to any one ticker for ablation

    print(f"🚀 Running LSTM ablation for {ticker}")
    ablation_metrics = run_ablation(df, ticker, "results/ablation")
    pd.DataFrame(ablation_metrics).to_csv("results/ablation/ablation_metrics.csv", index=False)
    print("✅ Ablation study completed and results saved.")
