# Causal Factor Discovery in S&P 500 Returns

This repository implements an end-to-end pipeline for uncovering causal relationships in stock returns using **Double Machine Learning (DML)**. It also benchmarks those causal effects against traditional and deep learning predictive models, with additional architectural ablation analysis for LSTM networks.

---

## 🧠 Project Objective
- Discover if short-term momentum has a causal effect on next-day stock returns
- Use DML to estimate Average Treatment Effect (ATE) and Conditional Average Treatment Effect (CATE)
- Benchmark with predictive models: Linear Regression, Random Forest, LSTM
- Perform LSTM ablation study (base, deep, wide, deep-wide)

---

## 🗂️ Project Structure

```
├── data/
│   ├── sp500_data.csv                  # Raw price data (from yfinance)
│   ├── sp500_data_features.csv         # Engineered features
│   └── debug_incomplete_rows.csv       # Optional debugging artifact
│
├── notebooks/                         # (Optional) Jupyter notebooks
│
├── results/                           # Output visualizations and metrics
│   ├── cate_{TICKER}.png              # CATE plots for each ticker
│   ├── loss_curve_{variant}.png       # LSTM training curves
│   ├── predictive_metrics.csv         # MAE, MSE, R2 for each model/ticker
│
├── src/
│   ├── data_loader.py                 # Fetch S&P 500 data from yfinance
│   ├── feature_engineer.py            # Create momentum, volatility, MA features
│   ├── dml_model.py                   # DML implementation per ticker
│   ├── predictive_models.py           # Train LR, RF, and LSTM
│   ├── lstm_abilation.py              # Deeper/wider LSTM experiments
│   ├── plot_cate.py                   # Visualize CATE distributions
│   ├── plot_model_comparision.py      # Generate bar/heatmap comparisons
│
├── main.py                            # Main pipeline: fetch → engineer → DML
├── requirements.txt                   # Python dependencies
├── README.md                          # You're here!
├── proposal.txt                       # Research planning document
├── training_log.txt                   # Logs from model training
```

---

## ⚙️ How to Run the Pipeline

### 1. Clone the repository
```bash
git clone https://github.com/your-username/causal-factor-discovery.git
cd causal-factor-discovery
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python main.py
```
This script will:
- Download data from `2019-01-01` to today for 5 tickers (AAPL, MSFT, JPM, AMZN, XOM)
- Generate engineered features
- Estimate causal effects using DML
- Store raw data, features, and visualizations in `/data` and `/results`

---

## 📊 Outputs

### 📈 Causal Estimation (via DML)
- ATE values printed to console per ticker
- `results/cate_{TICKER}.png`: CATE visualizations

### 🔍 Predictive Benchmarking
- Linear Regression
- Random Forest
- LSTM (2-layer, early stopping)
- Metrics: MAE, MSE, R² → saved to `results/predictive_metrics.csv`

### 🧪 LSTM Ablation Study
- Variants: base, deep, wide, deep-wide
- Training/validation losses plotted in `results/loss_curve_*.png`
- Metrics printed + stored in `ablation_metrics.csv`

---

## 🧪 Models Used

### Causal:
- Double Machine Learning (with cross-fitting and orthogonalization)

### Predictive:
- Linear Regression (OLS)
- Random Forest Regressor
- Long Short-Term Memory (LSTM)

---

## 🧬 Feature Set
- `Return`: Daily percentage return
- `Momentum_3`: 3-day return difference
- `Volatility_3`: 3-day rolling std deviation
- `MA_3`, `MA_5`: 3-day and 5-day moving averages
- `VolumeLog`: Log-transformed volume
- `NextReturn`: Target variable (next-day return)

---

## 📚 References
- Chernozhukov et al. (2018). *Double Machine Learning for Treatment and Structural Parameters*
- Gu, Kelly & Xiu (2020). *Empirical Asset Pricing via Machine Learning*
- Athey & Imbens (2015). *Causal Inference for Statistics, Social and Biomedical Sciences*


---

## 🧠 Author
**Ankit Sanjyal**  
MSc Data Science | Fordham University  
🔗 [ankitsanjyal.github.io](https://ankitsanjyal.github.io/) | [GitHub](https://github.com/ANKITSANJYAL)
