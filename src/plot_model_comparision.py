import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(df, metric="MAE", save_as=None):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Ticker", y=metric, hue="Model", palette="Set2")
    plt.title(f"{metric} Comparison Across Models")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("results/predictive_metrics.csv")

    plot_model_comparison(df, metric="MAE", save_as="results/compare_mae.png")
    plot_model_comparison(df, metric="R2", save_as="results/compare_r2.png")
