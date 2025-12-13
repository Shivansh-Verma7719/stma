import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Set visual style
sns.set_theme(style="whitegrid")


def analyze_single_stock(filepath, output_folder):
    stock_name = os.path.basename(filepath).replace(".csv", "")
    print(f"--- Starting Deep Dive Analysis for: {stock_name} ---")

    # 1. LOAD DATA (Must be from fin_process folder with indicators)
    df = pd.read_csv(filepath)

    # Validation
    required_cols = ["bias_index", "intc", "v", "rel_vol", "smooth_velocity", "psi"]
    if not all(c in df.columns for c in required_cols):
        print(
            "Error: Input file missing required columns. Make sure it comes from 'fin_process'."
        )
        return

    # Ensure time is datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # --- METHOD 1: REGRESSION (Hypocrisy Score) ---
    print("Running Regression...")
    df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))
    temp_reg = df.dropna(subset=["log_returns", "bias_index"]).copy()

    if len(temp_reg) > 10:
        X = temp_reg[["bias_index"]]
        y = temp_reg["log_returns"]
        model = LinearRegression().fit(X, y)
        temp_reg["expected_return"] = model.predict(X)
        temp_reg["reg_residual"] = temp_reg["log_returns"] - temp_reg["expected_return"]
        temp_reg["bias_score_regression"] = temp_reg["reg_residual"].abs()

        # Merge back
        df.loc[temp_reg.index, "bias_score_regression"] = temp_reg[
            "bias_score_regression"
        ]

    # --- METHOD 2: STATISTICAL (Silent Shocks) ---
    print("Running Statistical Checks...")
    mean_ret = df["log_returns"].mean()
    std_ret = df["log_returns"].std()
    df["z_score_price"] = (
        (df["log_returns"] - mean_ret) / std_ret if std_ret != 0 else 0
    )

    df["is_volume_shock"] = df["rel_vol"] > 3.0
    df["silent_anomaly"] = False

    # Shock + Neutral Bias
    shock = (df["z_score_price"].abs() > 3) | (df["is_volume_shock"])
    neutral = df["bias_index"].abs() < 0.1
    df.loc[shock & neutral, "silent_anomaly"] = True

    # --- METHOD 3: ISOLATION FOREST (Context Anomalies) ---
    print("Running AI Anomaly Detection...")
    features = ["bias_index", "smooth_velocity", "psi", "macd", "ImpulseHisto"]
    ml_data = df[features].dropna()

    if len(ml_data) > 50:
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(ml_data)
        iso = IsolationForest(contamination=0.05, random_state=42)
        ml_data["iso_forest_outlier"] = iso.fit_predict(scaled_X)
        df.loc[ml_data.index, "iso_forest_outlier"] = ml_data["iso_forest_outlier"]

    # --- METHOD 4: DBSCAN (Rational Clusters) ---
    print("Running Clustering...")
    cluster_data = df[["bias_index", "log_returns"]].dropna()
    if len(cluster_data) > 50:
        scaler = StandardScaler()
        scaled_C = scaler.fit_transform(cluster_data)
        db = DBSCAN(eps=0.5, min_samples=10)
        cluster_data["dbscan_cluster"] = db.fit_predict(scaled_C)
        df.loc[cluster_data.index, "dbscan_cluster"] = cluster_data["dbscan_cluster"]

    # --- GENERATE PLOTS ---
    print("Generating Plots...")
    stock_out_dir = os.path.join(output_folder, stock_name)
    if not os.path.exists(stock_out_dir):
        os.makedirs(stock_out_dir)

    # Plot 1: Bias Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["bias_score_regression"], bins=30, kde=True, color="purple")
    plt.title(f"{stock_name}: Distribution of Market-News Divergence")
    plt.xlabel("Magnitude of Deviation (Residual)")
    plt.savefig(os.path.join(stock_out_dir, "1_bias_dist.png"))
    plt.close()

    # Plot 2: Scatter (News vs Price)
    plt.figure(figsize=(10, 6))
    plot_data = df[df["bias_index"].abs() > 0.05]
    sns.scatterplot(
        x="bias_index",
        y="log_returns",
        hue="iso_forest_outlier",
        data=plot_data,
        palette={1: "blue", -1: "red"},
        alpha=0.7,
    )
    plt.title(f"{stock_name}: Sentiment vs Return (Red = Anomalies)")
    plt.axhline(0, color="grey", ls="--")
    plt.axvline(0, color="grey", ls="--")
    plt.savefig(os.path.join(stock_out_dir, "2_sentiment_vs_price.png"))
    plt.close()

    # Plot 3: Silent Shocks Timeline
    df["YearMonth"] = df["time"].dt.to_period("M")
    silent_counts = df.groupby("YearMonth")["silent_anomaly"].sum()
    silent_counts.index = silent_counts.index.astype(str)

    plt.figure(figsize=(12, 6))
    silent_counts.plot(kind="bar", color="orange", width=0.8)
    plt.title(f"{stock_name}: Timeline of Silent Shocks (Possible Leaks)")

    # Fix X-axis labels
    n = len(silent_counts)
    step = max(1, n // 15)
    plt.xticks(
        ticks=range(0, n, step),
        labels=silent_counts.index[::step],
        rotation=45,
        ha="right",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(stock_out_dir, "3_silent_shocks.png"))
    plt.close()

    # --- GENERATE TEXT REPORT ---
    report_path = os.path.join(stock_out_dir, f"{stock_name}_REPORT.txt")

    avg_bias = df["bias_score_regression"].mean()
    total_silent = df["silent_anomaly"].sum()

    # Find top 3 weirdest days
    weirdest_days = (
        df[df["iso_forest_outlier"] == -1]
        .sort_values("bias_score_regression", ascending=False)
        .head(3)
    )

    with open(report_path, "w") as f:
        f.write(f"=== DEEP DIVE REPORT: {stock_name} ===\n\n")
        f.write(f"1. SENSITIVITY SCORE (Residual): {avg_bias:.5f}\n")
        f.write("   (Higher = Stock acts more irrationally/ignores news)\n\n")

        f.write(f"2. SILENT SHOCK COUNT: {total_silent}\n")
        f.write("   (Number of times price exploded with NO news)\n\n")

        f.write("3. TOP 3 ANOMALIES (The 'Lies'):\n")
        for i, row in weirdest_days.iterrows():
            f.write(f"   Date: {row['time'].date()}\n")
            f.write(f"   News Sentiment: {row['bias_index']:.2f}\n")
            f.write(f"   Price Move: {row['log_returns']:.2%}\n")
            f.write(f"   Reason: High Divergence + AI Outlier\n")
            f.write("   -----------------\n")

    print(f"Done! Results saved to: {stock_out_dir}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Point this to the file you want to analyze
    # IMPORTANT: Use a file from 'fin_process' so it has indicators!
    TARGET_FILE = (
        "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/AAPL.csv"
    )

    # 2. Where to save the folder
    OUTPUT_DIR = (
        "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/single_stock_analysis"
    )

    if os.path.exists(TARGET_FILE):
        analyze_single_stock(TARGET_FILE, OUTPUT_DIR)
    else:
        print(f"File not found: {TARGET_FILE}")
