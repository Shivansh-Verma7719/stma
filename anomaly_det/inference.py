import pandas as pd
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import time

# Set visual style for plots
sns.set_theme(style="whitegrid")

# --- GLOBAL CONFIG ---
# Define paths based on your directory structure
BASE_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/"
MAX_WORKERS = 50  # Number of threads for file reading


def process_single_stock_merge(f1):
    """
    Worker function: Reads all 4 method files for a SINGLE stock and merges them.
    Returns: The merged DataFrame for that stock (or None if failed).
    """
    try:
        # Define sub-folder paths
        path_m2 = os.path.join(BASE_DIR, "statistical")
        path_m3 = os.path.join(BASE_DIR, "forest")
        path_m4 = os.path.join(BASE_DIR, "clustering")

        # 1. Identify the core filename (e.g., "AAPL.csv")
        basename = os.path.basename(f1)

        # 2. Construct paths for other methods using the exact same filename
        f2 = os.path.join(path_m2, basename)
        f3 = os.path.join(path_m3, basename)
        f4 = os.path.join(path_m4, basename)

        # 3. Load Method 1 (Master) - Regression
        df1 = pd.read_csv(f1)

        # 4. Merge Method 2 (Statistical)
        if os.path.exists(f2):
            df2 = pd.read_csv(f2)
            cols_m2 = ["time", "z_score_price", "is_volume_shock", "silent_anomaly"]
            df1 = pd.merge(df1, df2[cols_m2], on="time", how="left")

        # 5. Merge Method 3 (Isolation Forest)
        if os.path.exists(f3):
            df3 = pd.read_csv(f3)
            cols_m3 = [
                "time",
                "iso_forest_outlier",
                "iso_forest_score",
                "smooth_velocity",
                "psi",
            ]
            df1 = pd.merge(df1, df3[cols_m3], on="time", how="left")

        # 6. Merge Method 4 (DBSCAN)
        if os.path.exists(f4):
            df4 = pd.read_csv(f4)
            cols_m4 = ["time", "dbscan_cluster"]
            df1 = pd.merge(df1, df4[cols_m4], on="time", how="left")

        return df1

    except Exception as e:
        print(f"Warning: Failed to merge for {os.path.basename(f1)}: {e}")
        return None


def aggregate_all_results(base_dir):
    """
    Multi-threaded aggregation of all result files.
    """
    print("--- Starting Global Aggregation (Multi-Threaded) ---")
    path_m1 = os.path.join(base_dir, "regression")

    # Get all Method 1 files as the "base" list
    reg_files = glob.glob(os.path.join(path_m1, "*.csv"))

    if not reg_files:
        print(f"Error: No Method 1 results found in {path_m1}.")
        sys.exit()

    print(f"Found {len(reg_files)} stock files. Merging...")

    all_dfs = []

    # --- PARALLEL EXECUTION START ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map the worker function to the file list
        # list() forces the iterator to execute immediately
        results = list(executor.map(process_single_stock_merge, reg_files))
    # --- PARALLEL EXECUTION END ---

    # Filter out None results (failures)
    all_dfs = [df for df in results if df is not None]

    if not all_dfs:
        print("No data could be aggregated.")
        return pd.DataFrame()

    print("Merging into Master DataFrame...")
    # Create the Master DataFrame
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure time is datetime for plotting
    master_df["time"] = pd.to_datetime(master_df["time"])

    print(f"Aggregation Complete. Total rows: {len(master_df)}")
    return master_df


def generate_plots(df, output_dir):
    """
    Generates visualizations using the continuous 'bias_index'.
    """
    print("--- Generating Visualizations ---")

    # Plot 1: Distribution of Market Bias (Residuals)
    plt.figure(figsize=(10, 6))
    sns.histplot(df["bias_score_regression"], bins=50, kde=True, color="skyblue")
    plt.title("Distribution of Market-News Divergence (Bias Score)")
    plt.xlabel("Magnitude of Deviation (Residual)")
    plt.ylabel("Frequency (Days)")
    plt.savefig(os.path.join(output_dir, "plot_1_bias_distribution.png"))
    plt.close()

    # Plot 2: Bias by Sentiment Direction (Boxplot)
    active_days = df[df["bias_index"].abs() > 0.1].copy()
    active_days["Sentiment Direction"] = active_days["bias_index"].apply(
        lambda x: "Positive Sentiment" if x > 0 else "Negative Sentiment"
    )

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="Sentiment Direction",
        y="bias_score_regression",
        data=active_days,
        palette="vlag",
    )
    plt.title("Market Hypocrisy: Do we ignore Good or Bad news more?")
    plt.ylabel("Bias Score (Divergence Magnitude)")
    plt.savefig(os.path.join(output_dir, "plot_2_bias_by_sentiment.png"))
    plt.close()

    # Plot 3: Scatter of Continuous Sentiment vs Return
    plt.figure(figsize=(10, 6))
    plot_data = df[df["bias_index"].abs() > 0.05]

    sns.scatterplot(
        x="bias_index",
        y="log_returns",
        hue="iso_forest_outlier",
        data=plot_data,
        palette={1: "blue", -1: "red"},
        alpha=0.6,
    )
    plt.title("Continuous Sentiment vs. Price Return (Red = Anomalies)")
    plt.xlabel("Bias Index (-1 to +1)")
    plt.ylabel("Market Return (Log)")
    plt.axhline(0, color="grey", linestyle="--")
    plt.axvline(0, color="grey", linestyle="--")
    plt.legend(title="Is Anomaly?")
    plt.savefig(os.path.join(output_dir, "plot_3_sentiment_vs_return.png"))
    plt.close()

    # Plot 4: "Silent Shocks" Over Time (FIXED X-AXIS)
    # Aggregate by Month
    df["YearMonth"] = df["time"].dt.to_period("M")
    silent_counts = df.groupby("YearMonth")["silent_anomaly"].sum()

    # FIX: Convert index to clean strings for plotting
    silent_counts.index = silent_counts.index.astype(str)

    plt.figure(figsize=(12, 6))
    ax = silent_counts.plot(kind="bar", color="salmon", width=0.8)

    plt.title("Frequency of Silent Market Shocks (Neutral Sentiment) Over Time")
    plt.ylabel("Count of Anomalies")
    plt.xlabel("Month")

    # FIX: Calculate a step size to show max ~15 labels (prevent overcrowding)
    n = len(silent_counts)
    step = max(1, n // 15)

    # Manually set ticks and labels
    plt.xticks(
        ticks=range(0, n, step),
        labels=silent_counts.index[::step],
        rotation=45,
        ha="right",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_4_silent_shocks_time_series.png"))
    plt.close()

    print("Plots saved to output directory.")


def generate_text_report(df, output_dir):
    """
    Generates the text-based Media Bias Report using continuous variables.
    """
    print("--- Generating Text Report ---")

    report_file = os.path.join(output_dir, "media_bias_report.txt")

    # Calculate Metrics
    avg_bias = df["bias_score_regression"].mean()
    total_days = len(df)
    total_silent_shocks = df["silent_anomaly"].sum()
    silent_rate = (total_silent_shocks / total_days) * 100

    # Confirm Bias Events:
    bias_threshold = df["bias_score_regression"].quantile(0.90)

    confirmed_events = df[
        (df["bias_score_regression"] > bias_threshold)
        & (df["iso_forest_outlier"] == -1)
        & (df["bias_index"].abs() > 0.5)
    ]

    with open(report_file, "w") as f:
        f.write("=== MEDIA BIAS STUDY: FINAL ANALYSIS ===\n\n")
        f.write(f"Total Data Points: {total_days}\n")
        f.write(f"Stocks Analyzed: {df['stock_name'].nunique()}\n")
        f.write("--------------------------------------------------\n\n")

        f.write("1. MARKET EFFICIENCY SCORE (The 'Bias' Metric)\n")
        f.write(f"   Average Regression Residual: {avg_bias:.5f}\n")
        f.write(
            "   (Interpretation: The average % that price deviates from expected sentiment reaction.)\n\n"
        )

        f.write("2. COVERAGE GAPS (The 'Silent' Metric)\n")
        f.write(f"   Total Silent Shocks: {total_silent_shocks}\n")
        f.write(f"   Rate: {silent_rate:.2f}% of all trading days\n")
        f.write(
            "   (Interpretation: Major price/volume shocks occurring during Neutral Sentiment.)\n\n"
        )

        f.write("3. CONFIRMED BIAS INCIDENTS\n")
        f.write(f"   Total High-Confidence Anomalies: {len(confirmed_events)}\n")
        f.write(
            "   (Definition: Strong Sentiment (Index > 0.5) ignored by Market + Structural Anomaly.)\n\n"
        )

        f.write("4. TOP 5 MOST BIASED EVENTS (Case Studies)\n")
        top_5 = confirmed_events.sort_values(
            "bias_score_regression", ascending=False
        ).head(5)

        for i, row in top_5.iterrows():
            f.write(f"   Date: {row['time'].date()} | Stock: {row['stock_name']}\n")
            f.write(
                f"   Bias Index: {row['bias_index']:.4f} | Actual Return: {row['log_returns']:.4f}\n"
            )
            f.write(f"   Divergence Score: {row['bias_score_regression']:.4f}\n")
            f.write("   --------------------\n")

    print(f"Report saved to: {report_file}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    start_time = time.time()

    # Output folder
    OUTPUT_DIR = os.path.join(BASE_DIR, "final_study_output")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Aggregate (Now Parallelized)
    global_df = aggregate_all_results(BASE_DIR)

    if not global_df.empty:
        # 2. Visualize (Must be Sequential)
        generate_plots(global_df, OUTPUT_DIR)

        # 3. Report (Must be Sequential)
        generate_text_report(global_df, OUTPUT_DIR)

        # 4. Save Data
        global_df.to_csv(
            os.path.join(OUTPUT_DIR, "study_master_dataset.csv"), index=False
        )

    end_time = time.time()
    print(
        f"--- Full Study Inference Complete in {end_time - start_time:.2f} seconds ---"
    )
