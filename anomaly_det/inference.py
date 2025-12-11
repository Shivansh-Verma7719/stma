import pandas as pd
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for plots
sns.set_theme(style="whitegrid")


def aggregate_all_results(base_dir):
    """
    Merges the separated result files from Method 1, 2, 3, and 4 into a single Global DataFrame.
    """
    print("--- Starting Global Aggregation ---")

    # Define paths based on your directory structure
    path_m1 = os.path.join(base_dir, "regression")  # Regression
    path_m2 = os.path.join(base_dir, "statistical")  # Statistical
    path_m3 = os.path.join(base_dir, "forest")  # Isolation Forest
    path_m4 = os.path.join(base_dir, "clustering")  # DBSCAN

    all_dfs = []

    # Get all Method 1 files as the "base" list
    # Pattern: STOCKNAME.csv (e.g., A.csv, AAPL.csv)
    reg_files = glob.glob(os.path.join(path_m1, "*.csv"))

    if not reg_files:
        print(f"Error: No Method 1 results found in {path_m1}.")
        sys.exit()

    print(f"Found {len(reg_files)} stock files. Merging...")

    for f1 in reg_files:
        try:
            # 1. Identify the core filename (e.g., "A.csv")
            basename = os.path.basename(f1)
            # Extract stock name from filename (e.g., "A.csv" -> "A")
            stock_name = os.path.splitext(basename)[0]

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
                # Merge on time
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

            all_dfs.append(df1)

        except Exception as e:
            print(f"Warning: Failed to merge for {basename}: {e}")

    if not all_dfs:
        print("No data could be aggregated.")
        return pd.DataFrame()

    # Create the Master DataFrame
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure time is datetime for plotting
    master_df["time"] = pd.to_datetime(master_df["time"])

    print(f"Aggregation Complete. Total rows: {len(master_df)}")
    return master_df


def generate_plots(df, output_dir):
    """
    Generates visualizations for the Media Bias study.
    """
    print("--- Generating Visualizations ---")

    # Plot 1: Distribution of Market Bias (Residuals)
    # This shows if the market generally follows news (peak at 0) or deviates (fat tails)
    plt.figure(figsize=(10, 6))
    sns.histplot(df["bias_score_regression"], bins=50, kde=True, color="skyblue")
    plt.title("Distribution of Market-News Divergence (Bias Score)")
    plt.xlabel("Magnitude of Deviation (Residual)")
    plt.ylabel("Frequency (Days)")
    plt.savefig(os.path.join(output_dir, "plot_1_bias_distribution.png"))
    plt.close()

    # Plot 2: Bias by Sentiment (Pos vs Neg)
    # Question: Does the market ignore bad news more than good news?
    # Filter for news days only
    news_only = df[df["net_sentiment"].abs() > 0].copy()
    news_only["Sentiment Type"] = news_only["net_sentiment"].apply(
        lambda x: "Positive" if x > 0 else "Negative"
    )

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="Sentiment Type", y="bias_score_regression", data=news_only, palette="vlag"
    )
    plt.title("Market Hypocrisy: Do we ignore Good or Bad news more?")
    plt.ylabel("Bias Score (Divergence)")
    plt.savefig(os.path.join(output_dir, "plot_2_bias_by_sentiment.png"))
    plt.close()

    # Plot 3: Scatter of News vs Return (The "Cloud")
    # This visualizes Method 4 (DBSCAN) logic
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="net_sentiment",
        y="log_returns",
        hue="iso_forest_outlier",
        data=df[df["flag"] == 1],
        palette={1: "blue", -1: "red"},
        alpha=0.6,
    )
    plt.title("News Sentiment vs. Price Return (Red = Anomalies)")
    plt.xlabel("News Sentiment (-1 to +1)")
    plt.ylabel("Market Return")
    plt.axhline(0, color="grey", linestyle="--")
    plt.axvline(0, color="grey", linestyle="--")
    plt.legend(title="Is Anomaly?")
    plt.savefig(os.path.join(output_dir, "plot_3_sentiment_vs_return.png"))
    plt.close()

    # Plot 4: "Silent Shocks" Over Time
    # Aggregating by Month to see if "Leaks" are increasing
    df["YearMonth"] = df["time"].dt.to_period("M")
    silent_counts = df.groupby("YearMonth")["silent_anomaly"].sum()

    plt.figure(figsize=(12, 6))
    silent_counts.plot(kind="bar", color="salmon")
    plt.title("Frequency of Silent Market Shocks (No News Coverage) Over Time")
    plt.ylabel("Count of Anomalies")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_4_silent_shocks_time_series.png"))
    plt.close()

    print("Plots saved to output directory.")


def generate_text_report(df, output_dir):
    """
    Generates the text-based Media Bias Report.
    """
    print("--- Generating Text Report ---")

    report_file = os.path.join(output_dir, "media_bias_report.txt")

    # Calculate Metrics
    avg_bias = df["bias_score_regression"].mean()
    total_days = len(df)
    total_silent_shocks = df["silent_anomaly"].sum()
    silent_rate = (total_silent_shocks / total_days) * 100

    # Confirm Bias Events: High Residual AND ML Anomaly AND News Day
    bias_threshold = df["bias_score_regression"].quantile(0.90)
    confirmed_events = df[
        (df["bias_score_regression"] > bias_threshold)
        & (df["iso_forest_outlier"] == -1)
        & (df["flag"] == 1)
    ]

    with open(report_file, "w") as f:
        f.write("=== MEDIA BIAS STUDY: FINAL ANALYSIS ===\n\n")
        f.write(f"Total Data Points: {total_days}\n")
        f.write(f"Stocks Analyzed: {df['stock_name'].nunique()}\n")
        f.write("--------------------------------------------------\n\n")

        f.write("1. MARKET EFFICIENCY SCORE (The 'Bias' Metric)\n")
        f.write(f"   Average Regression Residual: {avg_bias:.5f}\n")
        f.write(
            "   (Interpretation: The average % that price deviates from what news predicts.)\n\n"
        )

        f.write("2. COVERAGE GAPS (The 'Silent' Metric)\n")
        f.write(f"   Total Silent Shocks: {total_silent_shocks}\n")
        f.write(f"   Rate: {silent_rate:.2f}% of all trading days\n")
        f.write(
            "   (Interpretation: Days with major price action but ZERO news coverage.)\n\n"
        )

        f.write("3. CONFIRMED BIAS INCIDENTS\n")
        f.write(f"   Total High-Confidence Anomalies: {len(confirmed_events)}\n")
        f.write(
            "   (Definition: Market moved opposite to News AND Volume/Volatility was abnormal.)\n\n"
        )

        f.write("4. TOP 5 MOST BIASED EVENTS (Case Studies)\n")
        top_5 = confirmed_events.sort_values(
            "bias_score_regression", ascending=False
        ).head(5)
        for i, row in top_5.iterrows():
            f.write(f"   Date: {row['time'].date()} | Stock: {row['stock_name']}\n")
            f.write(
                f"   News Sentiment: {row['net_sentiment']} | Actual Return: {row['log_returns']:.4f}\n"
            )
            f.write(f"   Divergence Score: {row['bias_score_regression']:.4f}\n")
            f.write("   --------------------\n")

    print(f"Report saved to: {report_file}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # CONFIGURATION
    # Ensure this path points to where your results_method1, results_method2, etc. folders are
    BASE_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/"

    # Output folder
    OUTPUT_DIR = os.path.join(BASE_DIR, "final_study_output")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Aggregate
    global_df = aggregate_all_results(BASE_DIR)

    if not global_df.empty:
        # 2. Visualize
        generate_plots(global_df, OUTPUT_DIR)

        # 3. Report
        generate_text_report(global_df, OUTPUT_DIR)

        # 4. Save Data
        global_df.to_csv(
            os.path.join(OUTPUT_DIR, "study_master_dataset.csv"), index=False
        )
