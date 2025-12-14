import pandas as pd
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
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS = 50  # Number of threads for file reading


def process_single_stock(file_path):
    """
    Worker function: Reads a single statistical result file.
    Returns: DataFrame if valid, None otherwise.
    """
    try:
        df = pd.read_csv(file_path)

        # Ensure required columns exist for Plot 4
        if "time" not in df.columns or "silent_anomaly" not in df.columns:
            return None

        # Add stock name for metadata
        if "stock_name" not in df.columns:
            basename = os.path.basename(file_path)
            df["stock_name"] = os.path.splitext(basename)[0]

        return df

    except Exception as e:
        print(f"Warning: Failed to read {os.path.basename(file_path)}: {e}")
        return None


def aggregate_statistical_results_in_memory(data_dict):
    """
    Aggregates statistical results from a dictionary of DataFrames (in-memory version).
    Takes a dict of DataFrames keyed by ticker.
    Returns: pd.DataFrame - Aggregated master DataFrame
    """
    print("--- Starting Aggregation of Statistical Data ---")

    all_dfs = []

    for ticker, df in data_dict.items():
        try:
            # Ensure required columns exist
            if "time" not in df.columns or "silent_anomaly" not in df.columns:
                continue

            # Add stock name for metadata
            if "stock_name" not in df.columns:
                df["stock_name"] = ticker

            all_dfs.append(df.copy())

        except Exception as e:
            print(f"Warning: Failed to process {ticker}: {e}")
            continue

    if not all_dfs:
        print("No valid data could be aggregated.")
        return pd.DataFrame()

    print("Merging into Master DataFrame...")
    try:
        master_df = pd.concat(all_dfs, ignore_index=True)
    except Exception as e:
        print(f"Error during concatenation: {e}")
        return pd.DataFrame()

    # Ensure time format is correct for plotting
    master_df["time"] = pd.to_datetime(master_df["time"], errors="coerce")

    print(f"Aggregation Complete. Total rows: {len(master_df)}")
    return master_df


def aggregate_statistical_results(base_dir):
    """
    Aggregates all files from the 'statistical' folder.
    """
    print("--- Starting Aggregation of Statistical Data ---")

    # We now look ONLY in the 'statistical' folder (Method 2 Output)
    path_stats = os.path.join(base_dir, "statistical")

    if not os.path.exists(path_stats):
        print(f"Error: Directory {path_stats} does not exist.")
        print("Please run 'method2_statistical.py' first.")
        sys.exit()

    stat_files = glob.glob(os.path.join(path_stats, "*.csv"))

    if not stat_files:
        print(f"Error: No result files found in {path_stats}.")
        sys.exit()

    print(f"Found {len(stat_files)} stock files. Merging...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_single_stock, stat_files))

    all_dfs = [df for df in results if df is not None]

    if not all_dfs:
        print("No valid data could be aggregated.")
        return pd.DataFrame()

    print("Merging into Master DataFrame...")
    try:
        master_df = pd.concat(all_dfs, ignore_index=True)
    except Exception as e:
        print(f"Error during concatenation: {e}")
        return pd.DataFrame()

    # Ensure time format is correct for plotting
    master_df["time"] = pd.to_datetime(master_df["time"], errors="coerce")

    print(f"Aggregation Complete. Total rows: {len(master_df)}")
    return master_df


def generate_silent_shock_plot(df, output_dir):
    """
    Generates ONLY Plot 4: Frequency of Silent Market Shocks over time.
    """
    print("--- Generating Silent Shock Visualization ---")

    try:
        if "silent_anomaly" in df.columns and "time" in df.columns:
            # Drop rows with invalid time
            df = df.dropna(subset=["time"])

            # Group by Month
            df["YearMonth"] = df["time"].dt.to_period("M")
            silent_counts = df.groupby("YearMonth")["silent_anomaly"].sum()

            if not silent_counts.empty:
                # Convert period index to string for plotting
                silent_counts.index = silent_counts.index.astype(str)

                plt.figure(figsize=(12, 6))
                silent_counts.plot(kind="bar", color="salmon", width=0.8)

                plt.title(
                    "Frequency of Silent Market Shocks (Neutral Sentiment) Over Time"
                )
                plt.ylabel("Count of Anomalies")
                plt.xlabel("Month")

                # Format X-axis labels to avoid clutter
                n = len(silent_counts)
                step = max(1, n // 15)
                plt.xticks(
                    ticks=range(0, n, step),
                    labels=silent_counts.index[::step],
                    rotation=45,
                    ha="right",
                )
                plt.tight_layout()

                output_path = os.path.join(
                    output_dir, "plot_4_silent_shocks_time_series.png"
                )
                plt.savefig(output_path)
                plt.close()
                print(f"Graph saved to: {output_path}")
            else:
                print("No silent anomalies found to plot.")
        else:
            print("Required columns (time, silent_anomaly) missing.")

    except Exception as e:
        print(f"Error generating Plot 4: {e}")


def generate_brief_report(df, output_dir):
    """
    Generates a minimal stats report focusing only on Silent Shocks.
    """
    if df.empty or "silent_anomaly" not in df.columns:
        return

    report_file = os.path.join(output_dir, "silent_shocks_report.txt")

    total_days = len(df)
    total_silent = df["silent_anomaly"].sum()
    silent_rate = (total_silent / total_days) * 100 if total_days > 0 else 0

    try:
        with open(report_file, "w") as f:
            f.write("=== SILENT SHOCK ANALYSIS ===\n")
            f.write(f"Total Data Points Analyzed: {total_days}\n")
            f.write(f"Total Silent Shocks Detected: {int(total_silent)}\n")
            f.write(f"Global Rate: {silent_rate:.2f}%\n")
            f.write("-----------------------------\n")
            f.write(
                "Definition: Days where Price moved > 2 Sigma but Media Sentiment remained Neutral (< 0.5 Sigma).\n"
            )

        print(f"Stats report saved to: {report_file}")
    except Exception as e:
        print(f"Error writing report: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    start_time = time.time()

    # Output goes to 'final_study_output'
    OUTPUT_DIR = os.path.join(BASE_DIR, "final_study_output")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Aggregate Data from Statistical Folder
    global_df = aggregate_statistical_results(BASE_DIR)

    # 2. Generate Plot and Report
    if not global_df.empty:
        generate_silent_shock_plot(global_df, OUTPUT_DIR)
        generate_brief_report(global_df, OUTPUT_DIR)

    end_time = time.time()
    print(f"--- Process Complete in {end_time - start_time:.2f} seconds ---")
