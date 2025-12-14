import pandas as pd
import numpy as np
import os
import glob
import concurrent.futures
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set visual style
sns.set_theme(style="whitegrid")

# --- CONFIGURATION ---
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "fin_process")
OUTPUT_DIR = os.path.join(BASE_DIR, "narrative_test")
MAX_WORKERS = 12
LAG_DAYS = 14  # Skip the first 14 days to let the Moving Average wash out


def run_momentum_test_in_memory(df, ticker=None):
    """
    Tests if Price Returns today affect Media Bias (after a 14-day lag) - in-memory version.
    Returns dict with stock_name and behavior, or None if failed.
    """
    try:
        df = df.copy()

        # Handle missing stock_name automatically
        if "stock_name" not in df.columns:
            if ticker:
                df["stock_name"] = ticker
            else:
                df["stock_name"] = "UNKNOWN"

        # Ensure norm_bias_score is renamed to bias_index
        if "norm_bias_score" in df.columns:
            df["bias_index"] = df["norm_bias_score"]
        elif "bias_index" not in df.columns:
            return None

        # Check columns
        required_cols = ["bias_index", "intc", "time", "stock_name"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return None

        # Calculate Today's Price Return
        df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

        # Prepare Future Media Bias Windows (WITH LAG)
        lagged_bias = df["bias_index"].shift(-LAG_DAYS)

        # We only need the 28-day window for the general classification
        indexer_28 = pd.api.indexers.FixedForwardWindowIndexer(window_size=28)
        df["future_bias_28d"] = lagged_bias.rolling(window=indexer_28).mean()

        # Drop NaNs
        data = df.dropna(subset=["log_returns", "future_bias_28d"]).copy()

        if len(data) < 50:
            return None

        # Run Regression
        X = data["log_returns"]
        slope_28, _, _, _, _ = stats.linregress(X, data["future_bias_28d"])

        # Determine Behavior
        behavior = "Trend Follower" if slope_28 > 0 else "Contrarian"

        # Return Summary
        return {"stock_name": data["stock_name"].iloc[0], "behavior": behavior}

    except Exception as e:
        return None


def process_dataframes_in_memory(data_dict):
    """
    Processes multiple DataFrames in-memory.
    Takes a dict of DataFrames keyed by ticker.
    Returns: list of dicts with stock_name and behavior
    """
    results = []

    for ticker, df in data_dict.items():
        result = run_momentum_test_in_memory(df, ticker)
        if result is not None:
            results.append(result)

    return results


def run_momentum_test(input_file):
    """
    Tests if Price Returns today affect Media Bias (after a 14-day lag).
    Returns only what is needed for the classification pie chart.
    """
    try:
        df = pd.read_csv(input_file)

        # 1. FIX: Handle missing stock_name automatically
        if "stock_name" not in df.columns:
            filename = os.path.basename(input_file)
            ticker = os.path.splitext(filename)[0]
            df["stock_name"] = ticker

        # Ensure norm_bias_score is renamed to bias_index
        if "norm_bias_score" in df.columns:
            df["bias_index"] = df["norm_bias_score"]
        elif "bias_index" not in df.columns:
            return None

        # 2. Check columns
        required_cols = ["bias_index", "intc", "time", "stock_name"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return None

        # 3. Calculate Today's Price Return
        df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

        # 4. Prepare Future Media Bias Windows (WITH LAG)
        # We shift the bias column backwards by LAG_DAYS (14).
        # So at row 't', we are seeing the bias starting at 't+14'.
        lagged_bias = df["bias_index"].shift(-LAG_DAYS)

        # We only need the 28-day window for the general classification
        indexer_28 = pd.api.indexers.FixedForwardWindowIndexer(window_size=28)
        df["future_bias_28d"] = lagged_bias.rolling(window=indexer_28).mean()

        # Drop NaNs
        data = df.dropna(subset=["log_returns", "future_bias_28d"]).copy()

        if len(data) < 50:
            return None

        # 5. Run Regression (Effectively t+14 to t+42)
        X = data["log_returns"]
        slope_28, _, _, _, _ = stats.linregress(X, data["future_bias_28d"])

        # 6. Determine Behavior (NO FILTER - Raw Direction)
        behavior = "Trend Follower" if slope_28 > 0 else "Contrarian"

        # 7. Return Summary
        return {"stock_name": data["stock_name"].iloc[0], "behavior": behavior}

    except Exception as e:
        print(f"Error processing {os.path.basename(input_file)}: {str(e)}")
        return None


def process_single_file(file_path):
    return run_momentum_test(file_path)


def generate_pie_chart(master_df, output_dir):
    print("Generating Classification Pie Chart...")

    if master_df.empty:
        return

    try:
        # --- PLOT: The Follower Ratio (ALL STOCKS) ---
        counts = master_df["behavior"].value_counts()

        plt.figure(figsize=(8, 8))

        # Determine colors based on labels to ensure consistency
        colors = []
        for label in counts.index:
            if label == "Trend Follower":
                colors.append("#99ff99")  # Green
            else:
                colors.append("#ffcc99")  # Orange

        plt.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
        )

        plt.title("Is the Media a 'Trend Follower'? (All Stocks)", fontsize=14)

        output_path = os.path.join(output_dir, "plot_2_media_behavior_all.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Pie chart saved to: {output_path}")

    except Exception as e:
        print(f"Error generating plot: {e}")


if __name__ == "__main__":
    start_time = time.time()

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} not found.")
    else:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
        total_files = len(csv_files)

        print(f"Testing Narrative Momentum on {total_files} stocks...")

        if total_files > 0:
            all_results = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:
                futures = {
                    executor.submit(process_single_file, f): f for f in csv_files
                }
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    result = future.result()
                    if result is not None:
                        all_results.append(result)

                    # Simple progress update
                    if (i + 1) % max(1, total_files // 10) == 0:
                        print(f"Progress: {i + 1}/{total_files}...")

            if all_results:
                master_df = pd.DataFrame(all_results)

                # Save data for reference
                master_df.to_csv(
                    os.path.join(OUTPUT_DIR, "_MASTER_MOMENTUM_REPORT.csv"), index=False
                )

                generate_pie_chart(master_df, OUTPUT_DIR)

                # Console Summary
                follower_pct = (master_df["behavior"] == "Trend Follower").mean() * 100
                print("\n=== FINAL VERDICT ===")
                print(f"Stocks Analyzed: {len(master_df)}")
                print(f"Trend Followers: {follower_pct:.1f}%")

            else:
                print("No results.")
        else:
            print("No CSV files found.")

    print(f"Done in {time.time() - start_time:.2f}s")
