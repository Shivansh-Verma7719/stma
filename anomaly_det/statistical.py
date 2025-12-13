import pandas as pd
import numpy as np
import os
import glob
import concurrent.futures
import time

# --- CONFIGURATION ---
INPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/"
OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/statistical"
MAX_WORKERS = 12  # Number of parallel threads


def run_statistical_anomaly(input_file, output_file):
    """
    Core logic: Detects 'Silent Shocks' and 'Volume Anomalies'.
    Returns a status string (e.g., "Success", "Skipped").
    """
    # 1. Load Data
    df = pd.read_csv(input_file)

    # Check if necessary columns exist
    # UPDATED: We now require 'rel_vol' (from feature engineering)
    required_cols = ["intc", "v", "rel_vol", "bias_index", "time", "stock_name"]
    if not all(col in df.columns for col in required_cols):
        return f"Skipped: Missing required columns (need 'rel_vol' & 'bias_index')"

    # Check for sufficient data
    if len(df) < 10:
        return "Skipped: Not enough data points"

    # 2. Pre-calculation
    # Log Returns for Price
    df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

    # 3. Price Anomaly: Z-Score
    # "How many standard deviations is today's move?"
    mean_ret = df["log_returns"].mean()
    std_ret = df["log_returns"].std()

    # Avoid division by zero if std is 0
    if std_ret == 0:
        df["z_score_price"] = 0
    else:
        df["z_score_price"] = (df["log_returns"] - mean_ret) / std_ret

    # 4. Volume Anomaly: Relative Volume (The 10-Year Fix)
    # OLD WAY: Global IQR (Bad for long timeframes)
    # NEW WAY: Is today's volume 3x higher than the recent 50-day average?

    # We define a shock as Relative Volume > 3.0 (300% of normal)
    df["is_volume_shock"] = df["rel_vol"] > 3.0

    # 5. The "Silent" Anomaly (The Leak Detector)
    # Logic: A huge price shock (> 3 sigma) OR Volume Shock (> 3x normal)...
    # ... BUT the Bias Index is Neutral (No strong sentiment driving it).

    df["silent_anomaly"] = False

    # Define "Huge Shock" as > 3 Std Devs or Volume Shock
    shock_condition = (df["z_score_price"].abs() > 3) | (df["is_volume_shock"])

    # Define "No News" as Bias Index being very close to 0 (e.g., within +/- 0.1)
    neutral_threshold = 0.1
    no_news_condition = df["bias_index"].abs() < neutral_threshold

    df.loc[shock_condition & no_news_condition, "silent_anomaly"] = True

    # 6. Save Results
    output_cols = [
        "time",
        "stock_name",
        "bias_index",
        "intc",
        "v",
        "rel_vol",  # Added so you can see the relative volume score
        "z_score_price",
        "is_volume_shock",
        "silent_anomaly",
    ]

    final_df = df[output_cols].sort_values("z_score_price", key=abs, ascending=False)

    final_df.to_csv(output_file, index=False)
    return "Success"


def process_single_file(file_path):
    """
    Worker function to process a single CSV file.
    """
    try:
        filename = os.path.basename(file_path)
        output_name = f"{filename}"  # Keeping original name structure
        output_path = os.path.join(OUTPUT_DIR, output_name)

        # Run the core logic
        status = run_statistical_anomaly(file_path, output_path)

        # Return formatted message for the progress tracker
        if status == "Success":
            return f"Success: {filename}"
        else:
            return f"{status} ({filename})"

    except Exception as e:
        return f"Error ({os.path.basename(file_path)}): {str(e)}"


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    start_time = time.time()

    # Check input directory
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} not found.")
    else:
        # Create output directory
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: {OUTPUT_DIR}")

        # Find all processed CSV files
        csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
        total_files = len(csv_files)

        if not csv_files:
            print(f"No CSV files found in {INPUT_DIR}")
        else:
            print(
                f"Found {total_files} files. Starting statistical analysis with {MAX_WORKERS} threads..."
            )

            # Using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:
                # Submit all files
                future_to_file = {
                    executor.submit(process_single_file, f): f for f in csv_files
                }

                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_file):
                    completed_count += 1
                    result = future.result()

                    # Print progress
                    print(f"[{completed_count}/{total_files}] {result}")

            end_time = time.time()
            duration = end_time - start_time
            print(f"--- Statistical Analysis Complete in {duration:.2f} seconds ---")
