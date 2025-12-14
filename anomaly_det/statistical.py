import pandas as pd
import numpy as np
import os
import glob
import concurrent.futures
import time

# --- CONFIGURATION ---
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "fin_process")
OUTPUT_DIR = os.path.join(BASE_DIR, "statistical")
MAX_WORKERS = 12  # Number of parallel threads


def run_statistical_anomaly_in_memory(df):
    """
    Core logic: Detects 'Silent Shocks' and 'Volume Anomalies' (in-memory version).
    Returns: pd.DataFrame or None if failed
    """
    try:
        df = df.copy()

        # Ensure norm_bias_score is renamed to bias_index
        if "norm_bias_score" in df.columns:
            df["bias_index"] = df["norm_bias_score"]
        elif "bias_index" not in df.columns:
            return None

        # Fix Missing 'stock_name' automatically
        if "stock_name" not in df.columns:
            df["stock_name"] = "UNKNOWN"

        # Check if necessary columns exist
        required_cols = ["intc", "v", "rel_vol", "bias_index", "time", "stock_name"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return None

        # Check for sufficient data
        if len(df) < 10:
            return None

        # Pre-calculation: Log Returns for Price
        df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

        # Price Anomaly: Z-Score
        mean_ret = df["log_returns"].mean()
        std_ret = df["log_returns"].std()

        # Avoid division by zero if std is 0
        if std_ret == 0:
            df["z_score_price"] = 0
        else:
            df["z_score_price"] = (df["log_returns"] - mean_ret) / std_ret

        # Volume Anomaly: Relative Volume
        df["is_volume_shock"] = df["rel_vol"] > 3.0

        # The "Silent" Anomaly
        df["silent_anomaly"] = False

        # Define "Huge Shock" as > 3 Std Devs or Volume Shock
        shock_condition = (df["z_score_price"].abs() > 3) | (df["is_volume_shock"])

        # Define "No News" Threshold - Using normalized scores (0-1)
        neutral_threshold = 0.1
        no_news_condition = df["bias_index"].abs() < neutral_threshold

        df.loc[shock_condition & no_news_condition, "silent_anomaly"] = True

        # Output columns
        output_cols = [
            "time",
            "stock_name",
            "bias_index",
            "intc",
            "v",
            "rel_vol",
            "z_score_price",
            "is_volume_shock",
            "silent_anomaly",
        ]

        # Sort by absolute z-score (highest magnitude moves first)
        final_df = df[output_cols].sort_values(
            "z_score_price", key=abs, ascending=False
        )

        return final_df

    except Exception as e:
        return None


def run_statistical_anomaly(input_file, output_file):
    """
    Core logic: Detects 'Silent Shocks' and 'Volume Anomalies'.
    Returns a status string (e.g., "Success", "Skipped").
    """
    try:
        # 1. Load Data
        df = pd.read_csv(input_file)

        # Ensure norm_bias_score is renamed to bias_index
        if "norm_bias_score" in df.columns:
            df["bias_index"] = df["norm_bias_score"]
        elif "bias_index" not in df.columns:
            return f"Skipped: Missing norm_bias_score or bias_index column"

        # 2. Fix Missing 'stock_name' automatically
        if "stock_name" not in df.columns:
            # Use filename without extension (e.g. "LMT.csv" -> "LMT")
            filename = os.path.basename(input_file)
            ticker = os.path.splitext(filename)[0]
            df["stock_name"] = ticker

        # 3. Check if necessary columns exist
        # We check for all required columns including the newly added 'stock_name'
        # Note: 'bias_index' should be present (either originally or created above)
        required_cols = ["intc", "v", "rel_vol", "bias_index", "time", "stock_name"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return f"Skipped: Missing columns {missing_cols}"

        # Check for sufficient data
        if len(df) < 10:
            return "Skipped: Not enough data points"

        # 4. Pre-calculation
        # Log Returns for Price
        df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

        # 5. Price Anomaly: Z-Score
        # "How many standard deviations is today's move?"
        mean_ret = df["log_returns"].mean()
        std_ret = df["log_returns"].std()

        # Avoid division by zero if std is 0
        if std_ret == 0:
            df["z_score_price"] = 0
        else:
            df["z_score_price"] = (df["log_returns"] - mean_ret) / std_ret

        # 6. Volume Anomaly: Relative Volume (The 10-Year Fix)
        # We define a shock as Relative Volume > 3.0 (300% of normal)
        df["is_volume_shock"] = df["rel_vol"] > 3.0

        # 7. The "Silent" Anomaly (The Leak Detector)
        # Logic: A huge price shock (> 3 sigma) OR Volume Shock (> 3x normal)...
        # ... BUT the Bias Index is Neutral (No strong sentiment driving it).

        df["silent_anomaly"] = False

        # Define "Huge Shock" as > 3 Std Devs or Volume Shock
        shock_condition = (df["z_score_price"].abs() > 3) | (df["is_volume_shock"])

        # Define "No News" Threshold
        # Using normalized scores (0-1), 'Neutral' is anything < 0.1.
        neutral_threshold = 0.1

        no_news_condition = df["bias_index"].abs() < neutral_threshold

        df.loc[shock_condition & no_news_condition, "silent_anomaly"] = True

        # 8. Save Results
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

        # Sort by absolute z-score (highest magnitude moves first)
        final_df = df[output_cols].sort_values(
            "z_score_price", key=abs, ascending=False
        )

        final_df.to_csv(output_file, index=False)
        return "Success"

    except Exception as e:
        return f"Error: {str(e)}"


def process_dataframes_in_memory(data_dict):
    """
    Processes multiple DataFrames in-memory.
    Takes a dict of DataFrames keyed by ticker.
    Returns: dict[str, pd.DataFrame] - Dictionary mapping ticker to processed DataFrame
    """
    result = {}

    for ticker, df in data_dict.items():
        processed = run_statistical_anomaly_in_memory(df)
        if processed is not None:
            result[ticker] = processed

    return result


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

        # Find all CSV files
        csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
        total_files = len(csv_files)

        if not csv_files:
            print(f"No CSV files found in {INPUT_DIR}")
        else:
            print(
                f"Found {total_files} files. Starting statistical analysis (Raw Score Mode) with {MAX_WORKERS} threads..."
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
