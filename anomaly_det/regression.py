import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import glob
import concurrent.futures
import time

# --- CONFIGURATION ---
INPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/"
OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/regression"
MAX_WORKERS = 12  # Number of parallel threads


def run_regression_anomaly(input_file, output_file):
    """
    Core logic: Calculates the linear regression residuals between Bias Index and Returns.
    Returns a status string (e.g., "Success", "Skipped").
    """
    # 1. Load Data
    df = pd.read_csv(input_file)

    # Check if necessary columns exist
    required_cols = ["intc", "bias_index", "time", "stock_name"]
    if not all(col in df.columns for col in required_cols):
        return f"Skipped: Missing required columns (need 'bias_index')"

    # 2. Prepare Variables
    # Calculate Daily Log Returns: ln(Price_t / Price_t-1)
    df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

    # Drop the first row (NaN return) and any rows where bias_index is missing
    data_for_model = df.dropna(subset=["log_returns", "bias_index"]).copy()

    # Check if enough data points exist for regression
    if len(data_for_model) < 10:
        return "Skipped: Not enough data points"

    # 3. Train Regression Model
    # X = bias_index (The Cause: -1 to 1), y = Return (The Effect)
    X = data_for_model[["bias_index"]]
    y = data_for_model["log_returns"]

    model = LinearRegression()
    model.fit(X, y)

    # 4. Predict and Calculate Residuals
    data_for_model["expected_return"] = model.predict(X)

    # Residual = Actual Move - Expected Move
    data_for_model["reg_residual"] = (
        data_for_model["log_returns"] - data_for_model["expected_return"]
    )

    # bias_score_regression is the magnitude of the error
    data_for_model["bias_score_regression"] = data_for_model["reg_residual"].abs()

    # 5. Save Results
    output_cols = [
        "time",
        "stock_name",
        "bias_index",
        "intc",
        "log_returns",
        "expected_return",
        "bias_score_regression",
    ]
    final_df = data_for_model[output_cols].sort_values(
        "bias_score_regression", ascending=False
    )

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
        status = run_regression_anomaly(file_path, output_path)

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
                f"Found {total_files} files. Starting regression with {MAX_WORKERS} threads..."
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
            print(f"--- Regression Analysis Complete in {duration:.2f} seconds ---")
