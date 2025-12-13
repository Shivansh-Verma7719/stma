import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import glob
import concurrent.futures
import time

# --- CONFIGURATION ---
INPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/"
OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/forest/"
MAX_WORKERS = 12  # Number of parallel threads


def run_isolation_forest(input_file, output_file):
    """
    Core logic: Trains Isolation Forest to find context-based anomalies.
    Returns a status string.
    """
    # 1. Load Data
    df = pd.read_csv(input_file)

    # Check for necessary columns
    # UPDATED: Checking for 'bias_index' instead of positive/negative
    required_cols = [
        "bias_index",
        "smooth_velocity",
        "psi",
        "macd",
        "ImpulseHisto",
        "STX",
    ]
    if not all(col in df.columns for col in required_cols):
        return f"Skipped: Missing required columns"

    # 2. Preprocessing for ML
    # We need purely numeric data.

    # Map 'STX' (SuperTrend) from strings ('up'/'down') to numbers (1/-1)
    df["trend_score"] = df["STX"].map({"up": 1, "down": -1})
    # Fill any NaNs in trend_score (e.g. from start of data) with 0
    df["trend_score"] = df["trend_score"].fillna(0)

    # Select Features for the Model
    # UPDATED: Using 'bias_index' (Continuous) instead of binary flags
    features = [
        "bias_index",  # The News (Continuous -1 to 1)
        "smooth_velocity",  # The Price Reaction
        "psi",  # The Volatility State
        "macd",  # The Volume Flow (OBV)
        "ImpulseHisto",  # The Momentum
        "trend_score",  # The Market Regime
    ]

    # Create a clean subset for training (drop rows with NaNs)
    # We keep the index so we can merge back later
    ml_data = df[features].dropna()

    if len(ml_data) < 50:
        return "Skipped: Not enough clean data for ML"

    # 3. Scaling
    # Crucial because Volume (millions) is much larger than Sentiment (-1 to 1)
    scaler = StandardScaler()
    ml_data_scaled = scaler.fit_transform(ml_data)

    # 4. Train Isolation Forest
    # contamination=0.05 means we expect ~5% of days to be anomalies
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

    # Predict: -1 = Anomaly, 1 = Normal
    ml_data["iso_forest_outlier"] = iso_forest.fit_predict(ml_data_scaled)

    # Decision Function: The lower the score, the more abnormal it is.
    ml_data["iso_forest_score"] = iso_forest.decision_function(ml_data_scaled)

    # 5. Merge Results back to Original DataFrame
    df["iso_forest_outlier"] = np.nan
    df["iso_forest_score"] = np.nan

    df.loc[ml_data.index, "iso_forest_outlier"] = ml_data["iso_forest_outlier"]
    df.loc[ml_data.index, "iso_forest_score"] = ml_data["iso_forest_score"]

    # 6. Save Results
    # UPDATED output columns
    output_cols = [
        "time",
        "stock_name",
        "bias_index",
        "intc",
        "v",
        "iso_forest_outlier",
        "iso_forest_score",  # The ML Output
        "smooth_velocity",
        "psi",  # Context for manual review
    ]

    # Sort so the "weirdest" days (lowest score) are at the top
    final_df = df[output_cols].sort_values("iso_forest_score", ascending=True)

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
        status = run_isolation_forest(file_path, output_path)

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

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} not found.")
    else:
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
                f"Found {total_files} files. Starting Isolation Forest with {MAX_WORKERS} threads..."
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
            print(
                f"--- Isolation Forest Analysis Complete in {duration:.2f} seconds ---"
            )
