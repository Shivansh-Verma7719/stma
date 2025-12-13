import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import glob
import concurrent.futures
import time

# --- CONFIGURATION ---
INPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/"
OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/clustering/"
MAX_WORKERS = 12  # Number of parallel threads


def run_dbscan_clustering(input_file, output_file):
    """
    Core logic: Clusters days based on 'Bias Index' vs 'Log Returns'.
    Returns a status string.
    """
    # 1. Load Data
    df = pd.read_csv(input_file)

    # Check for columns
    # UPDATED: Checking for 'bias_index' instead of positive/negative
    required_cols = ["intc", "bias_index", "time", "stock_name"]
    if not all(col in df.columns for col in required_cols):
        return f"Skipped: Missing required columns"

    # 2. Prepare Features for Clustering
    # We want to cluster purely on the "News vs. Price" relationship

    # Log Returns
    df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

    # UPDATED: Use bias_index directly as the sentiment feature
    # Drop NaNs
    cluster_data = df[["bias_index", "log_returns"]].dropna()

    if len(cluster_data) < 50:
        return "Skipped: Not enough data for clustering"

    # 3. Scaling
    # Essential for DBSCAN because Sentiment (-1 to 1) and Returns (~ -0.05 to 0.05)
    # are on different scales.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data)

    # 4. Run DBSCAN
    # eps=0.5: Maximum distance between two samples for one to be considered in the neighborhood.
    # min_samples=10: Number of samples in a neighborhood for a point to be a core point.
    dbscan = DBSCAN(eps=0.5, min_samples=10)

    # Fit and Predict
    # Result: -1 is Noise (Anomaly), 0, 1, 2... are Clusters
    cluster_labels = dbscan.fit_predict(X_scaled)

    # 5. Merge Results
    df["dbscan_cluster"] = np.nan
    df.loc[cluster_data.index, "dbscan_cluster"] = cluster_labels

    # 6. Save Results
    # We prioritize the Noise points (-1)
    output_cols = [
        "time",
        "stock_name",
        "bias_index",
        "log_returns",
        "dbscan_cluster",  # The Output (-1 = Anomaly)
    ]

    # Sort: Put anomalies (-1) at the top
    final_df = df[output_cols].sort_values("dbscan_cluster")

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
        status = run_dbscan_clustering(file_path, output_path)

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
                f"Found {total_files} files. Starting DBSCAN clustering with {MAX_WORKERS} threads..."
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
            print(f"--- DBSCAN Clustering Complete in {duration:.2f} seconds ---")
