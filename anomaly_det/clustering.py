import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import glob


def run_dbscan_clustering(input_file, output_file):
    # 1. Load Data
    df = pd.read_csv(input_file)

    # Check for columns
    required_cols = ["intc", "positive", "negative", "time", "stock_name"]
    if not all(col in df.columns for col in required_cols):
        print(f"Skipping {os.path.basename(input_file)}: Missing required columns.")
        return

    # 2. Prepare Features for Clustering
    # We want to cluster purely on the "News vs. Price" relationship

    # Log Returns
    df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

    # Net Sentiment
    df["net_sentiment"] = df["positive"] - df["negative"]

    # Drop NaNs
    cluster_data = df[["net_sentiment", "log_returns"]].dropna()

    if len(cluster_data) < 50:
        print(
            f"Skipping {os.path.basename(input_file)}: Not enough data for clustering."
        )
        return

    # 3. Scaling
    # Essential for DBSCAN because Sentiment (range -1 to 1) and Returns (range -0.05 to 0.05)
    # are on different scales.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data)

    # 4. Run DBSCAN
    # eps=0.5: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples=10: The number of samples in a neighborhood for a point to be considered as a core point.
    # You might need to tune 'eps' depending on your specific data density.
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
        "flag",
        "net_sentiment",
        "log_returns",
        "dbscan_cluster",  # The Output (-1 = Anomaly)
    ]

    # Sort: Put anomalies (-1) at the top
    final_df = df[output_cols].sort_values("dbscan_cluster")

    final_df.to_csv(output_file, index=False)

    # Optional: Debug Print
    # n_noise = (cluster_labels == -1).sum()
    # print(f"  > Noise points (Anomalies): {n_noise}")


if __name__ == "__main__":
    # CONFIGURATION
    INPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/"
    OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/clustering/"

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} not found.")
    else:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: {OUTPUT_DIR}")

        csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

        if not csv_files:
            print(f"No CSV files found in {INPUT_DIR}")
        else:
            print(f"Found {len(csv_files)} files to process.")

            for file_path in csv_files:
                try:
                    filename = os.path.basename(file_path)
                    output_name = f"{filename}"
                    output_path = os.path.join(OUTPUT_DIR, output_name)

                    print(f"Processing {filename}...", end=" ")
                    run_dbscan_clustering(file_path, output_path)
                    print("Done.")

                except Exception as e:
                    print(f"\nError processing {filename}: {e}")

            print("--- DBSCAN Clustering Complete ---")
