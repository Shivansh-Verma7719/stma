import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import glob


def run_isolation_forest(input_file, output_file):
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
        print(f"Skipping {os.path.basename(input_file)}: Missing required columns.")
        return

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
    ml_data = df[features].dropna()

    if len(ml_data) < 50:
        print(f"Skipping {os.path.basename(input_file)}: Not enough clean data for ML.")
        return

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


if __name__ == "__main__":
    # CONFIGURATION
    INPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/"
    OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/forest/"

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
                    # Keeping original name
                    output_name = f"{filename}"
                    output_path = os.path.join(OUTPUT_DIR, output_name)

                    print(f"Processing {filename}...", end=" ")
                    run_isolation_forest(file_path, output_path)
                    print("Done.")

                except Exception as e:
                    print(f"\nError processing {filename}: {e}")

            print("--- Isolation Forest Analysis Complete ---")
