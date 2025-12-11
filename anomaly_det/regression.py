import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import glob


def run_regression_anomaly(input_file, output_file):
    # 1. Load Data
    df = pd.read_csv(input_file)

    # Check if necessary columns exist
    required_cols = ["intc", "positive", "negative", "time", "stock_name", "flag"]
    if not all(col in df.columns for col in required_cols):
        print(f"Skipping {os.path.basename(input_file)}: Missing required columns.")
        return

    # 2. Prepare Variables
    # Calculate Daily Log Returns: ln(Price_t / Price_t-1)
    df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

    # Calculate Net Sentiment (1 = Pos, -1 = Neg, 0 = Neutral)
    df["net_sentiment"] = df["positive"] - df["negative"]

    # Drop the first row (NaN return) to avoid errors
    data_for_model = df.dropna(subset=["log_returns", "net_sentiment"]).copy()

    # Check if enough data points exist for regression
    if len(data_for_model) < 10:
        print(f"Skipping {os.path.basename(input_file)}: Not enough data points.")
        return

    # 3. Train Regression Model
    # X = Sentiment (The Cause), y = Return (The Effect)
    X = data_for_model[["net_sentiment"]]
    y = data_for_model["log_returns"]

    model = LinearRegression()
    model.fit(X, y)

    # print(f"Model Coefficient (Impact of News): {model.coef_[0]:.5f}")

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
        "flag",
        "net_sentiment",
        "intc",
        "log_returns",
        "expected_return",
        "bias_score_regression",
    ]
    final_df = data_for_model[output_cols].sort_values(
        "bias_score_regression", ascending=False
    )

    final_df.to_csv(output_file, index=False)
    # print(f"Saved: {output_file}")


if __name__ == "__main__":
    # CONFIGURATION
    # Input is the output from the previous step (fin_process)
    INPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/"
    OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/regression"

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

        if not csv_files:
            print(f"No CSV files found in {INPUT_DIR}")
        else:
            print(f"Found {len(csv_files)} files to process.")

            for file_path in csv_files:
                try:
                    filename = os.path.basename(file_path)
                    # Name the result file clearly
                    output_name = f"{filename}"
                    output_path = os.path.join(OUTPUT_DIR, output_name)

                    print(f"Processing {filename}...", end=" ")
                    run_regression_anomaly(file_path, output_path)
                    print("Done.")

                except Exception as e:
                    print(f"\nError processing {filename}: {e}")

            print("--- Regression Analysis Complete ---")
