import pandas as pd
import numpy as np


def run_statistical_anomaly(input_file, output_file):
    print("--- Running Method 2: Statistical Anomaly (Z-Score & IQR) ---")

    # 1. Load Data
    df = pd.read_csv(input_file)

    # 2. Pre-calculation
    # Log Returns for Price
    df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

    # 3. Price Anomaly: Z-Score
    # "How many standard deviations is today's move?"
    mean_ret = df["log_returns"].mean()
    std_ret = df["log_returns"].std()

    df["z_score_price"] = (df["log_returns"] - mean_ret) / std_ret

    # 4. Volume Anomaly: IQR (Interquartile Range)
    # Volume is often non-normal (fat-tailed), so Z-score is bad. IQR is better.
    Q1 = df["v"].quantile(0.25)
    Q3 = df["v"].quantile(0.75)
    IQR = Q3 - Q1

    # Threshold: High Volume is anything above Q3 + 1.5*IQR
    vol_threshold = Q3 + (1.5 * IQR)
    df["is_volume_shock"] = df["v"] > vol_threshold

    # 5. The "Silent" Anomaly (The Leak Detector)
    # Logic: A huge price shock (> 3 sigma) OR Volume Shock...
    # ... BUT the News Flag is 0 (No news reported).

    df["silent_anomaly"] = False

    # Define "Huge Shock" as > 3 Std Devs or Volume Shock
    shock_condition = (df["z_score_price"].abs() > 3) | (df["is_volume_shock"])
    no_news_condition = df["flag"] == 0

    df.loc[shock_condition & no_news_condition, "silent_anomaly"] = True

    # 6. Save Results
    # We want to keep the Z-score to measure magnitude of the shock
    output_cols = [
        "time",
        "stock_name",
        "flag",
        "intc",
        "v",
        "z_score_price",
        "is_volume_shock",
        "silent_anomaly",
    ]

    final_df = df[output_cols].sort_values("z_score_price", key=abs, ascending=False)

    final_df.to_csv(output_file, index=False)

    print(
        f"Total 'Silent Anomalies' (Shocks with No News): {final_df['silent_anomaly'].sum()}"
    )
    print(f"Top 3 Price Shocks:\n{final_df[['time', 'z_score_price', 'flag']].head(3)}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    run_statistical_anomaly(
        "processed_anomaly_data.csv", "result_method2_statistical.csv"
    )
