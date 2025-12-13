import pandas as pd
import numpy as np


def add_synthetic_bias_index(input_file, output_file):
    # 1. Load the raw data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # 2. Rename 'Date' to 'time'
    if "Date" in df.columns:
        df.rename(columns={"Date": "time"}, inplace=True)
    elif "date" in df.columns:
        df.rename(columns={"date": "time"}, inplace=True)

    # 3. Add Placeholder Stock Name
    # (Extracts from filename if possible, else defaults)
    stock_name = input_file.split("/")[-1].replace(".csv", "")
    df["stock_name"] = stock_name

    # 4. Generate Synthetic 'bias_index' (-1 to 1)
    np.random.seed(42)  # For reproducibility

    # Step A: Generate random noise (-1 to 1)
    # We use a normal distribution centered at 0 to mimic neutral bias being most common
    raw_noise = np.random.normal(loc=0, scale=0.5, size=len(df))

    # Step B: Create a "Trend" using a Moving Average
    # Media sentiment doesn't jump from -1 to +1 daily; it trends.
    # We smooth the noise over a 14-day window (2 weeks)
    smooth_bias = pd.Series(raw_noise).rolling(window=14, min_periods=1).mean()

    # Step C: Clip to ensure it stays strictly within -1 and 1
    df["bias_index"] = smooth_bias.clip(lower=-1.0, upper=1.0)

    # Optional: Introduce occasional "Viral Spikes" (Random days with extreme values)
    # This simulates a breaking news event that breaks the trend
    spike_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[spike_indices, "bias_index"] = np.random.uniform(
        -1, 1, size=len(spike_indices)
    )

    # 5. Reorder columns
    # Expected: time, stock_name, bias_index, open, high, low, close, volume
    cols_order = ["time", "stock_name", "bias_index"]
    remaining_cols = [c for c in df.columns if c not in cols_order]

    df = df[cols_order + remaining_cols]

    # 6. Save
    df.to_csv(output_file, index=False)
    print(f"Success! Saved prepared data with bias_index to: {output_file}")
    print(df[["time", "bias_index"]].head())


if __name__ == "__main__":
    # Settings
    # Update these paths to match your local setup
    RAW_FILE = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fetch/A.csv"
    READY_FILE = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_data/A.csv"

    add_synthetic_bias_index(RAW_FILE, READY_FILE)
