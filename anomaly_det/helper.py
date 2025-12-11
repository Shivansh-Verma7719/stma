import pandas as pd
import numpy as np


def add_synthetic_flags(input_file, output_file):
    # 1. Load the raw data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # 2. Rename 'Date' to 'time'
    # (Crucial: The ImpulseMACD script specifically looks for a 'time' column)
    if "Date" in df.columns:
        df.rename(columns={"Date": "time"}, inplace=True)

    # 3. Add Placeholder Stock Name
    df["stock_name"] = "TEST_TICKER"

    # 4. Generate Synthetic News Flags (For Trial Run)
    np.random.seed(42)  # Ensures reproducible results

    # Create a 'flag' (News Published?) with 20% probability
    df["flag"] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

    # Initialize sentiment columns
    df["positive"] = 0
    df["negative"] = 0

    # Logic: If flag=1, randomly assign positive/negative sentiment
    # We use a mask to only update rows where news exists
    mask_news = df["flag"] == 1
    n_news = mask_news.sum()

    if n_news > 0:
        # Randomly assign 1 or 0 to positive/negative on news days
        # This allows for mixed sentiment (both 1) or neutral news (both 0)
        df.loc[mask_news, "positive"] = np.random.binomial(1, 0.5, size=n_news)
        df.loc[mask_news, "negative"] = np.random.binomial(1, 0.5, size=n_news)

    # 5. Reorder columns to match the 'feature_engineering.py' expectation
    # Expected: stock_name, flag, positive, negative, open, high, low, close, volume
    # We keep 'time' for the time-series index

    # flexible column selection to keep original OHLCV
    cols_order = ["time", "stock_name", "flag", "positive", "negative"]
    remaining_cols = [c for c in df.columns if c not in cols_order]

    df = df[cols_order + remaining_cols]

    # 6. Save
    df.to_csv(output_file, index=False)
    print(f"Success! Saved prepared data to: {output_file}")
    print(df.head())


if __name__ == "__main__":
    # Settings
    RAW_FILE = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_data/A.csv"  # Your input file
    READY_FILE = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/A_prepared.csv"  # The file to feed into feature_engineering.py

    add_synthetic_flags(RAW_FILE, READY_FILE)
