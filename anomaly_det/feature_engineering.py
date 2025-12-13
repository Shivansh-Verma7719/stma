import pandas as pd
import numpy as np
import os
import glob
import concurrent.futures
import time

# Import your custom indicator modules
import indicators.obv as obv
import indicators.impulsemacd as impulsemacd
import indicators.squeeze as squeeze
import indicators.velocity_indicator as velocity_indicator
import indicators.supertrend as supertrend

# --- CONFIGURATION ---
INPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_data/"
OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/"
MAX_WORKERS = 12  # Number of threads


def load_and_prep_data(filepath):
    """
    Loads raw data and renames columns to match indicator requirements.
    Expected Raw Columns: bias_index, open, high, low, close, volume, stock_name
    """
    df = pd.read_csv(filepath)

    # Standardize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # MAPPING: User Data -> Indicator Requirements
    rename_map = {
        "o": "into",
        "open": "into",
        "h": "inth",
        "high": "inth",
        "l": "intl",
        "low": "intl",
        "c": "intc",
        "close": "intc",
        "v": "v",
        "volume": "v",
        "vol": "v",
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure 'time' column exists
    if "time" not in df.columns:
        if "date" in df.columns:
            df.rename(columns={"date": "time"}, inplace=True)
        else:
            df["time"] = df.index

    return df


def add_indicators(df):
    """
    Runs the specific custom indicators selected for the Media Bias study.
    """
    # 1. SuperTrend (Context)
    df = supertrend.SuperTrend(df, period=10, multiplier=3)

    # 2. Impulse MACD (Momentum)
    macd_df = impulsemacd.macd(df, lengthMA=34, lengthSignal=9)
    # Merge carefully
    if "time" in df.columns and "open_time" in macd_df.columns:
        df = pd.merge(
            df,
            macd_df[
                ["open_time", "ImpulseMACD", "ImpulseHisto", "ImpulseMACDCDSignal"]
            ],
            left_on="time",
            right_on="open_time",
            how="left",
        )
        df.drop(columns=["open_time"], inplace=True)
    else:
        df["ImpulseMACD"] = macd_df["ImpulseMACD"]
        df["ImpulseHisto"] = macd_df["ImpulseHisto"]
        df["ImpulseMACDCDSignal"] = macd_df["ImpulseMACDCDSignal"]

    # 3. Squeeze (Volatility)
    df = squeeze.squeeze_index2(df, conv=2.0, length=20, col="intc")

    # 4. Velocity (Shock Magnitude)
    # Note: Uses the new Percentage-based logic from the updated velocity_indicator.py
    df = velocity_indicator.calculate(df, lookback=20, ema_length=10)

    # 5. OBV Custom (Smart Money)
    df = obv.calculate_custom_indicators(
        df, window_len=20, v_len=20, len10=10, slow_length=26
    )

    # 6. RELATIVE VOLUME (The 10-Year Fix)
    df["vol_ma_50"] = df["v"].rolling(window=50).mean()
    df["rel_vol"] = df["v"] / df["vol_ma_50"]

    return df


def clean_and_save(df, output_path):
    """
    Cleans up intermediate calculation columns and saves the final dataset.
    """
    # Drop NaN values created by lookbacks (e.g. the first 50 days for vol_ma)
    df.dropna(inplace=True)

    cols_to_keep = [
        # Original Data
        "time",
        "stock_name",
        "bias_index",
        "into",
        "inth",
        "intl",
        "intc",
        "v",
        # Generated Features
        "STX",
        "ImpulseHisto",
        "psi",
        "smooth_velocity",
        "macd",
        "b5",
        "rel_vol",
    ]

    # Filter columns if they exist
    final_cols = [c for c in cols_to_keep if c in df.columns]
    final_df = df[final_cols]

    # Save
    final_df.to_csv(output_path, index=False)


def process_single_file(file_path):
    """
    Worker function to process a single CSV file.
    Returns the filename if successful, or None if failed.
    """
    try:
        filename = os.path.basename(file_path)
        output_name = f"{filename}"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        # RUN PIPELINE
        raw_df = load_and_prep_data(file_path)
        enriched_df = add_indicators(raw_df)
        clean_and_save(enriched_df, output_path)

        return f"Success: {filename}"

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

        # Find all CSV files in the input directory
        csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
        total_files = len(csv_files)

        if not csv_files:
            print(f"No CSV files found in {INPUT_DIR}")
        else:
            print(
                f"Found {total_files} files. Starting multi-threaded processing with {MAX_WORKERS} workers..."
            )

            # Using ThreadPoolExecutor to handle the threading logic
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:
                # Submit all files to the pool
                future_to_file = {
                    executor.submit(process_single_file, f): f for f in csv_files
                }

                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_file):
                    completed_count += 1
                    result = future.result()

                    # Print progress bar style output
                    print(f"[{completed_count}/{total_files}] {result}")

            end_time = time.time()
            duration = end_time - start_time
            print(f"--- Batch Processing Complete in {duration:.2f} seconds ---")
