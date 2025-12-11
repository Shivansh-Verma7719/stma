import pandas as pd
import numpy as np
import os

# Import your custom indicator modules
# Ensure these .py files are in the same directory
import indicators.obv as obv
import indicators.impulsemacd as impulsemacd
import indicators.squeeze as squeeze
import indicators.velocity_indicator as velocity_indicator
import indicators.supertrend as supertrend


def load_and_prep_data(filepath):
    """
    Loads raw data and renames columns to match indicator requirements.
    Expected Raw Columns: stock_name, flag, positive, negative, open, high, low, close, volume
    """
    # Load data (assuming CSV, change read_csv to read_excel if needed)
    df = pd.read_csv(filepath)

    # Standardize column names to lowercase for safety
    df.columns = df.columns.str.lower().str.strip()

    # MAPPING: User Data -> Indicator Requirements
    # Indicators expect: 'into' (open), 'inth' (high), 'intl' (low), 'intc' (close), 'v' (volume)
    # Adjust the 'dict' keys below if your raw CSV headers are slightly different
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

    # Ensure 'time' column exists for ImpulseMACD (using index if necessary)
    if "time" not in df.columns:
        df["time"] = df.index

    return df


def add_indicators(df):
    """
    Runs the specific custom indicators selected for the Media Bias study.
    """
    print("--- Calculating Indicators ---")

    # 1. SuperTrend (Context Filter)
    # Uses period=10, multiplier=3 (Standard)
    print("Adding SuperTrend...")
    df = supertrend.SuperTrend(df, period=10, multiplier=3)
    # SuperTrend returns a cleaned DF, so we update 'df'

    # 2. Impulse MACD (Fast Reaction)
    # Uses lengthMA=34, lengthSignal=9 (Common for Impulse)
    print("Adding Impulse MACD...")
    # Note: impulsemacd returns a separate DF, we must merge it back
    macd_df = impulsemacd.macd(df, lengthMA=34, lengthSignal=9)
    # Merge on 'time' or index to ensure alignment
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
        # Fallback if time merging fails (direct assignment assuming index matches)
        df["ImpulseMACD"] = macd_df["ImpulseMACD"]
        df["ImpulseHisto"] = macd_df["ImpulseHisto"]
        df["ImpulseMACDCDSignal"] = macd_df["ImpulseMACDCDSignal"]

    # 3. Squeeze (Volatility/Indifference)
    # Uses conv=2.0 (Keltner factor), length=20
    print("Adding TTM Squeeze...")
    df = squeeze.squeeze_index2(df, conv=2.0, length=20, col="intc")
    # Adds column 'psi' to df

    # 4. Velocity (Shock Detector)
    # Lookback=20, EMA Length=10
    print("Adding Velocity...")
    df = velocity_indicator.calculate(df, lookback=20, ema_length=10)
    # Adds 'velocity' and 'smooth_velocity'

    # 5. OBV Custom (Smart Money)
    # Params: window_len=20, v_len=20, len10=10 (for EMA), slow_length=26
    print("Adding Custom OBV...")
    df = obv.calculate_custom_indicators(
        df, window_len=20, v_len=20, len10=10, slow_length=26
    )
    # Adds 'macd' (OBV-MACD), 'b5' (Slope/Breakout)

    return df


def clean_and_save(df, output_path):
    """
    Cleans up intermediate calculation columns and saves the final dataset.
    """
    # 1. Drop NaN values created by lookback periods (usually first ~30 rows)
    df.dropna(inplace=True)

    # 2. Select ONLY the columns relevant for Anomaly Detection
    # You can comment this out if you want to keep everything
    cols_to_keep = [
        # Original Data
        "time",
        "stock_name",
        "flag",
        "positive",
        "negative",
        "into",
        "inth",
        "intl",
        "intc",
        "v",
        # Generated Features
        "STX",  # SuperTrend Direction (Context)
        "ImpulseHisto",  # Impulse MACD Histogram (Momentum)
        "psi",  # Squeeze Metric (Volatility Compression)
        "smooth_velocity",  # Price Velocity (Impact Magnitude)
        "macd",  # Custom OBV MACD (Volume Divergence)
        "b5",  # OBV Slope Breakout (Volume Trend)
    ]

    # Filter columns if they exist
    final_cols = [c for c in cols_to_keep if c in df.columns]
    final_df = df[final_cols]

    # Save
    final_df.to_csv(output_path, index=False)
    print(f"--- Success! Processed data saved to {output_path} ---")
    print(f"Final Data Shape: {final_df.shape}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # CONFIGURATION
    INPUT_FILE = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_data/A.csv"  # <--- REPLACE THIS with your actual file name
    OUTPUT_FILE = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/processed_anomaly_data.csv"

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(
            f"Error: {INPUT_FILE} not found. Please ensure your data file is in the folder."
        )
    else:
        # RUN PIPELINE
        raw_df = load_and_prep_data(INPUT_FILE)
        enriched_df = add_indicators(raw_df)
        clean_and_save(enriched_df, OUTPUT_FILE)
