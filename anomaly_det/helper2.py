import os
import pandas as pd
import glob

# ================= CONFIGURATION =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIRECTORY = os.path.join(BASE_DIR, "fetch")
OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "fin_data")
# =================================================


def process_market_data(data_dict):
    """
    Processes market data in-memory. Takes a dict of DataFrames keyed by ticker.
    Returns: dict[str, pd.DataFrame] - Dictionary mapping ticker to processed DataFrame
    """
    result = {}

    for ticker, df in data_dict.items():
        try:
            df = df.copy()

            # Ensure ticker column exists (use key if missing)
            if "ticker" not in df.columns:
                df["ticker"] = ticker

            # Check if required columns exist (basic validation)
            required_cols = [
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "norm_bias_score",
            ]
            if not all(col in df.columns for col in required_cols):
                print(f"[Skipped] Missing columns for {ticker}")
                continue

            # Rename columns
            rename_map = {
                "norm_bias_score": "bias_index",
            }
            df = df.rename(columns=rename_map)

            # Select and Reorder columns (keep all columns, just ensure bias_index exists)
            # We keep all columns in case there are other useful ones
            if "bias_index" not in df.columns:
                print(f"[Skipped] bias_index not created for {ticker}")
                continue

            result[ticker] = df

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return result


def process_market_csvs():
    # 1. Create Output Directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: {OUTPUT_DIRECTORY}")

    # 2. Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(INPUT_DIRECTORY, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {INPUT_DIRECTORY}")
        return

    print(f"Found {len(csv_files)} files to process.")

    # 3. Process each file
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path)
            print(f"Processing {filename}...", end=" ")

            # Read CSV
            df = pd.read_csv(file_path)

            # Check if required columns exist (basic validation)
            required_cols = [
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "norm_bias_score",
            ]
            if not all(col in df.columns for col in required_cols):
                print(f"[Skipped] Missing columns in {filename}")
                continue

            # Rename columns
            # Mapping: source_name -> target_name
            rename_map = {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "norm_bias_score": "bias_index",
            }
            df = df.rename(columns=rename_map)

            # Select and Reorder columns
            # Target format: date, ticker, o, h, l, c, volume, bias_index
            final_columns = [
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "bias_index",
            ]
            df_final = df[final_columns]

            # Save to output directory
            output_path = os.path.join(OUTPUT_DIRECTORY, filename)
            df_final.to_csv(output_path, index=False)

            print("Done.")

        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print("\nAll files processed.")


if __name__ == "__main__":
    process_market_csvs()
