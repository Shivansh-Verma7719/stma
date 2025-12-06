import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_FILE = (
    "/Users/arnav/Desktop/workspaces/stma/stma/sp500_top250.csv"  # Your list of symbols
)
TARGET_DIRECTORY = (
    "/Users/arnav/Desktop/workspaces/stma/stma/fin_data"  # Folder to clean up
)
COLUMN_NAME = "Symbol"  # Column header in CSV


def check_files():
    # 1. Check if the directory exists first
    if not os.path.exists(TARGET_DIRECTORY):
        print(f"Error: The directory '{TARGET_DIRECTORY}' does not exist.")
        return

    # 2. Read the input file
    try:
        # If your file is Excel, change this to: pd.read_excel(INPUT_FILE)
        df = pd.read_csv(INPUT_FILE)

        # Check if column exists
        if COLUMN_NAME not in df.columns:
            print(f"Error: Column '{COLUMN_NAME}' not found in {INPUT_FILE}")
            print(f"Available columns are: {list(df.columns)}")
            return

        # Get the list of unique symbols and clean whitespace
        symbols = df[COLUMN_NAME].unique()

    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_FILE}' was not found.")
        return

    print(
        f"Checking {len(symbols)} symbols against directory '{TARGET_DIRECTORY}'...\n"
    )

    # 3. Iterate and Check
    found_count = 0
    missing_symbols = []

    for symbol in symbols:
        # distinct cleanup: ensure it's a string and remove spaces
        clean_symbol = str(symbol).strip()

        # Construct the expected filename (e.g., NVDA -> NVDA.csv)
        expected_filename = f"{clean_symbol}.csv"

        # Create the full path safely (works on Windows/Mac/Linux)
        full_path = os.path.join(TARGET_DIRECTORY, expected_filename)

        if os.path.exists(full_path):
            # File exists
            found_count += 1
            # Optional: Print found files
            # print(f"[OK] Found {expected_filename}")
        else:
            # File missing
            missing_symbols.append(clean_symbol)
            print(f"[MISSING] Could not find: {expected_filename}")

    # 4. Final Summary
    print("-" * 30)
    print(f"Summary:")
    print(f"Total Found:   {found_count}")
    print(f"Total Missing: {len(missing_symbols)}")

    if missing_symbols:
        print("\nList of missing symbols:")
        print(missing_symbols)


if __name__ == "__main__":
    check_files()
