import pandas as pd
import yfinance as yf
import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
INPUT_FILE = "/Users/arnav/Desktop/workspaces/stma/stma/sp500_top250.csv"  # The file containing the list of tickers
OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/fin_data"  # Directory to save the CSV files
YEARS_BACK = 10  # How many years of history to fetch
MAX_WORKERS = 10  # Number of simultaneous downloads


def get_date_range():
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=YEARS_BACK * 365)
    return start_date, end_date


def sanitize_ticker(ticker):
    """
    Converts tickers to Yahoo Finance format.
    Example: 'BRK.B' becomes 'BRK-B'
    """
    if isinstance(ticker, str):
        return ticker.replace(".", "-")
    return str(ticker)


def process_ticker(ticker, start_date, end_date):
    """
    Downloads and formats data for a single ticker.
    Returns a status message string.
    """
    clean_ticker = sanitize_ticker(ticker)
    file_path = os.path.join(OUTPUT_DIR, f"{clean_ticker}.csv")

    # CHECK: Skip if file already exists
    if os.path.exists(file_path):
        return f"Skipped {clean_ticker}: Already downloaded."

    try:
        # Download daily data
        # auto_adjust=True: OHLC is adjusted for splits/dividends.
        stock_data = yf.download(
            clean_ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if stock_data.empty:
            return f"Warning: No data found for {clean_ticker}"

        # --- DATA CLEANING & FORMATTING ---

        # 1. Handle MultiIndex Columns (flatten them)
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Keep only the top level (Price) and drop the Ticker level
            stock_data.columns = stock_data.columns.get_level_values(0)

        # 2. Remove Duplicate Columns
        # This fixes issues where columns might appear twice (e.g. Open, Open)
        stock_data = stock_data.loc[:, ~stock_data.columns.duplicated()]

        # 3. Reset Index to make 'Date' a proper column
        stock_data.reset_index(inplace=True)

        # 4. Ensure 'Date' column is formatted as YYYY-MM-DD
        if "Date" in stock_data.columns:
            stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.date

        # 5. STRICT COLUMN SELECTION & ORDERING
        desired_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        # Check if we have all necessary columns
        missing_cols = [c for c in desired_columns if c not in stock_data.columns]
        if missing_cols:
            return f"Error {clean_ticker}: Missing columns {missing_cols}"

        # Select exactly these 6 columns in this order
        stock_data = stock_data[desired_columns]

        # 6. Save to CSV (index=False prevents the 0,1,2... row numbers)
        stock_data.to_csv(file_path, index=False)

        return f"Successfully downloaded {clean_ticker}"

    except Exception as e:
        return f"Error downloading {clean_ticker}: {e}"


def main():
    # 1. Create the output directory
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    # 2. Read the input CSV
    print(f"Reading tickers from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
        if "Symbol" not in df.columns:
            print("Error: Column 'Symbol' not found in CSV.")
            return
        tickers = df["Symbol"].tolist()
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    start_date, end_date = get_date_range()
    total_tickers = len(tickers)

    print(f"Found {total_tickers} tickers. Starting multithreaded download...")
    print(f"Timeframe: {start_date} to {end_date}")
    print(f"Max Workers: {MAX_WORKERS}")
    print("-" * 50)

    # 3. Multithreaded Execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(process_ticker, ticker, start_date, end_date): ticker
            for ticker in tickers
        }

        # Process results as they complete
        completed_count = 0
        for future in as_completed(future_to_ticker):
            completed_count += 1
            result = future.result()
            print(f"[{completed_count}/{total_tickers}] {result}")

    print("\nProcess complete.")


if __name__ == "__main__":
    main()
