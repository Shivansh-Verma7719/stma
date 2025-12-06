import pandas as pd
import yfinance as yf
import os

# --- Configuration ---
OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/fin_data"

# Fixed Date Range
START_DATE = "2015-11-30"
END_DATE = "2025-11-24"


def sanitize_ticker(ticker):
    """
    Converts tickers to Yahoo Finance format.
    Example: 'BRK.B' becomes 'BRK-B'
    """
    if isinstance(ticker, str):
        return ticker.replace(".", "-")
    return str(ticker)


def fetch_single_ticker(ticker):
    clean_ticker = sanitize_ticker(ticker)

    # Ensure directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    file_path = os.path.join(OUTPUT_DIR, f"{clean_ticker}.csv")

    print(f"Fetching data for: {clean_ticker}...")
    print(f"Range: {START_DATE} to {END_DATE}")

    try:
        # Download data
        stock_data = yf.download(
            clean_ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            progress=True,
            auto_adjust=True,
        )

        if stock_data.empty:
            print(f"❌ Error: No data found for {clean_ticker}. Check the spelling.")
            return

        # --- DATA CLEANING ---

        # 1. Handle MultiIndex Columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)

        # 2. Remove Duplicate Columns
        stock_data = stock_data.loc[:, ~stock_data.columns.duplicated()]

        # 3. Reset Index
        stock_data.reset_index(inplace=True)

        # 4. Format Date
        if "Date" in stock_data.columns:
            stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.date

        # 5. STRICT COLUMN SELECTION
        desired_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        # Check for missing columns
        missing_cols = [c for c in desired_columns if c not in stock_data.columns]
        if missing_cols:
            print(f"❌ Error: Missing columns {missing_cols}")
            return

        # Select and Reorder
        stock_data = stock_data[desired_columns]

        # 6. Save to CSV
        stock_data.to_csv(file_path, index=False)

        print(f"✅ Success! Saved to: {file_path}")
        print(f"   Rows retrieved: {len(stock_data)}")

    except Exception as e:
        print(f"❌ Critical Error: {e}")


if __name__ == "__main__":
    # Input Loop
    while True:
        user_input = input("\nEnter symbol (or 'q' to quit): ").strip().upper()
        if user_input == "Q":
            break
        if user_input:
            fetch_single_ticker(user_input)
