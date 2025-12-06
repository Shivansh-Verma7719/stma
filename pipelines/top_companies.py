import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# --- Configuration ---
INPUT_FILE = "stma/data/sp500.csv"
OUTPUT_FILE = "sp500_top200.csv"  # Renamed to reflect the content
MAX_WORKERS = 20  # Fast parallel fetching
TOP_N = 250  # Number of stocks to keep


def sanitize_ticker(ticker):
    """
    Converts tickers to Yahoo Finance format.
    Example: 'BRK.B' becomes 'BRK-B'
    """
    if isinstance(ticker, str):
        return ticker.replace(".", "-")
    return str(ticker)


def fetch_market_cap(ticker):
    """
    Fetches the market cap for a single ticker using fast_info.
    Returns a tuple: (Original_Ticker, Market_Cap_Value)
    """
    clean_ticker = sanitize_ticker(ticker)

    try:
        # Ticker.fast_info is efficient for scalar data like market cap
        dat = yf.Ticker(clean_ticker).fast_info
        mcap = dat["market_cap"]
        return ticker, mcap
    except Exception as e:
        # Return None if data cannot be found
        return ticker, None


def main():
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

    print(f"Fetching Market Caps for {len(tickers)} tickers...")
    print(f"Max Workers: {MAX_WORKERS}")
    print("-" * 50)

    # Dictionary to store results: { 'Symbol': MarketCap }
    mcap_results = {}

    # Multithreaded Fetching
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(fetch_market_cap, ticker): ticker for ticker in tickers
        }

        completed_count = 0
        total_tickers = len(tickers)

        for future in as_completed(future_to_ticker):
            completed_count += 1
            ticker, mcap = future.result()

            # Print status occasionally to avoid clutter, or on errors
            if completed_count % 10 == 0 or mcap is None:
                status = "Success" if mcap else "No Data"
                print(f"[{completed_count}/{total_tickers}] {ticker}: {status}")

            mcap_results[ticker] = mcap

    print("-" * 50)
    print("Processing data...")

    # 1. Map market caps to dataframe
    df["Market Cap"] = df["Symbol"].map(mcap_results)

    # 2. Sort by Market Cap in Descending order
    # (NaN values will be placed at the end by default)
    df = df.sort_values(by="Market Cap", ascending=False)

    # 3. Slice to keep only the top N
    df_top = df.head(TOP_N)

    # 4. Save to new CSV
    df_top.to_csv(OUTPUT_FILE, index=False)

    print(f"Success! Filtered top {TOP_N} stocks by market cap.")
    print(f"Saved to: {OUTPUT_FILE}")

    # Preview
    print("\nTop 5 entries:")
    print(df_top[["Symbol", "Market Cap"]].head().to_string(index=False))

    print("\nBottom 5 entries (of the top 200):")
    print(df_top[["Symbol", "Market Cap"]].tail().to_string(index=False))


if __name__ == "__main__":
    main()
