import pandas as pd
import requests
import time
from datetime import datetime
import os
import math # Import math for ceil function

# --- Configuration ---

# PullPush API endpoint for submissions search
BASE_URL = "https://api.pullpush.io/reddit/search/submission/"

# List of subreddits to search
SUBREDDIT_LIST = [
    'wallstreetbets',
    'stocks',
    'investing',
    'StockMarket',
    'finance',
    'dividendinvesting',
    'pennystocks',
    'trading',
    'RobinHood',
    'wealthmanagement',
    'TechnicalAnalysis',
    'PersonalFinance',
    'ETFs',
    'MemeStocks',
    'ValueInvesting'
]

# Time parameters
TIME_FRAME_YEARS = 5
SECONDS_IN_YEAR = 365 * 24 * 60 * 60
FIVE_YEARS_AGO_EPOCH = int(time.time() - (TIME_FRAME_YEARS * SECONDS_IN_YEAR))

# API Request Limits
MAX_RETRIES = 5
SLEEP_TIME = 2  # Initial sleep time between retries/pages (seconds)
RESULTS_PER_PAGE = 100  # Max results to fetch per API call (safe limit)

# File Paths
INPUT_CSV = 'data/sp500.csv'
OUTPUT_CSV = 'data/reddit_submissions.csv'

# --- Utility Functions ---

def fetch_submissions(query_keyword, subreddit_name, start_time_epoch, symbol):
    """
    Fetches submissions from the PullPush API for a specific keyword and time range,
    for a SINGLE subreddit, handling pagination and retries.
    """
    all_data = []
    
    # The 'before' parameter is used for pagination, starting from 'now' and moving backwards.
    current_before_epoch = int(time.time()) 
    
    print(f"  > Searching for keyword: '{query_keyword}' in subreddit: r/{subreddit_name}...")

    # Pagination loop: runs as long as the 'before' timestamp is newer than the 5-year limit
    while current_before_epoch > start_time_epoch:
        
        # Construct the URL parameters
        params = {
            'q': query_keyword,
            'subreddit': subreddit_name, # Only a single subreddit name here
            'size': RESULTS_PER_PAGE,
            'after': start_time_epoch, 
            'before': current_before_epoch,
            'sort': 'desc', # sort by newest first (descending timestamp)
            'sort_type': 'created_utc'
        }

        retries = 0
        success = False
        data = None

        # Retry loop with exponential backoff
        while retries < MAX_RETRIES:
            try:
                # Use a small timeout to quickly retry on connection issues
                response = requests.get(BASE_URL, params=params, timeout=10)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json().get('data', [])
                success = True
                break
            except requests.exceptions.HTTPError as e:
                # Since we resolved the URL length issue, this should be mostly rate limit (429) or other API issues
                # Note: The 400 error will still occur if 'subreddit' is empty, but we ensure it's not here.
                print(f"    [Error] HTTP Error: {e}. Retrying in {SLEEP_TIME * (2 ** retries)}s...")
            except requests.exceptions.RequestException as e:
                print(f"    [Error] Connection Error: {e}. Retrying in {SLEEP_TIME * (2 ** retries)}s...")

            time.sleep(SLEEP_TIME * (2 ** retries))
            retries += 1
        
        if not success:
            print(f"    [FAILURE] Max retries reached for '{query_keyword}' in r/{subreddit_name}. Moving on.")
            break
        
        if not data:
            # No more data in this time slice, we are done
            break

        # Process the fetched data
        new_records = []
        for post in data:
            new_records.append({
                'Symbol': symbol,
                'Keyword': query_keyword,
                'Subreddit': post.get('subreddit'),
                'Title': post.get('title'),
                'URL': post.get('full_link'),
                'Score': post.get('score'),
                'Created_UTC': post.get('created_utc'),
                'Date': datetime.fromtimestamp(post.get('created_utc')).strftime('%Y-%m-%d %H:%M:%S')
            })

        all_data.extend(new_records)

        # Update the 'before' time for the next iteration (get the timestamp of the oldest post)
        current_before_epoch = data[-1]['created_utc'] - 1 # Subtract 1 second to avoid duplicates
        
        # Print progress and take a quick break to be polite to the API
        newest_post_date = datetime.fromtimestamp(data[0]['created_utc']).strftime('%Y-%m-%d')
        oldest_post_date = datetime.fromtimestamp(data[-1]['created_utc']).strftime('%Y-%m-%d')
        print(f"    - Fetched {len(data)} posts. Range: {oldest_post_date} to {newest_post_date}. Total: {len(all_data)}")
        
        # We increase the sleep time slightly here since we're making many small requests
        time.sleep(SLEEP_TIME)

    return all_data

# --- Main Script Execution ---

def main():
    """
    Main function to orchestrate the data fetching process.
    """
    print("--- Reddit Submissions Fetcher (PullPush API) ---")
    print(f"Searching {len(SUBREDDIT_LIST)} subreddits for posts in the last {TIME_FRAME_YEARS} years.")
    print(f"Start Time (Epoch): {FIVE_YEARS_AGO_EPOCH}")
    
    if not os.path.exists(INPUT_CSV):
        print(f"\n[ERROR] Input CSV file '{INPUT_CSV}' not found. Please create it first.")
        return

    try:
        # 1. Read company data
        company_data = pd.read_csv(INPUT_CSV)
        
        # List to hold all collected submissions
        all_submissions_df = pd.DataFrame()

        # 2. Iterate through each company
        for index, row in company_data.iterrows():
            symbol = str(row['Symbol']).strip()
            security = str(row['Security']).strip()
            
            print(f"\n[COMPANY {index+1}/{len(company_data)}] Processing {symbol} ({security})...")

            # 3. Define keywords to search: Symbol and Security name
            keywords = [symbol]
            if symbol.lower() != security.lower():
                 keywords.append(security)
            
            company_submissions = []

            # 4. Fetch data for each keyword, iterating over subreddits one by one
            for keyword in keywords:
                
                # We skip very short/generic keywords that are not the symbol
                if len(keyword) > 2:
                    
                    # Iterate through the full subreddit list, one subreddit at a time
                    for subreddit_name in SUBREDDIT_LIST:
                        # Call the fetching function with the single subreddit
                        results = fetch_submissions(keyword, subreddit_name, FIVE_YEARS_AGO_EPOCH, symbol)
                        company_submissions.extend(results)

                else:
                    print(f"  > Skipping short/generic keyword: '{keyword}'")


            # 5. Convert to DataFrame and append
            if company_submissions:
                df = pd.DataFrame(company_submissions)
                all_submissions_df = pd.concat([all_submissions_df, df], ignore_index=True)
            
            # Using drop_duplicates here to count unique posts collected across all keywords and subreddits for this company
            unique_posts_count = len(all_submissions_df.drop_duplicates(subset=['URL', 'Title', 'Created_UTC']))
            print(f"\n--- {symbol} DONE. Total unique posts collected so far: {unique_posts_count}")


        # 6. Final cleanup and saving
        if not all_submissions_df.empty:
            # Remove duplicate posts
            final_df = all_submissions_df.drop_duplicates(subset=['URL', 'Title', 'Created_UTC']).reset_index(drop=True)
            final_df = final_df.sort_values(by='Created_UTC', ascending=False)
            
            # Save the final DataFrame to a CSV file
            final_df.to_csv(OUTPUT_CSV, index=False)
            
            print(f"\n=======================================================")
            print(f"✅ Success! Total unique submissions collected: {len(final_df)}")
            print(f"Data saved to '{OUTPUT_CSV}'")
            print(f"=======================================================")
        else:
            print("\n[INFO] No submissions were collected.")

    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Initialize the app ID context (standard practice in this environment)
    appId = 'default-app-id' 
    if 'appId' in locals():
      appId = appId 
    
    main()