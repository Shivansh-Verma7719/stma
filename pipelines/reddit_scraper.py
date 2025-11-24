import requests
import csv
import json
import time
import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os
import sys

# --- CONFIGURATION ---
COMPANY_CSV_PATH = 'data/sp500.csv'
OUTPUT_FILE = 'reddit_data.jsonl' # JSONL is the best format (one JSON object per line)
LOG_FILE = 'fetch_log.txt' # To log errors and progress

# List of subreddits to search within
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

# Timeframe
YEARS_TO_FETCH = 10
START_DATE = datetime.datetime.now() - relativedelta(years=YEARS_TO_FETCH)
END_DATE = datetime.datetime.now()

# Parallelism
# This is 'n', the number of parallel jobs to run.
# Start with 10-15. You can increase if the API doesn't rate-limit you.
MAX_WORKERS = 15

# API
BASE_URL = "https://api.pullpush.io/reddit/search/submission/"
REQUEST_TIMEOUT = 30  # seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5 # seconds

# --- END CONFIGURATION ---

# A thread-safe lock for writing to files
# This ensures two threads don't write to the same file at the exact same time
FILE_LOCK = Lock()

def log_message(message):
    """Logs a message to the console and a log file in a thread-safe way."""
    timestamp = datetime.datetime.now().isoformat()
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with FILE_LOCK:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

def load_companies_from_csv(csv_path):
    """
    Loads company and keyword data from the CSV file.
    Returns:
        list: A list of dictionaries, e.g.,
              [{'company': 'Apple', 'keywords': ['AAPL', 'iPhone', ...]}, ...]
    """
    if not os.path.exists(csv_path):
        log_message(f"CRITICAL ERROR: Company CSV not found at {csv_path}")
        sys.exit(1)

    companies = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            company_name = row.get('Security')
            symbol = row.get('Symbol')
            
            if company_name and symbol:
                # Use both Symbol and Company Name as keywords
                keywords = [symbol, company_name]
                companies.append({'company': company_name, 'keywords': keywords})
    log_message(f"Loaded {len(companies)} companies from CSV.")
    return companies

def generate_time_chunks(start_date, end_date):
    """
    Generates monthly (start_timestamp, end_timestamp) tuples
    from the start date to the end date.
    """
    chunks = []
    current_date = start_date
    while current_date < end_date:
        next_date = current_date + relativedelta(months=1)
        start_ts = int(current_date.timestamp())
        end_ts = int(next_date.timestamp()) - 1
        chunks.append((start_ts, end_ts))
        current_date = next_date
    log_message(f"Generated {len(chunks)} monthly time chunks from {start_date} to {end_date}.")
    return chunks

def create_job_queue(companies, subreddits, time_chunks):
    """
    Creates a master list of all jobs to be executed.
    Each job is a dictionary.
    """
    job_queue = []
    for company in companies:
        for keyword in company['keywords']:
            for subreddit in subreddits:
                for start_ts, end_ts in time_chunks:
                    job = {
                        'company': company['company'],
                        'keyword': keyword,
                        'subreddit': subreddit,
                        'start_ts': start_ts,
                        'end_ts': end_ts,
                        'filename': f"{company['company']}_{keyword}_{subreddit}_{start_ts}.json"
                    }
                    job_queue.append(job)
    log_message(f"Created a total job queue of {len(job_queue)} tasks.")
    return job_queue

def fetch_job(session, job):
    """
    The main function for each thread.
    Fetches data for a single job and returns the fetched posts.
    """
    params = {
        'q': job['keyword'],
        'subreddit': job['subreddit'],
        'after': job['start_ts'],
        'before': job['end_ts'],
        'fields': ['id', 'created_utc', 'title', 'selftext', 'subreddit', 'score', 'num_comments', 'permalink', 'url'],
        'size': 100, # Max allowed by API
        'sort': 'desc',
        'sort_type': 'created_utc'
    }

    all_posts_for_this_job = []
    current_timestamp = job['end_ts']

    # We loop *within* the job to handle pagination
    # The API only returns 100 posts at a time. We keep fetching
    # until no more posts are returned for this time window.
    while True:
        params['before'] = current_timestamp

        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)

                if response.status_code == 200:
                    data = response.json().get('data', [])
                    if not data:
                        # No more data for this job, break the pagination loop
                        return job, all_posts_for_this_job

                    all_posts_for_this_job.extend(data)

                    # Set the 'before' timestamp to the oldest post we just received
                    # to get the *next* page
                    current_timestamp = data[-1]['created_utc']
                    time.sleep(0.5) # Small delay to be nice to the API
                    break # Success, break the retry loop

                elif response.status_code == 429: # Too Many Requests
                    log_message(f"WARNING: Rate limited. Job: {job['keyword']}/{job['subreddit']}. Sleeping for 60s.")
                    time.sleep(60)
                else:
                    log_message(f"WARNING: API returned status {response.status_code} for job {job['keyword']}/{job['subreddit']}. Retrying...")
                    time.sleep(RETRY_DELAY * (attempt + 1)) # Exponential backoff

            except requests.exceptions.RequestException as e:
                log_message(f"ERROR: Request failed for job {job['keyword']}/{job['subreddit']}: {e}. Retrying...")
                time.sleep(RETRY_DELAY * (attempt + 1))
        else:
            # All retries failed
            log_message(f"CRITICAL: All retries failed for job: {job['keyword']}/{job['subreddit']}. Skipping this job.")
            return job, [] # Return empty list for this failed job

    return job, all_posts_for_this_job

def save_results(job, posts):
    """
    Saves the results of a completed job to the main JSONL file.
    This is called by the main thread, so it uses the file lock for safety.
    """
    if not posts:
        return # Don't save empty results

    with FILE_LOCK:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for post in posts:
                # Add our own metadata
                record = {
                    'company': job['company'],
                    'keyword_searched': job['keyword'],
                    'post_data': post
                }
                f.write(json.dumps(record) + '\n')

def main():
    """Main function to orchestrate the parallel fetching."""
    log_message("--- Starting Parallel Reddit Fetcher ---")

    # 1. Load and Generate Inputs
    companies = load_companies_from_csv(COMPANY_CSV_PATH)
    time_chunks = generate_time_chunks(START_DATE, END_DATE)
    job_queue = create_job_queue(companies, SUBREDDIT_LIST, time_chunks)

    if not job_queue:
        log_message("No jobs to run. Exiting.")
        return

    log_message(f"Starting {len(job_queue)} jobs with {MAX_WORKERS} parallel workers...")

    # We use one ThreadPoolExecutor and one Session
    # to share connections across all threads (much faster).
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, requests.Session() as session:

        # Submit all jobs to the executor
        # We create a dictionary mapping a 'future' (the running job)
        # to the job info itself, so we know which job finished.
        futures = {executor.submit(fetch_job, session, job): job for job in job_queue}

        completed_count = 0
        total_jobs = len(futures)

        # Use as_completed to process results as soon as they are done
        # This is much more memory-efficient than waiting for all jobs.
        for future in as_completed(futures):
            job = futures[future]

            try:
                # Get the result from the thread
                original_job, posts_data = future.result()

                if posts_data:
                    save_results(original_job, posts_data)
                    log_message(f"[{completed_count}/{total_jobs}] SUCCESS: Saved {len(posts_data)} posts for job: {original_job['company']}/{original_job['keyword']}/{original_job['subreddit']}/{original_job['start_ts']}")
                else:
                    log_message(f"[{completed_count}/{total_jobs}] INFO: No data found for job: {original_job['company']}/{original_job['keyword']}/{original_job['subreddit']}/{original_job['start_ts']}")

            except Exception as e:
                log_message(f"CRITICAL: Job {job['company']}/{job['keyword']} failed in main loop: {e}")

            completed_count += 1

    log_message("--- All jobs complete. ---")

if __name__ == "__main__":
    # Clear log file on new run
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    main()