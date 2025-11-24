import pandas as pd
import requests
import time
import datetime as dt
import os
import sys
from io import StringIO
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import mediacloud.api
from dotenv import load_dotenv
from helpers.db_helper import insert_articles

import logging
from collections import deque
from rich.live import Live
from rich.table import Table
from rich.console import Console, Group
from rich.panel import Panel
from rich import box

load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/pipeline_errors_{dt.datetime.now().strftime('%Y%m%d_%H')}.log",
    level=logging.ERROR,
    format="%(asctime)s - Worker %(threadName)s - %(levelname)s - %(message)s",
)

# API keys – use multiple keys for parallel processing
API_KEYS = [
    os.getenv("MEDIA_CLOUD_API_KEY_1"),
    os.getenv("MEDIA_CLOUD_API_KEY_2"),
    os.getenv("MEDIA_CLOUD_API_KEY_3"),
]
API_KEYS = [k for k in API_KEYS if k]  # filter out missing/empty keys

START_DATE = dt.date(2015, 1, 1)
END_DATE = dt.date(2025, 11, 12)
US_NATIONAL_COLLECTION = 34412234  # Media Cloud US national collection ID

# Shared state for workers: worker_id -> dict
# dict keys: "company", "symbol", "status", "articles_found", "errors"
worker_states: Dict[int, Dict] = {}
recent_errors = deque(maxlen=5)


def get_sp500_companies() -> Optional[pd.DataFrame]:
    print("Fetching S&P 500 companies list...")
    url = "https://www.slickcharts.com/sp500"
    try:
        user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) "
            "Gecko/20100101 Firefox/111.0"
        )
        response = requests.get(url, headers={"User-Agent": user_agent})
        response.raise_for_status()
        sp500_df = pd.read_html(
            StringIO(response.text), match="Symbol", index_col="Symbol"
        )[0]
        # Ensure clean, 0-based integer index for easy chunking
        return sp500_df.reset_index()
    except Exception as e:
        msg = f"Error fetching S&P 500 list: {e}"
        print(msg)
        recent_errors.append(msg)
        return None


def create_search_query(company_name: str, stock_symbol: str) -> str:
    # Remove legal suffixes for cleaner name search
    clean_name = (
        company_name.replace(" Inc.", "")
        .replace(" Corp.", "")
        .replace(" Corporation", "")
        .replace(" Ltd.", "")
        .replace(" LLC", "")
        .replace(",", "")
    )
    # Use exact phrase match for company name to avoid loose matches
    # User reported issues with stock symbol matching subsets of words, so we prioritize company name.
    # Also filter for English language articles only.
    return f'"{clean_name}" AND language:en'


def update_worker_state(worker_id: int, **kwargs):
    """Helper to update the shared worker state safely."""
    if worker_id not in worker_states:
        worker_states[worker_id] = {
            "company": "-",
            "symbol": "-",
            "status": "Idle",
            "articles_found": 0,
            "errors": 0,
        }
    worker_states[worker_id].update(kwargs)


def countdown_sleep(worker_id: int, seconds: int):
    for remaining in range(seconds, 0, -1):
        update_worker_state(worker_id, status=f"Sleeping {remaining}s (Rate Limit)...")
        time.sleep(1)


def query_media_cloud(
    search_api: mediacloud.api.SearchApi, query: str, worker_id: int
):
    """Yields pages of stories."""
    pagination_token = None
    more_stories = True
    page_num = 1

    try:
        while more_stories:
            update_worker_state(worker_id, status=f"Fetching page {page_num}...")
            
            page, pagination_token = search_api.story_list(
                query,
                start_date=START_DATE,
                end_date=END_DATE,
                collection_ids=[US_NATIONAL_COLLECTION],
                pagination_token=pagination_token,
            )

            # Yield the current page of articles
            yield page
            
            update_worker_state(
                worker_id, 
                status=f"Page {page_num} done ({len(page)} articles)",
                articles_found=worker_states[worker_id]["articles_found"] + len(page)
            )
            
            more_stories = pagination_token is not None
            page_num += 1

            # Per-key rate limit ~2 req/min: 30s between calls
            if more_stories:
                countdown_sleep(worker_id, 30)

    except Exception as e:
        current_errors = worker_states[worker_id].get("errors", 0)
        update_worker_state(worker_id, status=f"Error: {str(e)[:50]}...", errors=current_errors + 1)
        logging.error(f"Worker {worker_id} failed to query '{query}': {e}", exc_info=True)
        recent_errors.append(f"[Worker {worker_id}] {str(e)[:100]}...")


def process_company(
    search_api: mediacloud.api.SearchApi,
    company_name: str,
    stock_symbol: str,
    worker_id: int,
):
    """Yields batches of processed articles."""
    update_worker_state(worker_id, company=company_name, symbol=stock_symbol, status="Starting query...")
    
    query = create_search_query(company_name, stock_symbol)
    
    # Iterate over pages yielded by query_media_cloud
    for articles_page in query_media_cloud(search_api, query, worker_id):
        yield [
            {
                "company_name": company_name,
                "stock_symbol": stock_symbol,
                "article_id": a.get("id", ""),
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "publish_date": a.get("publish_date", ""),
                "media_name": a.get("media_name", ""),
                "language": a.get("language", ""),
            }
            for a in articles_page
        ]


def chunk_dataframe(df: pd.DataFrame, n_chunks: int) -> List[pd.DataFrame]:
    if n_chunks <= 0:
        return [df]
    chunk_size = (len(df) + n_chunks - 1) // n_chunks
    return [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]


def worker_run(worker_id: int, api_key: str, companies_df: pd.DataFrame):
    update_worker_state(worker_id, status="Initializing...")
    search_api = mediacloud.api.SearchApi(api_key)

    total = len(companies_df)
    for local_idx, (_, row) in enumerate(companies_df.iterrows(), start=1):
        company = row["Company"]
        symbol = row["Symbol"]

        # Update progress
        update_worker_state(worker_id, status=f"Processing {local_idx}/{total}")
        
        # Iterate over batches yielded by process_company
        found_any = False
        for articles_batch in process_company(search_api, company, symbol, worker_id):
            if articles_batch:
                found_any = True
                update_worker_state(worker_id, status=f"Pushing {len(articles_batch)} to DB...")
                insert_articles(articles_batch)

        # Enforce per-key rate limit between company queries too
        countdown_sleep(worker_id, 30)

    update_worker_state(worker_id, status="Done", company="-", symbol="-")


def generate_layout() -> Group:
    table = Table(title="S&P 500 Media Cloud Pipeline", box=box.ROUNDED)
    table.add_column("Worker", justify="center", style="cyan", no_wrap=True)
    table.add_column("Company", style="magenta")
    table.add_column("Symbol", justify="center", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Articles", justify="right", style="blue")
    table.add_column("Errors", justify="right", style="red")

    for worker_id in sorted(worker_states.keys()):
        state = worker_states[worker_id]
        table.add_row(
            str(worker_id),
            state.get("company", "-"),
            state.get("symbol", "-"),
            state.get("status", "Idle"),
            str(state.get("articles_found", 0)),
            str(state.get("errors", 0)),
        )
    
    error_panel = Panel(
        "\n".join(recent_errors),
        title="Recent Errors",
        style="red",
        box=box.ROUNDED,
        height=7
    )
    
    return Group(table, error_panel)


def main():
    console = Console()
    console.print("[bold green]Starting S&P 500 Media Cloud Article Pipeline[/bold green]")

    if not API_KEYS:
        console.print("[bold red]Error: No MEDIA_CLOUD_API_KEY_X found.[/bold red]")
        return

    sp500_df = get_sp500_companies()
    if sp500_df is None:
        return

    total_companies = len(sp500_df)
    console.print(f"Found {total_companies} companies.")

    num_workers = min(3, len(API_KEYS), total_companies)
    console.print(f"Using {num_workers} parallel workers.")

    df_chunks = chunk_dataframe(sp500_df, num_workers)

    # Initialize states
    for i in range(num_workers):
        update_worker_state(i + 1, status="Waiting to start...")

    with Live(generate_layout(), refresh_per_second=4) as live:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                api_key = API_KEYS[worker_id]
                companies_chunk = df_chunks[worker_id]
                futures.append(
                    executor.submit(worker_run, worker_id + 1, api_key, companies_chunk)
                )

            # Monitor loop
            while True:
                live.update(generate_layout())
                
                # Check if all futures are done
                if all(f.done() for f in futures):
                    break
                
                time.sleep(0.25)
            
            # Check for exceptions
            for f in futures:
                try:
                    f.result()
                except Exception as e:
                    console.print(f"[bold red]Worker exception:[/bold red] {e}")

    console.print("[bold green]All workers finished.[/bold green]")


if __name__ == "__main__":
    main()
