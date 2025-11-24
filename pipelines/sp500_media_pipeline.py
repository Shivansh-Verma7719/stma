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
from helpers.db_helper import (
    insert_articles, 
    get_domain, 
    get_unprocessed_companies,
    update_company_state,
    mark_company_complete
)

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

# Global metrics tracking
import threading
global_metrics_lock = threading.Lock()
global_metrics = {
    "total_articles_pushed": 0,
    "unique_media_outlets": set(),
    "companies_processed": set(),
}


# Removed get_sp500_companies() - now using database-backed company list
# See scripts/seed_sp500.py to populate companies table


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
    search_api: mediacloud.api.SearchApi, query: str, worker_id: int, start_page: int = 1
):
    """Yields pages of stories, optionally resuming from a specific page."""
    pagination_token = None
    more_stories = True
    page_num = start_page

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
    start_page: int = 1,
):
    """Yields batches of processed articles with page tracking."""
    resume_msg = f" (resuming from page {start_page})" if start_page > 1 else ""
    update_worker_state(
        worker_id, 
        company=company_name, 
        symbol=stock_symbol, 
        status=f"Starting query...{resume_msg}"
    )
    
    query = create_search_query(company_name, stock_symbol)
    
    # Iterate over pages yielded by query_media_cloud
    for page_num, articles_page in enumerate(query_media_cloud(search_api, query, worker_id, start_page), start=start_page):
        # Update database with current page before processing
        update_company_state(stock_symbol, current_page=page_num)
        
        yield page_num, [
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


def worker_run(worker_id: int, api_key: str, companies_list: List[Dict]):
    """Process a list of companies with resumable state tracking."""
    update_worker_state(worker_id, status="Initializing...")
    search_api = mediacloud.api.SearchApi(api_key)

    total = len(companies_list)
    for local_idx, company_record in enumerate(companies_list, start=1):
        company = company_record["name"]
        symbol = company_record["symbol"]
        start_page = company_record.get("current_page", 0) + 1  # Resume from next page

        # Update progress
        update_worker_state(worker_id, status=f"Processing {local_idx}/{total}")
        
        try:
            # Iterate over batches yielded by process_company
            found_any = False
            for page_num, articles_batch in process_company(search_api, company, symbol, worker_id, start_page):
                if articles_batch:
                    found_any = True
                    update_worker_state(worker_id, status=f"Pushing {len(articles_batch)} to DB...")
                    insert_articles(articles_batch)
                    
                    # Update global metrics (articles and media outlets only - companies counted when fully done)
                    with global_metrics_lock:
                        global_metrics["total_articles_pushed"] += len(articles_batch)
                        for article in articles_batch:
                            domain = get_domain(article.get('url'))
                            if domain:
                                global_metrics["unique_media_outlets"].add(domain)
            
            # Mark company as fully processed after all pages are done
            mark_company_complete(symbol)
            with global_metrics_lock:
                global_metrics["companies_processed"].add(symbol)
        
        except Exception as e:
            # Persist error to database for debugging
            error_msg = str(e)[:500]  # Truncate long errors
            update_company_state(symbol, last_error=error_msg)
            update_worker_state(
                worker_id, 
                status=f"Error on {symbol}",
                errors=worker_states[worker_id].get("errors", 0) + 1
            )
            logging.error(f"Worker {worker_id} failed processing {symbol}: {e}", exc_info=True)
            recent_errors.append(f"[Worker {worker_id}] {symbol}: {error_msg[:100]}")

        # Enforce per-key rate limit between company queries too
        countdown_sleep(worker_id, 30)

    update_worker_state(worker_id, status="Done", company="-", symbol="-")


def generate_layout() -> Group:
    # Global metrics panel
    with global_metrics_lock:
        metrics_text = f"""[bold cyan]Total Articles Pushed:[/bold cyan] {global_metrics['total_articles_pushed']:,}
[bold magenta]Unique Media Outlets:[/bold magenta] {len(global_metrics['unique_media_outlets']):,}
[bold green]Companies Processed:[/bold green] {len(global_metrics['companies_processed']):,}"""
    
    metrics_panel = Panel(
        metrics_text,
        title="Global Metrics",
        style="bold white",
        box=box.DOUBLE,
        expand=False
    )
    
    # Worker status table
    table = Table(title="Worker Status", box=box.ROUNDED)
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
        "\n".join(recent_errors) if recent_errors else "No errors",
        title="Recent Errors",
        style="red",
        box=box.ROUNDED,
        height=7
    )
    
    return Group(metrics_panel, table, error_panel)


def main():
    console = Console()
    console.print("[bold green]Starting S&P 500 Media Cloud Article Pipeline (Stateful)[/bold green]")

    if not API_KEYS:
        console.print("[bold red]Error: No MEDIA_CLOUD_API_KEY_X found.[/bold red]")
        return

    # Fetch unprocessed companies from database
    console.print("Fetching unprocessed companies from database...")
    companies_list = get_unprocessed_companies()
    
    if not companies_list:
        console.print("[bold yellow]No unprocessed companies found.[/bold yellow]")
        console.print("Run 'python scripts/seed_sp500.py' to populate the database.")
        return

    total_companies = len(companies_list)
    console.print(f"Found {total_companies} unprocessed companies.")

    num_workers = min(3, len(API_KEYS), total_companies)
    console.print(f"Using {num_workers} parallel workers.")

    # Split companies list into chunks for workers
    chunk_size = (total_companies + num_workers - 1) // num_workers
    company_chunks = [
        companies_list[i:i + chunk_size] 
        for i in range(0, total_companies, chunk_size)
    ]

    # Initialize states
    for i in range(num_workers):
        update_worker_state(i + 1, status="Waiting to start...")

    with Live(generate_layout(), refresh_per_second=4) as live:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(len(company_chunks)):
                api_key = API_KEYS[worker_id]
                companies_chunk = company_chunks[worker_id]
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
