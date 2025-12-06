import requests
import time
import os
import sys
import json
import re
import logging
import threading
import datetime as dt
from urllib.parse import urlparse
from collections import deque
from typing import List, Dict, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock

from newspaper import Article
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from helpers.db_helper import (
    get_articles_for_scraping,
    update_articles_batch,
    get_scraping_progress,
    get_domain,
    upsert_media_outlet
)

from rich.live import Live
from rich.table import Table
from rich.console import Console, Group
from rich.panel import Panel
from rich import box

load_dotenv()

# ============================================================================
# Configuration - HPC Optimized
# ============================================================================

# CONFIGURABLE: Adjust these based on your VM capacity
# For 50 threads + 10-12 cores + 3-4GB RAM, recommended:
# NUM_WORKERS = 32-40 (threads are lightweight, I/O bound)
# BATCH_SIZE = 30-40 (smaller batches to avoid memory bloat)
# PUSH_BATCH_SIZE = 80-100 (balance between DB writes and memory)
NUM_WORKERS = int(os.getenv('NUM_WORKERS', 10))  # Configurable via env var, default 16
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 30))  # Smaller batches for memory efficiency
PUSH_BATCH_SIZE = int(os.getenv('PUSH_BATCH_SIZE', 80))  # Async push threshold
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 20))  # Timeout per article
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))  # Max retry attempts
RETRY_BACKOFF = 2  # Exponential backoff multiplier
DELAY_BETWEEN_ARTICLES = 0.3  # Reduced delay with more threads
RATE_LIMIT_DELAY = 30  # Seconds to wait after 429 error before retry
RATE_LIMIT_BACKOFF_MULTIPLIER = 2  # How much to increase delay on repeated 429s

# User agents for rotating to avoid 403 blocks
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
]

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/content_scraper_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.ERROR,
    format="%(asctime)s - Worker %(threadName)s - %(levelname)s - %(message)s",
)

# ============================================================================
# Global State Management
# ============================================================================

# Worker status tracking: worker_id -> dict
worker_states: Dict[int, Dict] = {}
worker_states_lock = Lock()

# Metrics tracking
global_metrics_lock = threading.Lock()
global_metrics = {
    "total_articles_processed": 0,
    "successful_scrapes": 0,
    "failed_scrapes": 0,
    "rate_limited_articles": 0,
    "pending_push": [],  # Deferred updates to be pushed
    "rate_limited_queue": deque(),  # Articles that hit rate limits (429), retry later
}

# Recent errors queue
recent_errors = deque(maxlen=10)

# Semaphore for batch pushing (async coordination)
push_semaphore = Semaphore(1)

# Rate limit tracking per domain
rate_limit_state_lock = Lock()
rate_limit_state = {}  # domain -> {'blocked_until': timestamp, 'delay': seconds}


# ============================================================================
# Utility Functions
# ============================================================================

def extract_social_handles(html_content: str) -> Dict:
    """Extract social media handles from article HTML."""
    handles = {}
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        patterns = {
            'twitter': r'twitter\.com/([a-zA-Z0-9_]+)',
            'facebook': r'facebook\.com/([a-zA-Z0-9\.]+)',
            'linkedin': r'linkedin\.com/company/([a-zA-Z0-9-]+)',
            'instagram': r'instagram\.com/([a-zA-Z0-9_\.]+)',
            'youtube': r'youtube\.com/(?:user/|c/|channel/|@)([a-zA-Z0-9_-]+)'
        }
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            for platform, pattern in patterns.items():
                match = re.search(pattern, href, re.IGNORECASE)
                if match:
                    if 'share' not in href and 'intent' not in href:
                        if platform not in handles:
                            handles[platform] = []
                        handle = match.group(1)
                        if handle not in handles[platform]:
                            handles[platform].append(handle)
    except Exception as e:
        logging.error(f"Error extracting social handles: {e}")
    
    return handles


def get_random_user_agent() -> str:
    """Get a random user agent to avoid detection."""
    import random
    return random.choice(USER_AGENTS)


def scrape_article_content(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[Dict]:
    """
    Scrape article content from URL.
    
    Returns:
        Dict with keys: content, html, social_handles, error_code (if failed)
        Or None if scraping failed
    """
    try:
        # Rotate user agents to bypass 403 blocks
        config = {
            'browser_user_agent': get_random_user_agent(),
            'request_timeout': timeout,
            'number_threads': 1,
            'fetch_images': False,
            'memoize_articles': False,
        }
        
        article = Article(url, keep_article_html=True, **config)
        article.download()
        article.parse()
        
        content = article.text
        html = article.html
        social_handles = extract_social_handles(html)
        
        return {
            'content': content,
            'html': html,
            'social_handles': social_handles
        }
    except Exception as e:
        error_str = str(e)
        error_str_lower = error_str.lower()
        
        # Extract HTTP status code if present
        error_code = None
        status_match = re.search(r'\b(\d{3})\b', error_str)
        if status_match:
            error_code = status_match.group(1)
        
        # Check for specific error types
        if '429' in error_str or 'too many requests' in error_str_lower:
            return {'rate_limited': True, 'url': url, 'error': error_str, 'error_code': '429'}
        
        # Common HTTP errors
        error_type = 'Unknown Error'
        if '403' in error_str or 'forbidden' in error_str_lower:
            error_type = 'HTTP 403 Forbidden'
            error_code = '403'
        elif '404' in error_str or 'not found' in error_str_lower:
            error_type = 'HTTP 404 Not Found'
            error_code = '404'
        elif '500' in error_str or 'internal server error' in error_str_lower:
            error_type = 'HTTP 500 Internal Server Error'
            error_code = '500'
        elif '502' in error_str or 'bad gateway' in error_str_lower:
            error_type = 'HTTP 502 Bad Gateway'
            error_code = '502'
        elif '503' in error_str or 'service unavailable' in error_str_lower:
            error_type = 'HTTP 503 Service Unavailable'
            error_code = '503'
        elif 'timeout' in error_str_lower or 'timed out' in error_str_lower:
            error_type = 'Connection Timeout'
            error_code = 'TIMEOUT'
        elif 'connection' in error_str_lower:
            error_type = 'Connection Error'
            error_code = 'CONN_ERR'
        elif 'ssl' in error_str_lower or 'certificate' in error_str_lower:
            error_type = 'SSL/Certificate Error'
            error_code = 'SSL_ERR'
        
        logging.error(f"Error scraping {url}: [{error_code}] {error_str}")
        return {'error': error_str, 'error_code': error_code, 'error_type': error_type}


def update_worker_state(worker_id: int, **kwargs):
    """Thread-safe worker state update."""
    with worker_states_lock:
        if worker_id not in worker_states:
            worker_states[worker_id] = {
                "status": "Idle",
                "articles_processed": 0,
                "articles_pending_push": 0,
                "errors": 0,
            }
        worker_states[worker_id].update(kwargs)


def countdown_sleep(worker_id: int, seconds: int, reason: str = "Rate Limit"):
    """Sleep with countdown status updates."""
    # For HPC with many threads, don't block on sleep - just log and move on
    if seconds > 0:
        time.sleep(seconds)
    update_worker_state(worker_id, status=f"Ready after {reason}")


# ============================================================================
# Worker and Batch Processing
# ============================================================================

def process_article_batch(articles_batch: List[Dict], worker_id: int) -> List[Dict]:
    """
    Process a batch of articles and return updates for DB push.
    
    Returns:
        List of update dicts ready for update_articles_batch()
    """
    updates = []
    
    for article in articles_batch:
        article_id = article['id']
        url = article['url']
        source = article.get('source', '')
        
        update_worker_state(worker_id, status=f"Scraping {url[:60]}...")
        
        # Small delay between articles to avoid rate limiting
        time.sleep(DELAY_BETWEEN_ARTICLES)
        
        try:
            scrape_result = scrape_article_content(url)
            
            # Check if we hit a rate limit (429)
            if scrape_result and scrape_result.get('rate_limited'):
                # Queue for retry on next pass - don't count as failure yet
                with global_metrics_lock:
                    global_metrics['rate_limited_queue'].append({
                        'id': article_id,
                        'url': url,
                        'source': source
                    })
                    global_metrics['rate_limited_articles'] += 1
                
                update_worker_state(worker_id, status="Rate limited - queued for retry")
                # Don't add to updates, we'll retry this article on next pass
                continue
            
            if scrape_result and not scrape_result.get('rate_limited'):
                # Check if scrape_result contains an error
                if 'error' in scrape_result:
                    # Scraping failed with specific error
                    error_code = scrape_result.get('error_code', 'UNKNOWN')
                    error_type = scrape_result.get('error_type', 'Unknown Error')
                    error_msg = scrape_result.get('error', '')[:200]
                    
                    updates.append({
                        'article_id': article_id,
                        'success': False,
                        'error_msg': f"[{error_code}] {error_type}: {error_msg}"
                    })
                    
                    with global_metrics_lock:
                        global_metrics['failed_scrapes'] += 1
                else:
                    # Successfully scraped
                    content = scrape_result['content']
                    social_handles = scrape_result['social_handles']
                    
                    # Get or create media outlet and save social handles to it
                    domain = get_domain(url)
                    media_outlet_id = None
                    if domain:
                        try:
                            # Pass social handles to be saved/merged in media outlets table
                            media_outlet_id = upsert_media_outlet(domain, source, social_handles)
                        except Exception as e:
                            logging.error(f"Error upserting media outlet {domain}: {e}")
                    
                    # Only mark as successful if we got actual content, not just handles
                    if content and content.strip():
                        updates.append({
                            'article_id': article_id,
                            'content': content,
                            'social_data': social_handles,
                            'media_outlet_id': media_outlet_id,
                            'success': True
                        })
                        
                        with global_metrics_lock:
                            global_metrics['successful_scrapes'] += 1
                    else:
                        # No content retrieved, mark as failed even if we got handles
                        # Note: Media outlet handles were still saved above
                        updates.append({
                            'article_id': article_id,
                            'success': False,
                            'error_msg': '[NO_CONTENT] Empty content - article may be paywalled or format not supported'
                        })
                        
                        with global_metrics_lock:
                            global_metrics['failed_scrapes'] += 1
            else:
                # scrape_result is None - unexpected failure
                updates.append({
                    'article_id': article_id,
                    'success': False,
                    'error_msg': '[UNKNOWN] Failed to download article - unexpected error'
                })
                
                with global_metrics_lock:
                    global_metrics['failed_scrapes'] += 1
        
        except Exception as e:
            error_msg = str(e)[:200]
            logging.error(f"Worker {worker_id} error processing article {article_id}: {e}")
            recent_errors.append(f"[W{worker_id}] Article {article_id}: {error_msg}")
            
            updates.append({
                'article_id': article_id,
                'success': False,
                'error_msg': error_msg
            })
            
            with global_metrics_lock:
                global_metrics['failed_scrapes'] += 1
                global_metrics['total_articles_processed'] += 1
    
    return updates


def async_push_batch():
    """
    Asynchronously push pending updates to the database.
    Called from worker threads but serialized with semaphore.
    """
    with push_semaphore:
        try:
            with global_metrics_lock:
                if not global_metrics['pending_push']:
                    return
                
                updates_to_push = global_metrics['pending_push'][:]
                global_metrics['pending_push'] = []
            
            # Push to DB outside the lock
            update_articles_batch(updates_to_push)
            
        except Exception as e:
            logging.error(f"Error in async batch push: {e}")
            recent_errors.append(f"Batch push error: {str(e)[:100]}")


def worker_run(worker_id: int):
    """
    Main worker thread function.
    Continuously fetches and processes article batches.
    On subsequent passes, retries articles that hit rate limits.
    """
    update_worker_state(worker_id, status="Starting...")
    
    total_processed = 0
    pass_number = 0
    
    try:
        while True:
            pass_number += 1
            
            # On pass 2+, retry rate-limited articles first
            if pass_number > 1:
                with global_metrics_lock:
                    if global_metrics['rate_limited_queue']:
                        retry_articles = []
                        while global_metrics['rate_limited_queue'] and len(retry_articles) < BATCH_SIZE // 2:
                            article = global_metrics['rate_limited_queue'].popleft()
                            retry_articles.append({
                                'id': article['id'],
                                'url': article['url'],
                                'source': article['source'],
                                'published_at': None,
                                'company_id': None
                            })
                        
                        if retry_articles:
                            update_worker_state(worker_id, status=f"Retrying {len(retry_articles)} rate-limited...")
                            updates = process_article_batch(retry_articles, worker_id)
                            with global_metrics_lock:
                                global_metrics['pending_push'].extend(updates)
                                global_metrics['total_articles_processed'] += len(retry_articles)
                            
                            if len(global_metrics['pending_push']) >= PUSH_BATCH_SIZE:
                                async_push_batch()
            
            # Fetch next batch of articles to scrape
            update_worker_state(worker_id, status="Fetching batch...")
            articles_batch = get_articles_for_scraping(batch_size=BATCH_SIZE, max_retries=MAX_RETRIES)
            
            if not articles_batch:
                # No more articles to process
                update_worker_state(worker_id, status="No more articles")
                break
            
            batch_size = len(articles_batch)
            update_worker_state(
                worker_id,
                status=f"Processing {batch_size} articles...",
                articles_pending_push=batch_size
            )
            
            # Process each article in the batch
            updates = process_article_batch(articles_batch, worker_id)
            
            # Add updates to pending queue
            with global_metrics_lock:
                global_metrics['pending_push'].extend(updates)
                global_metrics['total_articles_processed'] += batch_size
            
            # Push batch if it exceeds threshold
            if len(global_metrics['pending_push']) >= PUSH_BATCH_SIZE:
                async_push_batch()
            
            update_worker_state(worker_id, articles_processed=total_processed + batch_size)
            total_processed += batch_size
            
            # Light delay between batches (HPC: many threads, less contention)
            time.sleep(0.5)
        
        # Push any remaining updates on worker exit
        async_push_batch()
        
    except Exception as e:
        error_msg = str(e)[:200]
        update_worker_state(worker_id, status=f"Fatal error: {error_msg}")
        logging.error(f"Worker {worker_id} fatal error: {e}", exc_info=True)
        recent_errors.append(f"[W{worker_id}] Fatal: {error_msg}")
    finally:
        update_worker_state(worker_id, status="Done")


# ============================================================================
# Progress Monitoring
# ============================================================================

def generate_layout() -> Group:
    """Generate the real-time progress display."""
    
    # Get progress metrics
    progress = get_scraping_progress()
    
    # Progress metrics panel
    metrics_text = f"""[bold cyan]Total Articles:[/bold cyan] {progress.get('total_articles', 0):,}
[bold green]Scraped:[/bold green] {progress.get('scraped_articles', 0):,}
[bold yellow]Pending:[/bold yellow] {progress.get('pending_articles', 0):,}
[bold magenta]Rate Limited (Queued):[/bold magenta] {global_metrics['rate_limited_articles']:,}
[bold red]Failed (Retryable):[/bold red] {progress.get('failed_articles', 0):,}
[bold magenta]Session Processed:[/bold magenta] {global_metrics['total_articles_processed']:,}
[bold blue]Successful:[/bold blue] {global_metrics['successful_scrapes']:,}
[bold red]Failed:[/bold red] {global_metrics['failed_scrapes']:,}"""
    
    metrics_panel = Panel(
        metrics_text,
        title="Scraping Progress",
        style="bold white",
        box=box.DOUBLE,
        expand=False
    )
    
    # Worker status table
    table = Table(title="Worker Status", box=box.ROUNDED)
    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("Status", style="yellow")
    table.add_column("Processed", justify="right", style="green")
    table.add_column("Pending Push", justify="right", style="blue")
    table.add_column("Errors", justify="right", style="red")
    
    with worker_states_lock:
        for worker_id in sorted(worker_states.keys()):
            state = worker_states[worker_id]
            table.add_row(
                str(worker_id),
                state.get('status', 'Idle'),
                str(state.get('articles_processed', 0)),
                str(state.get('articles_pending_push', 0)),
                str(state.get('errors', 0)),
            )
    
    # Pending push buffer info
    with global_metrics_lock:
        pending_count = len(global_metrics['pending_push'])
    
    buffer_text = f"[bold]Pending push buffer:[/bold] {pending_count} updates (batch threshold: {PUSH_BATCH_SIZE})"
    buffer_panel = Panel(buffer_text, title="DB Push Buffer", style="blue", box=box.ROUNDED)
    
    # Recent errors
    error_text = "\n".join(recent_errors) if recent_errors else "No errors"
    error_panel = Panel(
        error_text,
        title="Recent Errors",
        style="red",
        box=box.ROUNDED,
        height=5
    )
    
    return Group(metrics_panel, buffer_panel, table, error_panel)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    console = Console()
    console.print("[bold green]Starting High-Performance Content Scraper Pipeline (v2)[/bold green]")
    console.print(f"[bold cyan]HPC Configuration:[/bold cyan]")
    console.print(f"  Workers: {NUM_WORKERS} threads")
    console.print(f"  Batch Size: {BATCH_SIZE} articles/fetch")
    console.print(f"  DB Push Threshold: {PUSH_BATCH_SIZE}")
    console.print(f"  Request Timeout: {REQUEST_TIMEOUT}s")
    console.print(f"  Delay Between Articles: {DELAY_BETWEEN_ARTICLES}s")
    console.print(f"[bold yellow]Note:[/bold yellow] Configure via environment variables:")
    console.print(f"  NUM_WORKERS={NUM_WORKERS}, BATCH_SIZE={BATCH_SIZE}, PUSH_BATCH_SIZE={PUSH_BATCH_SIZE}, etc.")
    
    # Check initial progress
    progress = get_scraping_progress()
    console.print(f"[bold yellow]Initial state:[/bold yellow] {progress.get('pending_articles', 0):,} articles pending, "
                  f"{progress.get('failed_articles', 0):,} retryable failures")
    
    if progress.get('pending_articles', 0) == 0:
        console.print("[bold yellow]No articles pending scraping.[/bold yellow]")
        return
    
    # Initialize worker states
    for i in range(1, NUM_WORKERS + 1):
        update_worker_state(i, status="Waiting to start...")
    
    # Run workers
    is_interactive = sys.stdout.isatty()
    
    if is_interactive:
        console.print("[bold cyan]Running in interactive mode with Live dashboard.[/bold cyan]")
    else:
        # If running in background (nohup), disable fancy dashboard
        console.print("[bold yellow]Running in background mode. Live dashboard disabled. Printing periodic updates.[/bold yellow]")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS, thread_name_prefix="Worker") as executor:
        # Submit all worker tasks
        futures = [executor.submit(worker_run, i) for i in range(1, NUM_WORKERS + 1)]
        
        try:
            if is_interactive:
                # Interactive Mode: Use Rich Live Dashboard
                with Live(generate_layout(), refresh_per_second=2) as live:
                    while True:
                        live.update(generate_layout())
                        
                        if all(f.done() for f in futures):
                            break
                        
                        time.sleep(0.5)
            else:
                # Background Mode: TQDM Progress Bar
                from tqdm import tqdm
                
                # Initial progress fetch
                progress = get_scraping_progress()
                initial_processed = global_metrics['total_articles_processed']
                # Total estimate: pending + already processed in this session (which starts at 0) 
                # Ideally we want total articles to do.
                total_estimate = progress.get('total_articles', 0)
                initial_scraped_db = progress.get('scraped_articles', 0)
                
                with tqdm(total=total_estimate, initial=initial_scraped_db, unit="docs", 
                          desc="Scraping Progress", mininterval=5.0, file=sys.stdout, 
                          dynamic_ncols=True) as pbar:
                    
                    last_scraped_total = initial_scraped_db
                    
                    while True:
                        if all(f.done() for f in futures):
                            break
                        
                        current_progress = get_scraping_progress()
                        current_scraped_total = current_progress.get('scraped_articles', 0)
                        current_failed = current_progress.get('failed_articles', 0)
                        
                        delta = current_scraped_total - last_scraped_total
                        if delta > 0:
                            pbar.update(delta)
                            last_scraped_total = current_scraped_total
                        
                        # Update postfix with session stats
                        pbar.set_postfix({
                            'Session': f"{global_metrics['total_articles_processed']:,}",
                            'Success': f"{global_metrics['successful_scrapes']:,}",
                            'Fail': f"{global_metrics['failed_scrapes']:,}",
                            'RateLimit': f"{global_metrics['rate_limited_articles']:,}",
                            'Q': len(global_metrics['pending_push'])
                        })
                        
                        time.sleep(5)
            
            # Wait for all futures and collect exceptions
            for f in futures:
                try:
                    f.result()
                except Exception as e:
                    console.print(f"[bold red]Worker exception:[/bold red] {e}")
        
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Keyboard interrupt detected. Shutting down gracefully...[/bold yellow]")
            executor.shutdown(wait=True)
            console.print("[bold green]Shutdown complete.[/bold green]")
            return
    
    # Final statistics
    console.print("\n[bold green]All workers finished.[/bold green]")
    final_progress = get_scraping_progress()
    console.print(f"[bold cyan]Final state:[/bold cyan] {final_progress.get('scraped_articles', 0):,} articles scraped, "
                  f"{final_progress.get('pending_articles', 0):,} remaining")
    console.print(f"[bold magenta]Session summary:[/bold magenta]")
    console.print(f"  Processed: {global_metrics['total_articles_processed']:,}")
    console.print(f"  Successful: {global_metrics['successful_scrapes']:,}")
    console.print(f"  Failed: {global_metrics['failed_scrapes']:,}")
    console.print(f"  Rate Limited (queued for next run): {global_metrics['rate_limited_articles']:,}")
    
    if global_metrics['rate_limited_queue']:
        console.print(f"\n[bold yellow]Note:[/bold yellow] {len(global_metrics['rate_limited_queue']):,} articles still in rate-limited queue.")
        console.print(f"  Run the script again to retry them after cooldown period.")
    
    success_rate = (global_metrics['successful_scrapes'] / global_metrics['total_articles_processed'] * 100) if global_metrics['total_articles_processed'] > 0 else 0
    console.print(f"\n[bold blue]Success Rate:[/bold blue] {success_rate:.1f}%")


if __name__ == "__main__":
    main()
