import psycopg2
import os
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from threading import Lock

# Load environment variables
load_dotenv()

# Configuration
PREF = 41_000_000
GAMMA = 1.0
WINDOW_DAYS = 14
NUM_THREADS = 10
TEST_MODE = False

console = Console()
print_lock = Lock()
results_lock = Lock()

# Global Cache for Outlet Data
outlet_cache = {}  # id -> {'stance': float, 'followers': float}

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432")
    )

def normalize_followers(followers):
    """
    norm(x) = ( log10(1 + x) / log10(1 + PREF) ) ^ GAMMA
    """
    if followers is None:
        followers = 0
    
    # Ensure non-negative
    followers = float(max(0, followers))
    
    numerator = math.log10(1 + followers)
    denominator = math.log10(1 + PREF)
    
    if denominator == 0:
        return 0.0
        
    return math.pow(numerator / denominator, GAMMA)

def calculate_article_sentiment(pos, neg):
    """
    Sa = pos - neg
    """
    if pos is None: pos = 0.0
    if neg is None: neg = 0.0
    
    return pos - neg

def preload_media_outlets():
    """Fetches all media outlets and caches their stance and followers."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, stance_score, avg_followers FROM media_outlets")
            rows = cur.fetchall()
            for r in rows:
                oid, stance, followers = r
                outlet_cache[oid] = {
                    'stance': float(stance) if stance is not None else 0.0,
                    'followers': float(followers) if followers is not None else 0.0
                }
        console.print(f"[green]Loaded {len(outlet_cache)} media outlets into cache.[/green]")
    finally:
        conn.close()

def process_company(ticker, bias_dates, progress, task_id):
    """
    Process a single company using Sliding Window Optimization.
    bias_dates: List of (id, date) tuples from bias_index, assumed sorted by date.
    """
    conn = get_db_connection()
    results = [] # List of (ticker, row_id, bias_score, date, article_count)
    
    try:
        if not bias_dates:
            return []

        # 1. Fetch ALL articles for this company in the relevant range
        # Range: [min_date - 14 days, max_date]
        min_date = bias_dates[0][1]
        max_date = bias_dates[-1][1]
        
        # We need datetime objects for comparison
        # Ensure bias_dates use datetime (postgres returns datetime usually)
        
        query = """
            SELECT a.published_at, a.pos_score, a.neg_score, a.media_outlet_id
            FROM articles a
            JOIN companies c ON a.company_id = c.id
            WHERE c.symbol = %s
            AND a.published_at >= %s - INTERVAL '%s days' 
            AND a.published_at <= %s + INTERVAL '1 day'
            AND a.media_outlet_id IS NOT NULL
            ORDER BY a.published_at ASC
        """
        with conn.cursor() as cur:
            # We use a slight buffer in dates to be safe regarding timezones/truncation
            cur.execute(query, (ticker, min_date, WINDOW_DAYS + 1, max_date))
            all_raw_articles = cur.fetchall()

        console.log(f"[green]Fetched {len(all_raw_articles)} articles for {ticker}[/green]")

        # 2. Pre-calculate Contributions
        # List of (timestamp, contribution)
        article_data = []
        for pub_at, pos, neg, outlet_id in all_raw_articles:
            outlet_data = outlet_cache.get(outlet_id)
            if not outlet_data:
                continue
                
            s_a = calculate_article_sentiment(pos, neg)
            # s_o = outlet_data['stance'] # Removed as per request (Shock term removed)
            p_o = outlet_data['followers']
            norm_pop = normalize_followers(p_o) # Removed as per request ("remove all other components")
            
            # New Formula: Sum S_a . P_o
            contribution = s_a * norm_pop
            article_data.append((pub_at, contribution))
            
        # 3. Sliding Window Calculation
        
        left_idx = 0
        right_idx = 0
        current_sum = 0.0
        current_count = 0
        n_articles = len(article_data)
        
        from datetime import timedelta
        
        for row_id, target_date in bias_dates: 
            target_dt = target_date
            if hasattr(target_date, 'date'):
                target_dt_date = target_date.date()
            else:
                target_dt_date = target_date
                
            # Window Start
            window_start_date = target_dt_date - timedelta(days=WINDOW_DAYS)
            
            # Advance Right Pointer (Add new articles)
            while right_idx < n_articles:
                art_dt, contrib = article_data[right_idx]
                art_date = art_dt.date()
                
                if art_date <= target_dt_date:
                    if art_date >= window_start_date: # Only add if within start too (optimization)
                        current_sum += contrib
                        current_count += 1
                    right_idx += 1
                else:
                    break
            
            # Since we only added stuff <= target in Right loop, we just need to ensure Left catches up.
            while left_idx < right_idx:
                art_dt, contrib = article_data[left_idx]
                art_date = art_dt.date()
                
                if art_date < window_start_date:
                    pass 
                else:
                    break
                left_idx += 1
            
            window_slice = article_data[left_idx:right_idx]
            
            daily_sum = sum(x[1] for x in window_slice)
            daily_count = len(window_slice)
            
            results.append((ticker, row_id, daily_sum, target_date, daily_count))

        # Batch Update if not Test Mode
        if not TEST_MODE and results:
            with conn.cursor() as cur:
                update_query = """
                    UPDATE bias_index 
                    SET bias_score = %s 
                    WHERE id = %s
                """
                # results: (ticker, row_id, bias_score, date, count)
                data_to_update = [(r[2], r[1]) for r in results]
                psycopg2.extras.execute_batch(cur, update_query, data_to_update)
                conn.commit()
                
        # Update progress
        progress.advance(task_id, advance=len(bias_dates))
        return results

    except Exception as e:
        with print_lock:
            console.print(f"[red]Error processing {ticker}: {e}[/red]")
            import traceback
            traceback.print_exc()
        return []
    finally:
        conn.close()

def main():
    global TEST_MODE
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod", action="store_true", help="Run in production mode (write to DB)")
    args = parser.parse_args()
    
    if args.prod:
        TEST_MODE = False
    
    mode_str = "[red]PRODUCTION[/red]" if not TEST_MODE else "[yellow]TEST (Dry Run)[/yellow]"
    console.print(f"[bold]Starting Bias Score Pipeline - Mode: {mode_str}[/bold]")
    console.print(f"Norm Params: PREF={PREF}, GAMMA={GAMMA}")
    
    # 1. Preload Outlets
    preload_media_outlets()
    
    conn = get_db_connection()
    try:
        # 2. Get Companies to Process
        # Group by ticker to assign to threads
        with conn.cursor() as cur:
            if TEST_MODE:
                console.print("Fetching random 10 companies for testing...")
                cur.execute("SELECT DISTINCT ticker FROM bias_index")
                all_tickers = [r[0] for r in cur.fetchall()]
                if not all_tickers:
                    console.print("[red]No data in bias_index![/red]")
                    return
                selected_tickers = random.sample(all_tickers, min(10, len(all_tickers)))
            else:
                console.print("Fetching ALL companies...")
                cur.execute("SELECT DISTINCT ticker FROM bias_index")
                selected_tickers = [r[0] for r in cur.fetchall()]
        
        console.print(f"Processing {len(selected_tickers)} companies...")
        
        # Prepare Tasks: Dictionary ticker -> list of (id, date)
        tasks = {}
        with conn.cursor() as cur:
            for ticker in selected_tickers:
                # Fetch target rows
                q = "SELECT id, date FROM bias_index WHERE ticker = %s"
                q += " ORDER BY date"
                
                cur.execute(q, (ticker,))
                rows = cur.fetchall()
                if rows:
                    tasks[ticker] = rows
        
        conn.close() # Close main thread conn
        
        # 3. Execution
        collected_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            main_task = progress.add_task("[cyan]Processing Companies...", total=sum(len(rows) for rows in tasks.values()))
            
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = []
                for ticker, rows in tasks.items():
                    futures.append(
                        executor.submit(process_company, ticker, rows, progress, main_task)
                    )
                
                for future in futures:
                    res = future.result()
                    if TEST_MODE and res:
                        collected_results.extend(res)

        # 4. Output (Test Mode)
        if TEST_MODE:
            table = Table(title=f"Bias Score Preview (Top 50 rows of {len(collected_results)})")
            table.add_column("Ticker", style="cyan")
            table.add_column("Date", style="magenta")
            table.add_column("Articles", style="green")
            table.add_column("Bias Score", style="bold yellow")
            
            # Sort by date (index 3)
            collected_results.sort(key=lambda x: x[3]) 
            
            for res in collected_results[:50]:
                # (ticker, row_id, bias_score, date, count)
                ticker = res[0]
                date_str = str(res[3])
                count = str(res[4])
                score = f"{res[2]:.4f}"
                
                table.add_row(ticker, date_str, count, score)
                
            console.print(table)

    except Exception as e:
        console.print(f"[red]Critical Error: {e}[/red]")

if __name__ == "__main__":
    # Import execution_extras here if needed
    import psycopg2.extras
    main()
