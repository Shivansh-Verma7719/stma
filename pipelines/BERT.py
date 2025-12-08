import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import psycopg2
from psycopg2.extras import execute_values
import os
import sys
import time
import threading
import queue
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console, Group
from rich import box
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
NUM_WORKERS = 10               # Number of worker threads (Pre/Post processing)
BATCH_SIZE = 64               # Articles per inference batch
PRODUCER_BATCH_SIZE = 2000    # Articles fetched from DB per chunk
PUSH_BATCH_SIZE = 500         # Articles to buffer before DB write
MODEL_NAME = "ProsusAI/finbert"

# Logging setup
console = Console()
log_lock = Lock()

# Global State
article_queue = queue.Queue(maxsize=PRODUCER_BATCH_SIZE * 2)
pending_push_queue = []       # List of results pending DB write
push_lock = Lock()            # Lock for DB writing
model_lock = Lock()           # Lock for GPU/MPS inference (Critical for Metal safety)
worker_states = {}            # Dict to track worker status
metrics = {
    "total_fetched": 0,
    "processed": 0,
    "pushed": 0,
    "errors": 0,
    "start_time": time.time()
}
metrics_lock = Lock()

# Check Device
if torch.backends.mps.is_available():
    device = "mps"
    process_device = 0
    print("Using MPS (Metal) acceleration on Mac.")
elif torch.cuda.is_available():
    device = "cuda:0"
    process_device = 0
    print("Using CUDA acceleration.")
else:
    device = "cpu"
    process_device = -1
    print("Using CPU.")

# Initialize Model Globally (One instance shared across threads)
print(f"Loading Model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# Move model to device manually to be sure
if device == 'mps':
    model.to('mps')
elif device.startswith('cuda'):
    model.to('cuda')

# Initialize pipeline
# Note: We set device depending on valid input for pipeline.
# For mps, usually passing device=model.device works best.
sentiment_pipe = pipeline(
    "sentiment-analysis", 
    model=model, 
    tokenizer=tokenizer, 
    device=model.device, # Use the model's device
    batch_size=BATCH_SIZE,
    truncation=True, 
    max_length=512,
    top_k=None # Return all scores
)

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432")
    )

# Producer Thread: Fetches articles from DB
def producer_run():
    conn = None
    try:
        conn = get_db_connection()
        # count total first
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM articles WHERE pos_score IS NULL")
            total = cur.fetchone()[0]
            with metrics_lock:
                metrics['total_to_do'] = total

        cur = conn.cursor(name='fetch_cursor') # Server-side cursor
        cur.execute("SELECT id, title, content FROM articles WHERE pos_score IS NULL")
        
        while True:
            rows = cur.fetchmany(PRODUCER_BATCH_SIZE)
            if not rows:
                break
            
            for row in rows:
                article_queue.put(row)
            
            with metrics_lock:
                metrics['total_fetched'] += len(rows)
            
            # Flow control - don't overfill memory if workers are slow
            while article_queue.qsize() > PRODUCER_BATCH_SIZE:
                time.sleep(1)
                
    except Exception as e:
        with log_lock:
            console.print(f"[red]Producer Error: {e}[/red]")
    finally:
        # Signal finish
        for _ in range(NUM_WORKERS):
            article_queue.put(None)
        if conn:
            conn.close()

# Worker Thread: Batch Inference
def worker_run(worker_id):
    worker_states[worker_id] = "Starting"
    local_batch = []
    
    while True:
        try:
            item = article_queue.get(timeout=2)
        except queue.Empty:
            continue
            
        if item is None:
            # Sentinel received, process remaining batch then exit
            if local_batch:
                process_batch(worker_id, local_batch)
            worker_states[worker_id] = "Done"
            article_queue.task_done()
            break
            
        local_batch.append(item)
        article_queue.task_done()
        
        if len(local_batch) >= BATCH_SIZE:
            process_batch(worker_id, local_batch)
            local_batch = []

def process_batch(worker_id, batch):
    worker_states[worker_id] = f"Inference ({len(batch)})"
    
    # Prepare text
    texts = []
    ids = []
    for row in batch:
        aid, title, content = row
        # Fallback to title if content missing
        text_parts = [title]
        if content and len(content.strip()) > 0:
            text_parts.append(content)
        full = " ".join(text_parts)[:2000] # Pre-truncate to avoid massive string ops
        texts.append(full)
        ids.append(aid)
        
    try:
        # Critical Section: GPU Inference
        # We lock to ensure Metal/CUDA doesn't get concurrent calls from multiple threads
        # destroying the context.
        with model_lock:
            # pipeline returns List[List[Dict]] because top_k=None (all scores)
            # e.g. [[{'label': 'positive', 'score': 0.9}, ...], ...]
            outputs = sentiment_pipe(texts)
            
        # Post-process (CPU)
        updates = []
        for idx, scores in enumerate(outputs):
            # scores is list of dicts: [{'label': 'positive', 'score':...}, ...]
            # FinBERT labels: 'positive', 'negative', 'neutral'
            score_map = {s['label']: s['score'] for s in scores}
            pos = score_map.get('positive', 0.0)
            neg = score_map.get('negative', 0.0)
            updates.append((pos, neg, ids[idx]))
            
        # Buffer for write
        with push_lock:
            pending_push_queue.extend(updates)
            should_push = len(pending_push_queue) >= PUSH_BATCH_SIZE
        
        if should_push:
            flush_to_db()
            
        with metrics_lock:
            metrics['processed'] += len(batch)
            
    except Exception as e:
        with log_lock:
            console.print(f"[red]Worker {worker_id} Error: {e}[/red]")
        with metrics_lock:
            metrics['errors'] += 1

# DB Writer Utility
def flush_to_db():
    # Calling function should acquire context if needed, but here we just take the lock 
    # to pop the buffer, then write outside lock if possible? 
    # Actually, simply locking the extract-and-write is safer.
    
    with push_lock:
        if not pending_push_queue:
            return
        to_push = pending_push_queue[:]
        pending_push_queue.clear()
        
    # Write to DB
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            execute_values(
                cur,
                "UPDATE articles SET pos_score = data.pos, neg_score = data.neg FROM (VALUES %s) AS data (pos, neg, id) WHERE articles.id = data.id",
                to_push
            )
            conn.commit()
        
        with metrics_lock:
            metrics['pushed'] += len(to_push)
            
    except Exception as e:
        with log_lock:
            console.print(f"[red]DB Write Error: {e}[/red]")
            # Put back? Risks infinite loop. Log and drop for now.
    finally:
        if conn:
            conn.close()

# Dashboard
def generate_layout():
    with metrics_lock:
        total = metrics.get('total_to_do', 0)
        processed = metrics['processed']
        pushed = metrics['pushed']
        errors = metrics['errors']
        elapsed = time.time() - metrics['start_time']
    
    rate = processed / elapsed if elapsed > 0 else 0
    remaining = total - processed if total else 0
    eta = remaining / rate if rate > 0 else 0
    
    # Grid
    layout = Table.grid(expand=True)
    layout.add_row(Panel(
        f"[bold cyan]Total Articles:[/bold cyan] {total:,}\n"
        f"[bold green]Processed:[/bold green] {processed:,} ({processed/max(1,total)*100:.1f}%)\n"
        f"[bold blue]Pushed to DB:[/bold blue] {pushed:,}\n"
        f"[bold red]Errors:[/bold red] {errors}\n"
        f"[bold yellow]Rate:[/bold yellow] {rate:.1f} articles/sec\n"
        f"[bold white]ETA:[/bold white] {eta/3600:.1f} hours",
        title="Global Metrics",
        style="white"
    ))
    
    # Worker Table
    w_table = Table(title="Worker Status", box=box.SIMPLE)
    w_table.add_column("ID", style="cyan")
    w_table.add_column("Status", style="green")
    
    for wid, status in worker_states.items():
        w_table.add_row(str(wid), status)
        
    layout.add_row(w_table)
    return layout

# Main
def main():
    console.print(f"[bold green]Starting Multithreaded Sentiment Pipeline[/bold green]")
    console.print(f"Workers: {NUM_WORKERS} | Batch: {BATCH_SIZE} | Device: {device}")
    
    # Start Producer
    prod_thread = threading.Thread(target=producer_run, daemon=True)
    prod_thread.start()
    
    # Start Workers
    threads = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=worker_run, args=(i,))
        t.start()
        threads.append(t)
        
    # Live Dashboard or Tqdm
    IS_HEADLESS = not sys.stdout.isatty() or os.getenv("HEADLESS") == "1"

    if IS_HEADLESS:
        print("Headless mode detected. Using tqdm for progress.")
        # Wait slightly for producer to potentially get total count
        time.sleep(2)
        
        with metrics_lock:
             total = metrics.get('total_to_do', 0)
             
        with tqdm(total=total, unit="articles") as pbar:
            last_processed = 0
            while any(t.is_alive() for t in threads):
                with metrics_lock:
                    current = metrics['processed']
                    errors = metrics['errors']
                
                diff = current - last_processed
                if diff > 0:
                    pbar.update(diff)
                    last_processed = current
                
                pbar.set_postfix(errors=errors)
                time.sleep(1)
            
            # Final update
            with metrics_lock:
                current = metrics['processed']
            diff = current - last_processed
            if diff > 0:
                pbar.update(diff)
    else:
        with Live(generate_layout(), refresh_per_second=4) as live:
            while any(t.is_alive() for t in threads):
                live.update(generate_layout())
                time.sleep(0.5)
            
    # Final flush
    flush_to_db()
    console.print("[bold green]Processing Complete![/bold green]")

if __name__ == "__main__":
    main()