import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import threading
import json
from urllib.parse import urlparse
import time

load_dotenv()

# Global lock for serializing dictionary table upserts
_upsert_lock = threading.Lock()

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def get_domain(url):
    try:
        return urlparse(url).netloc.replace('www.', '')
    except:
        return None

def _resolve_media_outlets(conn, domains_map):
    """
    Resolves IDs for a list of domains. Upserts if missing.
    domains_map: dict of domain -> media_name
    Returns: dict of domain -> id
    """
    if not domains_map:
        return {}
    
    resolved = {}
    cursor = conn.cursor()
    
    try:
        # 1. Fetch existing
        domains = list(domains_map.keys())
        cursor.execute(
            "SELECT domain, id FROM media_outlets WHERE domain = ANY(%s)",
            (domains,)
        )
        for row in cursor.fetchall():
            resolved[row[0]] = row[1]
            
        # 2. Insert missing
        missing_domains = [d for d in domains if d not in resolved]
        if missing_domains:
            values = [(d, domains_map[d], json.dumps({})) for d in missing_domains]
            
            # We use ON CONFLICT DO NOTHING to be safe, then fetch again
            execute_values(
                cursor,
                """
                INSERT INTO media_outlets (domain, name, social_handles)
                VALUES %s
                ON CONFLICT (domain) DO NOTHING
                """,
                values
            )
            
            # Fetch the newly inserted ones (and any that might have been inserted by race if lock wasn't perfect)
            cursor.execute(
                "SELECT domain, id FROM media_outlets WHERE domain = ANY(%s)",
                (missing_domains,)
            )
            for row in cursor.fetchall():
                resolved[row[0]] = row[1]
                
        conn.commit()
        return resolved
    except Exception as e:
        print(f"Error resolving media outlets: {e}")
        conn.rollback()
        return resolved

def _resolve_companies(conn, companies_map):
    """
    Resolves IDs for a list of companies. Upserts if missing.
    companies_map: dict of symbol -> company_name
    Returns: dict of symbol -> id
    """
    if not companies_map:
        return {}
        
    resolved = {}
    cursor = conn.cursor()
    
    try:
        # 1. Fetch existing
        symbols = list(companies_map.keys())
        cursor.execute(
            "SELECT symbol, id FROM companies WHERE symbol = ANY(%s)",
            (symbols,)
        )
        for row in cursor.fetchall():
            resolved[row[0]] = row[1]
            
        # 2. Insert missing
        missing_symbols = [s for s in symbols if s not in resolved]
        if missing_symbols:
            values = [(companies_map[s], s) for s in missing_symbols]
            
            execute_values(
                cursor,
                """
                INSERT INTO companies (name, symbol)
                VALUES %s
                ON CONFLICT (symbol) DO NOTHING
                """,
                values
            )
            
            cursor.execute(
                "SELECT symbol, id FROM companies WHERE symbol = ANY(%s)",
                (missing_symbols,)
            )
            for row in cursor.fetchall():
                resolved[row[0]] = row[1]
                
        conn.commit()
        return resolved
    except Exception as e:
        print(f"Error resolving companies: {e}")
        conn.rollback()
        return resolved

def insert_articles(articles_data):
    """
    Insert a list of article dictionaries into the database.
    Also upserts media outlets and companies, linking them.
    Uses locking to prevent deadlocks on dictionary tables.
    """
    if not articles_data:
        return

    conn = get_db_connection()
    if not conn:
        return

    try:
        # 1. Prepare batch data
        domains_to_resolve = {} # domain -> name
        companies_to_resolve = {} # symbol -> name
        
        for article in articles_data:
            url = article.get('url')
            domain = get_domain(url)
            if domain:
                domains_to_resolve[domain] = article.get('media_name')
                
            sym = article.get('stock_symbol')
            if sym:
                companies_to_resolve[sym] = article.get('company_name')

        # 2. Resolve IDs under lock
        # This serializes the "write" part of dictionary tables, preventing deadlocks
        # caused by interleaved transactions trying to insert the same keys.
        with _upsert_lock:
            outlet_map = _resolve_media_outlets(conn, domains_to_resolve)
            company_map = _resolve_companies(conn, companies_to_resolve)

        # 3. Prepare article inserts (no lock needed, just DB transaction)
        values = []
        for article in articles_data:
            url = article.get('url')
            domain = get_domain(url)
            sym = article.get('stock_symbol')
            
            outlet_id = outlet_map.get(domain)
            company_id = company_map.get(sym)
            
            values.append((
                article.get('title'),
                None, # Content
                url,
                article.get('media_name'),
                article.get('publish_date'),
                outlet_id,
                company_id
            ))
        
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO articles (title, content, url, source, published_at, media_outlet_id, company_id)
            VALUES %s
            ON CONFLICT (url) DO NOTHING
        """
        
        execute_values(cursor, insert_query, values)
        conn.commit()
        cursor.close()
        
    except Exception as e:
        print(f"Error inserting articles batch: {e}")
        conn.rollback()
    finally:
        if conn:
            conn.close()

def get_articles_without_content(limit=100):
    """Fetch articles that have no content."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query = "SELECT * FROM articles WHERE content IS NULL OR content = '' LIMIT %s"
        cursor.execute(query, (limit,))
        articles = cursor.fetchall()
        return articles
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return []
    finally:
        if conn:
            cursor.close()
            conn.close()

def update_article_content(article_id, content, media_outlet_id, social_data_json):
    """Update article with scraped content and linked outlet."""
    conn = get_db_connection()
    if not conn:
        return
        
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE articles 
            SET content = %s, media_outlet_id = %s, social_data = %s
            WHERE id = %s
            """,
            (content, media_outlet_id, social_data_json, article_id)
        )
        conn.commit()
    except Exception as e:
        print(f"Error updating article {article_id}: {e}")
        conn.rollback()
    finally:
            cursor.close()
            conn.close()

def seed_sp500_companies(companies_df):
    """
    Insert or update SP500 companies in the database with default state.
    This should be run initially and periodically to keep the company list updated.
    
    Args:
        companies_df: DataFrame with 'Symbol' and 'Company' columns
    """
    if companies_df is None or companies_df.empty:
        print("No companies to seed")
        return
    
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        values = [
            (row['Company'], row['Symbol'], 0, False, None)
            for _, row in companies_df.iterrows()
        ]
        
        # Upsert companies: if symbol exists, update name but preserve state
        # If new, insert with default state
        execute_values(
            cursor,
            """
            INSERT INTO companies (name, symbol, current_page, is_processed, last_error)
            VALUES %s
            ON CONFLICT (symbol) 
            DO UPDATE SET name = EXCLUDED.name
            """,
            values
        )
        
        conn.commit()
        print(f"Seeded {len(values)} companies into database")
    except Exception as e:
        print(f"Error seeding companies: {e}")
        conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()

def get_unprocessed_companies(limit=None):
    """
    Fetch companies that haven't been fully processed yet.
    Returns a list of dicts with company info and state.
    
    Args:
        limit: Optional limit on number of companies to fetch
    
    Returns:
        List of dicts with keys: id, name, symbol, current_page, is_processed, last_error
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        if limit:
            query = """
                SELECT id, name, symbol, current_page, is_processed, last_error, last_updated
                FROM companies 
                WHERE is_processed = FALSE
                ORDER BY last_updated ASC NULLS FIRST
                LIMIT %s
            """
            cursor.execute(query, (limit,))
        else:
            query = """
                SELECT id, name, symbol, current_page, is_processed, last_error, last_updated
                FROM companies 
                WHERE is_processed = FALSE
                ORDER BY last_updated ASC NULLS FIRST
            """
            cursor.execute(query)
        
        companies = cursor.fetchall()
        return companies
    except Exception as e:
        print(f"Error fetching unprocessed companies: {e}")
        return []
    finally:
        if conn:
            cursor.close()
            conn.close()

def update_company_state(symbol, current_page=None, is_processed=None, last_error=None):
    """
    Update the processing state for a company.
    
    Args:
        symbol: Company stock symbol
        current_page: Current page number being processed (optional)
        is_processed: Whether processing is complete (optional)
        last_error: Last error message if any (optional)
    """
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Build dynamic update query based on provided parameters
        updates = []
        params = []
        
        if current_page is not None:
            updates.append("current_page = %s")
            params.append(current_page)
        
        if is_processed is not None:
            updates.append("is_processed = %s")
            params.append(is_processed)
        
        if last_error is not None:
            updates.append("last_error = %s")
            params.append(last_error)
        
        if not updates:
            return  # Nothing to update
        
        params.append(symbol)
        query = f"UPDATE companies SET {', '.join(updates)} WHERE symbol = %s"
        
        cursor.execute(query, params)
        conn.commit()
    except Exception as e:
        print(f"Error updating company state for {symbol}: {e}")
        conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()

def mark_company_complete(symbol):
    """
    Mark a company as fully processed.
    
    Args:
        symbol: Company stock symbol
    """
    update_company_state(symbol, is_processed=True, last_error=None)

def reset_company_state(symbol):
    """
    Reset a company's processing state (useful for reprocessing).
    
    Args:
        symbol: Company stock symbol
    """
    update_company_state(symbol, current_page=0, is_processed=False, last_error=None)


# ============================================================================
# Content Scraper Helper Functions
# ============================================================================

def get_articles_for_scraping(batch_size=100, max_retries=3):
    """
    Fetch a batch of articles that need content scraping.
    Prioritizes articles that haven't been scraped yet, then retries based on retry count.
    Uses FOR UPDATE SKIP LOCKED to avoid contention between concurrent workers.
    
    Args:
        batch_size: Number of articles to fetch per batch
        max_retries: Maximum retry attempts for failed articles
    
    Returns:
        List of article dicts with keys: id, title, url, source, company_id, published_at
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Fetch articles without content, using row-level locking to prevent race conditions
        # SKIP LOCKED ensures we don't wait if another worker is processing the row
        query = """
            SELECT id, title, url, source, company_id, published_at
            FROM articles
            WHERE content IS NULL 
              AND content_scraped = FALSE
              AND scraping_retry_count < %s
              AND (last_scraped_at IS NULL OR last_scraped_at < NOW() - INTERVAL '1 hour')
            ORDER BY last_scraped_at ASC NULLS FIRST, id ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        """
        
        cursor.execute(query, (max_retries, batch_size))
        articles = cursor.fetchall()
        
        # Commit immediately to release locks on rows we're not fetching
        conn.commit()
        return articles
    except Exception as e:
        print(f"Error fetching articles for scraping: {e}")
        conn.rollback()
        return []
    finally:
        if conn:
            cursor.close()
            conn.close()


def update_articles_batch(updates_list):
    """
    Update multiple articles with scraped content and metadata.
    Batches all updates into a single transaction for efficiency.
    
    Args:
        updates_list: List of dicts with keys:
                     - article_id: int
                     - content: str
                     - social_data: dict (will be converted to JSON)
                     - media_outlet_id: int (optional)
                     - success: bool (True if scraping succeeded)
                     - error_msg: str (optional, only if success=False)
    """
    if not updates_list:
        return
    
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        for update in updates_list:
            article_id = update['article_id']
            success = update.get('success', True)
            
            if success:
                # Update with scraped content
                social_json = json.dumps(update.get('social_data', {}))
                cursor.execute(
                    """
                    UPDATE articles
                    SET content = %s,
                        social_data = %s,
                        media_outlet_id = %s,
                        content_scraped = TRUE,
                        last_scraped_at = NOW(),
                        scraping_retry_count = 0,
                        scraping_error = NULL
                    WHERE id = %s
                    """,
                    (update.get('content', ''), social_json, update.get('media_outlet_id'), article_id)
                )
            else:
                # Update with error information
                error_msg = update.get('error_msg', 'Unknown error')[:500]
                cursor.execute(
                    """
                    UPDATE articles
                    SET last_scraped_at = NOW(),
                        scraping_retry_count = scraping_retry_count + 1,
                        scraping_error = %s
                    WHERE id = %s
                    """,
                    (error_msg, article_id)
                )
        
        conn.commit()
    except Exception as e:
        print(f"Error updating articles batch: {e}")
        conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()


def get_scraping_progress():
    """
    Fetch overall scraping progress metrics.
    
    Returns:
        Dict with keys: total_articles, scraped_articles, pending_articles, failed_articles
    """
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN content_scraped = TRUE THEN 1 ELSE 0 END) as scraped,
                SUM(CASE WHEN content_scraped = FALSE AND content IS NULL THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN scraping_retry_count > 0 AND content_scraped = FALSE THEN 1 ELSE 0 END) as failed
            FROM articles
        """)
        
        result = cursor.fetchone()
        return {
            'total_articles': result[0] or 0,
            'scraped_articles': result[1] or 0,
            'pending_articles': result[2] or 0,
            'failed_articles': result[3] or 0
        }
    except Exception as e:
        print(f"Error fetching scraping progress: {e}")
        return {}
    finally:
        if conn:
            cursor.close()
            conn.close()


def upsert_media_outlet(domain, name, social_handles_dict=None):
    """
    Upsert a media outlet into the database and return its ID.
    Merges new social handles with existing ones.
    
    Args:
        domain: Domain name (unique identifier)
        name: Display name of the outlet
        social_handles_dict: Dict of social handles (e.g., {'twitter': ['handle1'], 'instagram': ['handle2']})
    
    Returns:
        ID of the inserted or existing media outlet
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        
        # First, check if outlet exists and get current handles
        cursor.execute(
            "SELECT id, social_handles FROM media_outlets WHERE domain = %s",
            (domain,)
        )
        existing = cursor.fetchone()
        
        if existing:
            outlet_id = existing[0]
            existing_handles = existing[1] if existing[1] else {}
            
            # Merge new handles with existing ones
            if social_handles_dict:
                merged_handles = dict(existing_handles)
                for platform, handles in social_handles_dict.items():
                    if platform in merged_handles:
                        # Merge handles, avoiding duplicates
                        existing_list = merged_handles[platform] if isinstance(merged_handles[platform], list) else [merged_handles[platform]]
                        new_list = handles if isinstance(handles, list) else [handles]
                        merged_handles[platform] = list(set(existing_list + new_list))
                    else:
                        merged_handles[platform] = handles if isinstance(handles, list) else [handles]
                
                # Update with merged handles
                cursor.execute(
                    "UPDATE media_outlets SET name = %s, social_handles = %s WHERE id = %s",
                    (name, json.dumps(merged_handles), outlet_id)
                )
            else:
                # Just update name if no new handles
                cursor.execute(
                    "UPDATE media_outlets SET name = %s WHERE id = %s",
                    (name, outlet_id)
                )
        else:
            # Insert new outlet
            social_json = json.dumps(social_handles_dict) if social_handles_dict else '{}'
            cursor.execute(
                """
                INSERT INTO media_outlets (domain, name, social_handles)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (domain, name, social_json)
            )
            result = cursor.fetchone()
            outlet_id = result[0] if result else None
        
        conn.commit()
        return outlet_id
    except Exception as e:
        print(f"Error upserting media outlet {domain}: {e}")
        conn.rollback()
        return None
    finally:
        if conn:
            cursor.close()
            conn.close()

