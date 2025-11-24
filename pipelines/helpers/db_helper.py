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
        if conn:
            cursor.close()
            conn.close()
