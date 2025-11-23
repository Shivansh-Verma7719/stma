import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

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

from urllib.parse import urlparse
import json

def get_domain(url):
    try:
        return urlparse(url).netloc.replace('www.', '')
    except:
        return None

def insert_articles(articles_data):
    """
    Insert a list of article dictionaries into the database.
    Also upserts media outlets and links them.
    
    Args:
        articles_data: List of dictionaries containing article data
    """
    if not articles_data:
        return

    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        
        # Prepare data for execute_values
        values = []
        for article in articles_data:
            url = article.get('url')
            media_name = article.get('media_name')
            
            # 1. Upsert Media Outlet
            domain = get_domain(url)
            outlet_id = None
            if domain:
                # Check/Insert Outlet
                cursor.execute("SELECT id FROM media_outlets WHERE domain = %s", (domain,))
                result = cursor.fetchone()
                if result:
                    outlet_id = result[0]
                else:
                    cursor.execute(
                        "INSERT INTO media_outlets (domain, name, social_handles) VALUES (%s, %s, %s) RETURNING id",
                        (domain, media_name, json.dumps({}))
                    )
                    outlet_id = cursor.fetchone()[0]
            
            values.append((
                article.get('title'),
                None, # Content is not present initially; scraped later
                url,
                media_name,
                article.get('publish_date'),
                outlet_id
            ))
        
        # Prepare the INSERT statement
        insert_query = """
            INSERT INTO articles (title, content, url, source, published_at, media_outlet_id)
            VALUES %s
            ON CONFLICT (url) DO NOTHING
        """
        
        execute_values(cursor, insert_query, values)
        conn.commit()
        print(f"Successfully inserted/processed {len(values)} articles.")
        
    except Exception as e:
        print(f"Error inserting articles: {e}")
        conn.rollback()
    finally:
        if conn:
            cursor.close()
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

def upsert_media_outlet(domain, name, social_handles_json):
    """
    Insert or update a media outlet.
    Returns the ID of the outlet.
    """
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        cursor = conn.cursor()
        # Try to find existing first
        cursor.execute("SELECT id FROM media_outlets WHERE domain = %s", (domain,))
        result = cursor.fetchone()
        
        if result:
            outlet_id = result[0]
            return outlet_id
        else:
            cursor.execute(
                "INSERT INTO media_outlets (domain, name, social_handles) VALUES (%s, %s, %s) RETURNING id",
                (domain, name, social_handles_json)
            )
            outlet_id = cursor.fetchone()[0]
            conn.commit()
            return outlet_id
            
    except Exception as e:
        print(f"Error upserting media outlet: {e}")
        conn.rollback()
        return None
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
