from transformers import pipeline
import psycopg2
from psycopg2.extras import DictCursor
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize BERT sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_db_connection():
    """Establish connection to PostgreSQL database"""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
    return conn

def extract_sentiment(text, max_length=512):
    """
    Extract sentiment from text using BERT
    Returns dict with 'pos' and 'neg' probabilities
    """
    if not text or len(text.strip()) == 0:
        return {"pos": 0.0, "neg": 0.0}
    
    # Truncate text to max length
    text = text[:max_length]
    
    result = sentiment_pipeline(text)[0]
    
    # Map POSITIVE/NEGATIVE to pos/neg scores
    if result['label'] == 'POSITIVE':
        return {"pos": round(result['score'], 4), "neg": round(1 - result['score'], 4)}
    else:
        return {"pos": round(1 - result['score'], 4), "neg": round(result['score'], 4)}

def process_articles():
    """Fetch articles and compute sentiment scores"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        # Fetch all articles
        cur.execute("SELECT id, title, content FROM articles")
        articles = cur.fetchall()
        
        for article in articles:
            # Combine title and content
            text = f"{article['title']} {article['content']}" if article['content'] else article['title']
            
            # Get sentiment scores
            sentiment = extract_sentiment(text)
            
            # Update article with sentiment data
            cur.execute(
                "UPDATE articles SET social_data = %s WHERE id = %s",
                (json.dumps(sentiment), article['id'])
            )
        
        conn.commit()
        print(f"Processed {len(articles)} articles")
        
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    process_articles()