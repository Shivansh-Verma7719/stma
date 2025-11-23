import pandas as pd
import requests
import time
import os
import sys
import json
import re
from urllib.parse import urlparse
from newspaper import Article
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Add project root to path to import helpers
from helpers.db_helper import get_db_connection, get_articles_without_content, upsert_media_outlet, update_article_content

load_dotenv()

def extract_social_handles(html_content):
    """
    Extract social media handles from HTML content using regex and BeautifulSoup.
    Returns a dictionary of handles.
    """
    handles = {}
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Common social media patterns
    patterns = {
        'twitter': r'twitter\.com/([a-zA-Z0-9_]+)',
        'facebook': r'facebook\.com/([a-zA-Z0-9\.]+)',
        'linkedin': r'linkedin\.com/company/([a-zA-Z0-9-]+)',
        'instagram': r'instagram\.com/([a-zA-Z0-9_\.]+)',
        'youtube': r'youtube\.com/(?:user/|c/|channel/|@)([a-zA-Z0-9_-]+)'
    }
    
    # Check links
    for a in soup.find_all('a', href=True):
        href = a['href']
        for platform, pattern in patterns.items():
            match = re.search(pattern, href, re.IGNORECASE)
            if match:
                # Avoid sharing buttons usually (simple heuristic)
                if 'share' not in href and 'intent' not in href:
                    if platform not in handles:
                        handles[platform] = []
                    if match.group(1) not in handles[platform]:
                        handles[platform].append(match.group(1))
    
    return handles

def get_domain(url):
    try:
        return urlparse(url).netloc.replace('www.', '')
    except:
        return None

def process_article(article_row):
    """
    Process a single article: scrape content, extract metadata, update DB.
    """
    article_id = article_row['id']
    url = article_row['url']
    print(f"Processing Article ID {article_id}: {url}")
    
    try:
        # Use newspaper3k for article extraction
        article = Article(url)
        article.download()
        article.parse()
        
        content = article.text
        html = article.html
        
        # Extract social handles from the page (could be outlet's or author's)
        social_handles = extract_social_handles(html)
        
        # Extract/Determine Media Outlet
        domain = get_domain(url)
        media_name = article_row['source'] or article.source_url or domain
        
        # Upsert Media Outlet
        # We assume the social handles found might belong to the outlet if they are in the footer/header
        # This is a simplification.
        outlet_id = upsert_media_outlet(domain, media_name, json.dumps(social_handles))
        
        # Update Article
        update_article_content(article_id, content, outlet_id, json.dumps(social_handles))
        print(f"  ✓ Updated article {article_id} and outlet {domain}")
        
    except Exception as e:
        print(f"  Error processing article {article_id}: {e}")

def main():
    print("=" * 60)
    print("Content Scraper Pipeline")
    print("=" * 60)
    
    # Fetch articles without content
    articles = get_articles_without_content(limit=100) # Process in batches
    
    if not articles:
        print("No articles found needing content scraping.")
        return
        
    print(f"Found {len(articles)} articles to process.")
    
    for article in articles:
        process_article(article)
        time.sleep(1) # Be polite
        
    print("\nBatch completed!")

if __name__ == "__main__":
    main()
