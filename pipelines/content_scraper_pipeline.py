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
from helpers.db_helper import get_articles_without_content, upsert_media_outlet, update_article_content

load_dotenv()

def extract_social_handles(html_content):
    handles = {}
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
                    if match.group(1) not in handles[platform]:
                        handles[platform].append(match.group(1))
    return handles

def get_domain(url):
    try:
        return urlparse(url).netloc.replace('www.', '')
    except:
        return None

def process_article(article_row):
    article_id = article_row['id']
    url = article_row['url']
    print(f"Processing Article ID {article_id}: {url}")
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        content = article.text
        html = article.html
        
        social_handles = extract_social_handles(html)
        domain = get_domain(url)
        media_name = article_row['source'] or article.source_url or domain
        
        # Upsert outlet with found handles
        outlet_id = upsert_media_outlet(domain, media_name, json.dumps(social_handles))
        
        update_article_content(article_id, content, outlet_id, json.dumps(social_handles))
        print(f"Updated article {article_id} and outlet {domain}")
        
    except Exception as e:
        print(f"Error processing article {article_id}: {e}")

def main():
    print("Starting Content Scraper Pipeline")
    
    articles = get_articles_without_content(limit=100)
    
    if not articles:
        print("No articles found needing content scraping.")
        return
        
    print(f"Found {len(articles)} articles to process.")
    
    for article in articles:
        process_article(article)
        time.sleep(1)

if __name__ == "__main__":
    main()
