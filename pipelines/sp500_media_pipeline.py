import pandas as pd
import requests
import time
import datetime as dt
import os
import sys
from typing import List, Dict
import mediacloud.api
from dotenv import load_dotenv
from helpers.db_helper import insert_articles

load_dotenv()

MEDIA_CLOUD_API_KEY = os.getenv("MEDIA_CLOUD_API_KEY_1")
START_DATE = dt.date(2015, 1, 1)
END_DATE = dt.date(2025, 11, 12)
# ID for US National Collection in Media Cloud
US_NATIONAL_COLLECTION = 34412234

def get_sp500_companies() -> pd.DataFrame:
    print("Fetching S&P 500 companies list...")
    url = 'https://www.slickcharts.com/sp500'
    try:
        # User-Agent is required to avoid 403 Forbidden from slickcharts
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0'
        response = requests.get(url, headers={'User-Agent': user_agent})
        sp500_df = pd.read_html(response.text, match='Symbol', index_col='Symbol')[0]
        return sp500_df.reset_index()
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return None

def create_search_query(company_name: str, stock_symbol: str) -> str:
    # Remove corporate suffixes to improve search relevance
    clean_name = company_name.replace(' Inc.', '').replace(' Corp.', '').replace(' Corporation', '')
    clean_name = clean_name.replace(' Ltd.', '').replace(' LLC', '').replace(',', '')
    return f'"{clean_name}" OR {stock_symbol}'

def query_media_cloud(search_api: mediacloud.api.SearchApi, query: str) -> List[Dict]:
    all_stories = []
    pagination_token = None
    more_stories = True
    page_num = 1
    
    try:
        while more_stories:
            print(f"Fetching page {page_num}...", end=" ")
            # Fetch stories with pagination
            page, pagination_token = search_api.story_list(
                query, 
                start_date=START_DATE,
                end_date=END_DATE,
                collection_ids=[US_NATIONAL_COLLECTION],
                pagination_token=pagination_token
            )
            
            print(f"Got {len(page)} articles")
            all_stories.extend(page)
            more_stories = pagination_token is not None
            page_num += 1
            
            # Media Cloud API has strict rate limits (often 2 req/min)
            if more_stories:
                time.sleep(30) 
                
    except Exception as e:
        print(f"Error querying Media Cloud: {e}")
    
    return all_stories

def process_company(search_api: mediacloud.api.SearchApi, company_name: str, stock_symbol: str) -> List[Dict]:
    print(f"\nProcessing: {company_name} ({stock_symbol})")
    query = create_search_query(company_name, stock_symbol)
    articles = query_media_cloud(search_api, query)
    
    return [{
        'company_name': company_name,
        'stock_symbol': stock_symbol,
        'article_id': a.get('id', ''),
        'title': a.get('title', ''),
        'url': a.get('url', ''),
        'publish_date': a.get('publish_date', ''),
        'media_name': a.get('media_name', ''),
        'language': a.get('language', ''),
    } for a in articles]

def main():
    print("Starting S&P 500 Media Cloud Article Pipeline")
    
    if not MEDIA_CLOUD_API_KEY:
        print("Error: MEDIA_CLOUD_API_KEY not found.")
        return

    search_api = mediacloud.api.SearchApi(MEDIA_CLOUD_API_KEY)
    sp500_df = get_sp500_companies()
    
    if sp500_df is None:
        return
    
    total = len(sp500_df)
    print(f"Found {total} companies.")
    
    for idx, row in sp500_df.iterrows():
        company = row['Company']
        symbol = row['Symbol']
        
        print(f"[{idx + 1}/{total}] ", end="")
        articles = process_company(search_api, company, symbol)
        
        if articles:
            print(f"Pushing {len(articles)} articles to database...")
            # Inserts articles and upserts linked media outlets
            # We commit to DB after each company to avoid bulk data loss
            insert_articles(articles)
        
        # Enforce rate limit between company queries
        time.sleep(30)

if __name__ == "__main__":
    main()
