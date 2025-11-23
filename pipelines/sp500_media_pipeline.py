import pandas as pd
import requests
import time
import datetime as dt
import os
from typing import List, Dict
import mediacloud.api

# Media Cloud API Configuration
MEDIA_CLOUD_API_KEY = "776c2fafbba1b9f24693f56d7f8e8d3a42f39f33"

# Date range
START_DATE = dt.date(2015, 1, 1)
END_DATE = dt.date(2025, 11, 12)

# US National Collection for broader coverage
US_NATIONAL_COLLECTION = 34412234


def get_sp500_companies() -> pd.DataFrame:
    print("Fetching S&P 500 companies list...")
    
    url = 'https://www.slickcharts.com/sp500'
    
    try:
        # Use proper user agent to avoid blocking
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0'
        response = requests.get(url, headers={'User-Agent': user_agent})
        
        # Parse the table with 'Symbol' column
        sp500_df = pd.read_html(response.text, match='Symbol', index_col='Symbol')[0]
        
        # Reset index to make Symbol a regular column
        sp500_df = sp500_df.reset_index()
        
        print(f"Successfully fetched {len(sp500_df)} companies")
        return sp500_df
    
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return None


def create_search_query(company_name: str, stock_symbol: str) -> str:
    """
    Create a search query for Media Cloud API
    
    Args:
        company_name: Company name
        stock_symbol: Stock ticker symbol
    
    Returns:
        Search query string
    """
    # Clean company name (remove common suffixes)
    clean_name = company_name.replace(' Inc.', '').replace(' Corp.', '').replace(' Corporation', '')
    clean_name = clean_name.replace(' Ltd.', '').replace(' LLC', '').replace(',', '')
    
    # Create query with both name and symbol
    query = f'"{clean_name}" OR {stock_symbol}'
    
    return query


def query_media_cloud(search_api: mediacloud.api.SearchApi, query: str, 
                      start_date: dt.date, end_date: dt.date) -> List[Dict]:
    """
    Query Media Cloud API using the official client for articles
    
    Args:
        search_api: MediaCloud SearchApi instance
        query: Search query string
        start_date: Start date (datetime.date object)
        end_date: End date (datetime.date object)
    
    Returns:
        List of article dictionaries
    """
    all_stories = []
    pagination_token = None
    more_stories = True
    max_retries = 3
    retry_count = 0
    
    page_num = 1
    try:
        while more_stories:
            try:
                # Fetch page of stories using official client
                if page_num == 1:
                    print(f"  Fetching page {page_num}...", end=" ")
                else:
                    print(f"  📄 Fetching page {page_num}... (Total so far: {len(all_stories)} articles)", end=" ")
                
                page, pagination_token = search_api.story_list(
                    query, 
                    start_date=start_date,
                    end_date=end_date,
                    collection_ids=[US_NATIONAL_COLLECTION],
                    pagination_token=pagination_token
                )
                
                print(f"✓ Got {len(page)} articles")
                all_stories += page
                more_stories = pagination_token is not None
                retry_count = 0  # Reset retry count on success
                page_num += 1
                
                # Small delay between pages to respect rate limit (2 req/min = 30 sec between)
                if more_stories:
                    print(f"  ⏳ Waiting 30 seconds before next page (rate limit: 2 requests/minute)...")
                    time.sleep(30)  # 30 seconds to stay under 2 requests per minute
                    
            except Exception as page_error:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"  ERROR: {type(page_error).__name__}: {page_error}")
                    print(f"  This may be due to API rate limiting or quota exceeded")
                    raise page_error
                print(f"  Retry {retry_count}/{max_retries} after error: {page_error}")
                time.sleep(5)  # Wait longer before retry
        
        print(f"  Found {len(all_stories)} articles")
    
    except Exception as e:
        print(f"  Error querying Media Cloud: {e}")
    
    return all_stories


def process_company(search_api: mediacloud.api.SearchApi, company_name: str, 
                   stock_symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Process a single company: query articles and return as DataFrame
    
    Args:
        search_api: MediaCloud SearchApi instance
        company_name: Company name
        stock_symbol: Stock ticker symbol
        start_date: Start date for search
        end_date: End date for search
    
    Returns:
        DataFrame with article information
    """
    print(f"\nProcessing: {company_name} ({stock_symbol})")
    
    query = create_search_query(company_name, stock_symbol)
    print(f"  Query: {query}")
    
    articles = query_media_cloud(search_api, query, start_date, end_date)
    
    if not articles:
        return pd.DataFrame()
    
    # Extract relevant fields
    processed_articles = []
    for article in articles:
        processed_articles.append({
            'company_name': company_name,
            'stock_symbol': stock_symbol,
            'article_id': article.get('id', ''),
            'title': article.get('title', ''),
            'url': article.get('url', ''),
            'publish_date': article.get('publish_date', ''),
            'media_name': article.get('media_name', ''),
            'language': article.get('language', ''),
        })
    
    return pd.DataFrame(processed_articles)


def save_results(df: pd.DataFrame, output_dir: str = 'output'):
    """
    Save results to a single mega CSV file
    
    Args:
        df: DataFrame with all results
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save complete results to one mega CSV
    output_file = f"{output_dir}/sp500_articles_mega_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved all results to: {output_file}")
    print(f"Total articles: {len(df)}")
    print(f"Total companies with articles: {df['company_name'].nunique()}")
    
    return output_file


def main():
    """
    Main pipeline execution
    """
    print("=" * 60)
    print("S&P 500 Media Cloud Article Pipeline")
    print("=" * 60)
    print(f"Date range: {START_DATE} to {END_DATE}")
    print()
    
    # Initialize Media Cloud API client
    print("Initializing Media Cloud API client...")
    search_api = mediacloud.api.SearchApi(MEDIA_CLOUD_API_KEY)
    print("API client ready!\n")
    
    # Step 1: Get S&P 500 companies
    sp500_df = get_sp500_companies()
    if sp500_df is None:
        print("Failed to fetch S&P 500 companies. Exiting.")
        return
    
    # Step 2: Query Media Cloud for each company
    all_articles = []
    companies_processed = 0
    total_companies = len(sp500_df)
    
    print(f"\n⏱️  Estimated time: ~{total_companies * 0.5:.0f} minutes ({total_companies * 0.5 / 60:.1f} hours)")
    print(f"   (30 seconds per company x {total_companies} companies)\n")
    
    for idx, row in sp500_df.iterrows():
        company_name = row['Company']  # Column name from SlickCharts
        stock_symbol = row['Symbol']
        
        # Query for this company
        print(f"[{companies_processed + 1}/{total_companies}] ", end="")
        articles_df = process_company(search_api, company_name, stock_symbol, START_DATE, END_DATE)
        
        if not articles_df.empty:
            all_articles.append(articles_df)
        
        companies_processed += 1
        
        # Save progress every 50 companies
        if companies_processed % 50 == 0 and all_articles:
            combined_df = pd.concat(all_articles, ignore_index=True)
            from datetime import datetime
            checkpoint_file = f"output/checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs('output', exist_ok=True)
            combined_df.to_csv(checkpoint_file, index=False)
            print(f"\n  💾 Checkpoint saved: {checkpoint_file} ({len(combined_df)} articles so far)\n")
        
        # Rate limiting: wait between requests to respect API limit of 2 requests per minute
        time.sleep(30)  # 30 seconds between companies (2 requests per minute max)
        
        # Optional: Process only first N companies for testing
        # if idx >= 10:  # Uncomment to test with first 10 companies
        #     break
    
    # Step 3: Combine and save results to ONE MEGA CSV
    if all_articles:
        combined_df = pd.concat(all_articles, ignore_index=True)
        print(f"\n{'=' * 60}")
        print(f"Total articles found: {len(combined_df)}")
        print(f"Companies with articles: {combined_df['company_name'].nunique()}")
        print(f"{'=' * 60}\n")
        
        save_results(combined_df)
    else:
        print("\nNo articles found for any company.")
    
    print("\nPipeline completed!")


if __name__ == "__main__":
    main()
