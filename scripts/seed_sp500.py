#!/usr/bin/env python3
"""
Seed script to populate the companies table with SP500 companies.
Run this once initially and periodically to keep the company list updated.
"""

import sys
import os

# Add parent directory to path so we can import from helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

import pandas as pd
import requests
from io import StringIO
from helpers.db_helper import seed_sp500_companies
from dotenv import load_dotenv

load_dotenv()


def fetch_sp500_companies():
    """Fetch current SP500 company list from Slickcharts."""
    print("Fetching S&P 500 companies list...")
    url = "https://www.slickcharts.com/sp500"
    try:
        user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) "
            "Gecko/20100101 Firefox/111.0"
        )
        response = requests.get(url, headers={"User-Agent": user_agent})
        response.raise_for_status()
        sp500_df = pd.read_html(
            StringIO(response.text), match="Symbol", index_col="Symbol"
        )[0]
        # Reset index to get Symbol as a column
        sp500_df = sp500_df.reset_index()
        print(f"Found {len(sp500_df)} companies")
        return sp500_df
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return None


def main():
    print("=" * 60)
    print("SP500 Companies Seed Script")
    print("=" * 60)
    
    # Fetch companies
    companies_df = fetch_sp500_companies()
    if companies_df is None:
        print("Failed to fetch companies. Exiting.")
        return 1
    
    # Seed database
    print("\nSeeding database...")
    seed_sp500_companies(companies_df)
    
    print("\n✅ Seed completed successfully!")
    print("You can now run the sp500_media_pipeline.py to process companies.")
    return 0


if __name__ == "__main__":
    exit(main())
