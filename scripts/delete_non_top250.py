#!/usr/bin/env python3
"""
Script to delete companies from the database that are NOT in the S&P 500 Top 250 CSV.
Also deletes related articles for those companies.
"""

import sys
import os
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path so we can import from helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from helpers.db_helper import get_db_connection

def delete_non_top250_companies():
    # 1. Load the Top 250 CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'sp500_top250.csv')
    try:
        df = pd.read_csv(csv_path)
        # Assuming the column name is 'Symbol' based on previous file view
        allowed_symbols = set(df['Symbol'].unique())
        print(f"Loaded {len(allowed_symbols)} allowed symbols from {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file at {csv_path}: {e}")
        return

    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database.")
        return

    try:
        cursor = conn.cursor()
        
        # 2. Get all companies currently in the DB
        cursor.execute("SELECT id, symbol, name FROM companies")
        all_companies = cursor.fetchall()
        print(f"Found {len(all_companies)} companies in the database.")
        
        companies_to_delete = []
        company_ids_to_delete = []
        
        for company_id, symbol, name in all_companies:
            if symbol not in allowed_symbols:
                companies_to_delete.append(symbol)
                company_ids_to_delete.append(company_id)
        
        print(f"Identified {len(companies_to_delete)} companies to delete.")
        
        if not companies_to_delete:
            print("No companies to delete. Database is already clean.")
            return

        # 3. Delete related articles first
        if company_ids_to_delete:
            # Format the ids for SQL IN clause
            ids_tuple = tuple(company_ids_to_delete)
            
            print(f"Deleting articles relating to {len(company_ids_to_delete)} companies...")
            cursor.execute(
                "DELETE FROM articles WHERE company_id IN %s",
                (ids_tuple,)
            )
            print(f"Deleted {cursor.rowcount} articles.")

            # 4. Delete the companies
            print(f"Deleting {len(company_ids_to_delete)} companies...")
            cursor.execute(
                "DELETE FROM companies WHERE id IN %s",
                (ids_tuple,)
            )
            print(f"Deleted {cursor.rowcount} companies.")
        
        conn.commit()
        print("Successfully cleaned up the database.")
        
    except Exception as e:
        print(f"An error occurred during database operations: {e}")
        conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    delete_non_top250_companies()
