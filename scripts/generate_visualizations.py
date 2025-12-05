#!/usr/bin/env python3
"""
Script to generate visualizations for Sp500 companies and their article coverage.
Generates:
1. Top companies by article count (Bar Chart)
2. Distribution of article counts (Histogram/KDE)
3. Market Cap vs Article Count (Scatter Plot)
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Add parent directory to path so we can import from helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from helpers.db_helper import get_db_connection

def ensure_viz_dir():
    viz_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    return viz_dir

def get_data():
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database.")
        sys.exit(1)
        
    try:
        query = """
        SELECT 
            c.symbol, 
            c.name, 
            COUNT(a.id) as article_count
        FROM companies c
        LEFT JOIN articles a ON c.id = a.company_id
        GROUP BY c.id, c.symbol, c.name
        ORDER BY article_count DESC
        """
        df_db = pd.read_sql(query, conn)
        return df_db
    finally:
        conn.close()

def load_market_cap_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'sp500_top250.csv')
    try:
        df = pd.read_csv(csv_path)
        # Ensure consistent column naming
        return df[['Symbol', 'Market Cap']]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()

def generate_visualizations():
    viz_dir = ensure_viz_dir()
    
    print("Fetching data from database...")
    df_articles = get_data()
    
    print("Loading market cap data...")
    df_mcap = load_market_cap_data()
    
    # Merge datasets
    if not df_mcap.empty:
        df = pd.merge(df_articles, df_mcap, left_on='symbol', right_on='Symbol', how='left')
    else:
        df = df_articles

    # Set style
    sns.set_theme(style="whitegrid")
    
    # --- Visualization 1: Top 20 Companies by Article Count ---
    plt.figure(figsize=(12, 8))
    top_20 = df.head(20)
    sns.barplot(x='article_count', y='symbol', data=top_20, palette='viridis', hue='symbol', legend=False)
    plt.title('Top 20 S&P 500 Companies by Article Count', fontsize=16)
    plt.xlabel('Number of Articles', fontsize=12)
    plt.ylabel('Company Symbol', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'top_20_articles.png'))
    plt.close()
    print("Generated top_20_articles.png")

    # --- Visualization 2: Distribution of Article Counts ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df['article_count'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Article Counts per Company', fontsize=16)
    plt.xlabel('Number of Articles', fontsize=12)
    plt.ylabel('Frequency (Number of Companies)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'article_count_distribution.png'))
    plt.close()
    print("Generated article_count_distribution.png")

    # --- Visualization 3: Market Cap vs Article Count ---
    if 'Market Cap' in df.columns and not df['Market Cap'].isnull().all():
        plt.figure(figsize=(10, 8))
        # Convert Market Cap to Billions for readability if it's large numbers
        # Assuming Market Cap is in raw units? The CSV view showed e.g. 4399934120330.811
        # That looks like 4.3 Trillion. So it's raw bytes/dollars.
        df['Market Cap ($B)'] = df['Market Cap'] / 1e9
        
        sns.scatterplot(x='Market Cap ($B)', y='article_count', data=df, alpha=0.7, size='article_count', sizes=(20, 200), color='coral')
        
        # Log scale might be better if there are huge outliers, but let's try linear first or log-log
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Market Cap ($B) vs Article Count (Log-Log Scale)', fontsize=16)
        plt.xlabel('Market Cap ($ Billions)', fontsize=12)
        plt.ylabel('Article Count', fontsize=12)
        
        # Annotate top outliers
        for i, row in df.head(10).iterrows():
            plt.text(row['Market Cap ($B)'], row['article_count'], row['symbol'], fontsize=9, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'market_cap_vs_articles.png'))
        plt.close()
        print("Generated market_cap_vs_articles.png")

    print(f"\nAll visualizations saved to: {viz_dir}")

if __name__ == "__main__":
    generate_visualizations()
