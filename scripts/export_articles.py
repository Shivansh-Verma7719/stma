import sys
import os
import pandas as pd

# Add the pipelines directory to sys.path to allow importing helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
pipelines_dir = os.path.join(os.path.dirname(current_dir), 'pipelines')
sys.path.append(pipelines_dir)

from helpers.db_helper import get_articles_without_content

def main():
    print("Starting Article Export...")
    
    # Fetch 1000 articles
    articles = get_articles_without_content(limit=1000)
    
    if not articles:
        print("No articles found to export.")
        return
        
    print(f"Found {len(articles)} articles. Saving to CSV...")
    
    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'articles.csv')
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully saved articles to {output_path}")

if __name__ == "__main__":
    main()
