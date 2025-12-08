import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root and pipelines dir to path to find helpers
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
pipelines_dir = k_root = project_root / 'pipelines'
sys.path.append(str(pipelines_dir))

try:
    from helpers.db_helper import get_db_connection
except ImportError:
    # Fallback if specific path setup is needed
    sys.path.append(str(project_root))
    from pipelines.helpers.db_helper import get_db_connection

def analyze_duplicates():
    print("Connecting to database...")
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to DB.")
        return

    try:
        print("Fetching title counts...")
        query = """
            SELECT title, COUNT(*) as count 
            FROM articles 
            WHERE title IS NOT NULL 
            GROUP BY title
        """
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("No articles found.")
            return

        total_titles = df['count'].sum()
        unique_titles = len(df)
        duplicates_mask = df['count'] > 1
        duplicate_titles = df[duplicates_mask]
        
        print("\n=== Duplicate Analysis ===")
        print(f"Total Articles: {total_titles:,}")
        print(f"Unique Titles:  {unique_titles:,}")
        print(f"Duplicate Titles: {len(duplicate_titles):,} ({(len(duplicate_titles)/unique_titles)*100:.2f}%)")
        print(f"Total Redundant Entries: {total_titles - unique_titles:,}")
        
        if not duplicate_titles.empty:
            print("\nTop 10 Most Duplicated Titles:")
            print(duplicate_titles.sort_values('count', ascending=False).head(10).to_string(index=False))

            # --- Backup and Clean Logic ---
            print("\nPreparing to clean duplicates...")
            
            # 1. Identify IDs to delete
            # We want to keep one instance (e.g., min_id) for each duplicate title
            # This query identifies all IDs that are NOT the minimum ID for their title group
            # (i.e., the redundant ones)
            print("Fetching redundant IDs...")
            dup_query = """
                WITH RankedArticles AS (
                    SELECT 
                        id, 
                        title, 
                        url,
                        published_at,
                        ROW_NUMBER() OVER(PARTITION BY title ORDER BY id ASC) as rn
                    FROM articles
                    WHERE title IS NOT NULL
                )
                SELECT id, title, url, published_at
                FROM RankedArticles
                WHERE rn > 1
            """
            
            duplicates_df = pd.read_sql_query(dup_query, conn)
            
            if duplicates_df.empty:
                print("No actual redundant rows found for deletion.")
                return

            print(f"Found {len(duplicates_df):,} redundant rows to delete.")
            
            # 2. CSV Backup
            backup_dir = os.path.join(project_root, 'data', 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            import datetime as dt
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(backup_dir, f'duplicates_backup_{timestamp}.csv')
            
            print(f"Backing up redundant entries to: {backup_file}")
            duplicates_df.to_csv(backup_file, index=False)
            print("Backup complete.")
            
            # 3. Deletion
            if len(duplicates_df) > 0:
                confirm = input(f"\n[WARNING] Are you sure you want to DELETE {len(duplicates_df):,} rows from the database? (yes/no): ")
                if confirm.lower() == 'yes':
                    print("Deleting...")
                    ids_to_delete = tuple(duplicates_df['id'].tolist())
                    
                    cursor = conn.cursor()
                    # Delete in chunks to avoid blowing up transaction log if massive
                    chunk_size = 1000
                    total_deleted = 0
                    
                    for i in range(0, len(ids_to_delete), chunk_size):
                        chunk = ids_to_delete[i:i + chunk_size]
                        if len(chunk) == 1:
                             # tuple of one element needs trailing comma for SQL IN clause syntax or just =
                             cursor.execute("DELETE FROM articles WHERE id = %s", (chunk[0],))
                        else:
                             cursor.execute("DELETE FROM articles WHERE id IN %s", (chunk,))
                        conn.commit()
                        total_deleted += len(chunk)
                        print(f"Deleted {total_deleted}/{len(ids_to_delete)}...")
                    
                    print(f"Successfully deleted {total_deleted} duplicate entries.")
                else:
                    print("Deletion cancelled.")
        else:
            print("\nNo duplicates found! Great job.")

    except Exception as e:
        print(f"Error analyzing duplicates: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_duplicates()
