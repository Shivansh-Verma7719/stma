import psycopg2
import os
from dotenv import load_dotenv
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))
from helpers.db_helper import get_db_connection


# Load environment variables
load_dotenv()

def copy_stock_to_bias():
    conn = None
    try:
        print("Connecting to database...")
        conn = get_db_connection()
        cur = conn.cursor()

        print("Copying data from stock_prices to bias_index...")
        query = """
        INSERT INTO bias_index (ticker, date, open, high, low, close, volume)
        SELECT ticker, date, open, high, low, close, volume
        FROM stock_prices;
        """
        
        cur.execute(query)
        rows_affected = cur.rowcount
        conn.commit()
        
        print(f"Successfully copied {rows_affected} rows.")

    except Exception as e:
        print(f"Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    copy_stock_to_bias()
