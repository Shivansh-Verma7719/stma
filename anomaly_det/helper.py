import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables (Same as your previous config)
load_dotenv()

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(
    BASE_DIR, "fetch"
)  # The directory where you want the files
TABLE_NAME = "bias_index"


def fetch_data_from_db():
    """
    Fetches data from database and returns a dictionary of DataFrames keyed by ticker.
    Returns: dict[str, pd.DataFrame] - Dictionary mapping ticker to DataFrame
    """
    # Connect to DB
    print("Connecting to database...")
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432"),
    )

    try:
        # Fetch the data
        print(f"Fetching data from table '{TABLE_NAME}'...")
        query = f'SELECT * FROM "{TABLE_NAME}"'
        df = pd.read_sql(query, conn)

        if df.empty:
            print("Table is empty.")
            return {}

        print(f"Fetched {len(df)} rows. Splitting by ticker...")

        # Group by Ticker
        unique_tickers = df["ticker"].unique()
        print(f"Found {len(unique_tickers)} unique tickers.")

        result = {}
        for ticker in unique_tickers:
            # Filter data for this specific ticker
            ticker_df = df[df["ticker"] == ticker].copy()
            result[ticker] = ticker_df

        return result

    except Exception as e:
        print(f"Error: {e}")
        return {}

    finally:
        conn.close()


def export_tickers_to_csv():
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created directory: {OUTPUT_DIRECTORY}")

    # 2. Connect to DB
    print("Connecting to database...")
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432"),
    )

    try:
        # 3. Fetch the data
        # We use pandas read_sql to pull everything into a DataFrame at once.
        # This is usually faster than 250 separate queries.
        print(f"Fetching data from table '{TABLE_NAME}'...")

        # We wrap "final" in quotes because it can sometimes be a reserved keyword
        query = f'SELECT * FROM "{TABLE_NAME}"'

        df = pd.read_sql(query, conn)

        if df.empty:
            print("Table is empty.")
            return

        print(f"Fetched {len(df)} rows. Splitting by ticker...")

        # 4. Group by Ticker and Save
        # This assumes your column name in the DB is literally 'ticker'
        # if it is 'symbol' or 'instrument', change 'ticker' below to that name.
        unique_tickers = df["ticker"].unique()
        print(f"Found {len(unique_tickers)} unique tickers.")

        for ticker in unique_tickers:
            # Filter data for this specific ticker
            ticker_df = df[df["ticker"] == ticker]

            # Sanitize ticker for filename (e.g. remove / or \ to prevent path errors)
            safe_filename = str(ticker).replace("/", "_").replace("\\", "_")
            file_path = os.path.join(OUTPUT_DIRECTORY, f"{safe_filename}.csv")

            # Write to CSV
            # index=False prevents pandas from adding a 0,1,2... row number column
            ticker_df.to_csv(file_path, index=False)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        conn.close()
        print("Done.")


if __name__ == "__main__":
    export_tickers_to_csv()
