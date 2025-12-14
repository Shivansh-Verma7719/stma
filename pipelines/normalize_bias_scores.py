import sys
import os
import numpy as np
import pandas as pd
import psycopg2.extras

# Add pipelines to path to import helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))
from helpers.db_helper import get_db_connection

def normalize_bias_scores():
    conn = get_db_connection()
    try:
        print("Fetching raw bias scores...")
        query = "SELECT id, bias_score FROM bias_index WHERE bias_score IS NOT NULL"
        df = pd.read_sql(query, conn)
        
        if df.empty:
            print("No data found.")
            return

        print(f"Loaded {len(df)} rows. Calculating normalized scores...")
        
        bias = df['bias_score'].values
        
        # Robust Scale choices
        # Median Absolute Deviation
        median = np.median(bias)
        mad = np.median(np.abs(bias - median))
        s = max(mad, 1e-6)
        
        # P_ref (99th percentile of absolute values)
        p_ref = np.percentile(np.abs(bias), 99.0)
        if p_ref <= 0:
            p_ref = np.max(np.abs(bias)) + 1e-6
            
        print(f"Normalization Params: s (MAD)={s:.4f}, p_ref={p_ref:.4f}")
        
        # Compute asinh normalized value
        den = np.arcsinh(p_ref / s)
        normed = np.arcsinh(bias / s) / den
        normed = np.clip(normed, -1.0, 1.0)
        
        df['norm_bias_score'] = normed
        
        print("Updating database...")
        update_query = """
            UPDATE bias_index
            SET norm_bias_score = %s
            WHERE id = %s
        """
        
        # Batch update
        data_to_update = list(zip(df['norm_bias_score'], df['id']))
        
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, update_query, data_to_update)
            conn.commit()
            
        print("Successfully updated normalized scores.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    normalize_bias_scores()
