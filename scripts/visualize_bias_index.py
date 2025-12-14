import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add pipelines to path to import helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))
from helpers.db_helper import get_db_connection

def visualize_bias_index():
    conn = get_db_connection()
    try:
        print("Fetching bias data (Raw and Normalized)...")
        query = """
            SELECT ticker, date, bias_score as raw_score, norm_bias_score as norm_score 
            FROM bias_index 
            WHERE bias_score IS NOT NULL AND norm_bias_score IS NOT NULL 
            ORDER BY date
        """
        df = pd.read_sql(query, conn)
        
        if df.empty:
            print("No data found in bias_index table.")
            return

        print(f"Loaded {len(df)} rows. Analyzing...")
        
        # --- Analysis 1: Time Series Statistics (Normalized) ---
        mean_bias = df['norm_score'].mean()
        std_bias = df['norm_score'].std()
        upper_bound = mean_bias + 2 * std_bias
        lower_bound = mean_bias - 2 * std_bias
        
        outliers = df[(df['norm_score'] > upper_bound) | (df['norm_score'] < lower_bound)]
        outlier_pct = (len(outliers) / len(df)) * 100
        
        print("\n--- Time Series Stats (Normalized) ---")
        print(f"Mean: {mean_bias:.4f}")
        print(f"Std Dev: {std_bias:.4f}")
        print(f"Outliers (> 2 Std Dev): {len(outliers)} ({outlier_pct:.2f}%)")
        
        # --- Analysis 2: Saturation Statistics ---
        raw_abs = df['raw_score'].abs()
        norm_abs = df['norm_score'].abs()
        
        # Re-calc reference p_ref for context
        p_ref = np.percentile(raw_abs, 99.0)
        
        clamped = df[norm_abs >= 0.999999]
        clamped_pct = (len(clamped) / len(df)) * 100
        
        high_saturation = df[norm_abs >= 0.9]
        high_sat_pct = (len(high_saturation) / len(df)) * 100
        
        thresh_raw = 0.0
        if not high_saturation.empty:
            thresh_raw = df.loc[high_saturation.index, 'raw_score'].abs().min()
            
        print("\n--- Saturation Stats ---")
        print(f"Clamped (>= 1.0): {len(clamped)} ({clamped_pct:.3f}%)")
        print(f"Highly Saturated (>= 0.9): {len(high_saturation)} ({high_sat_pct:.3f}%)")
        print(f"Saturation Threshold (Raw score where Norm >= 0.9): ~{thresh_raw:.2f}")
        print(f"p_ref (99th percentile of raw): {p_ref:.2f}")

        # Ensure output directory exists
        os.makedirs('visualizations', exist_ok=True)

        # --- Plot 1: Normalized Bias Time Series ---
        plt.figure(figsize=(15, 8))
        
        # Plot individual companies
        unique_tickers = df['ticker'].unique()
        for ticker in unique_tickers:
            subset = df[df['ticker'] == ticker]
            plt.plot(subset['date'], subset['norm_score'], alpha=0.05, color='gray', linewidth=0.5)
            
        # Plot Mean and Bands (Norm)
        plt.axhline(y=mean_bias, color='red', linestyle='-', linewidth=2, label=f'Mean ({mean_bias:.2f})')
        plt.axhline(y=upper_bound, color='green', linestyle='--', linewidth=2, label=f'+2 Std Dev ({upper_bound:.2f})')
        plt.axhline(y=lower_bound, color='green', linestyle='--', linewidth=2, label=f'-2 Std Dev ({lower_bound:.2f})')
        
        # Plot Daily Average (Norm)
        daily_mean_norm = df.groupby('date')['norm_score'].mean()
        plt.plot(daily_mean_norm.index, daily_mean_norm.values, color='blue', linewidth=2, label='Daily Average')
        
        plt.title(f'Normalized Bias Index Over Time\n{outlier_pct:.2f}% Outliers (> 2 Std Dev)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Bias Score', fontsize=12)
        plt.legend(loc='upper right')
        plt.ylim(-1.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/bias_index_normalized.png', dpi=300)
        plt.close()
        print("Saved visualizations/bias_index_normalized.png")

        # --- Plot 2: Raw Bias Time Series (New) ---
        plt.figure(figsize=(15, 8))
        
        # Calculate stats for Raw just for the plot lines
        raw_mean = df['raw_score'].mean()
        raw_std = df['raw_score'].std()
        
        # Plot individual companies
        for ticker in unique_tickers:
            subset = df[df['ticker'] == ticker]
            plt.plot(subset['date'], subset['raw_score'], alpha=0.05, color='gray', linewidth=0.5)
            
        # Plot Mean and Bands (Raw)
        plt.axhline(y=raw_mean, color='red', linestyle='-', linewidth=2, label=f'Mean ({raw_mean:.2f})')
        plt.axhline(y=raw_mean + 2*raw_std, color='green', linestyle='--', linewidth=2, label=f'+2 Std ({raw_mean + 2*raw_std:.2f})')
        plt.axhline(y=raw_mean - 2*raw_std, color='green', linestyle='--', linewidth=2, label=f'-2 Std ({raw_mean - 2*raw_std:.2f})')

        # Plot Daily Average (Raw)
        daily_mean_raw = df.groupby('date')['raw_score'].mean()
        plt.plot(daily_mean_raw.index, daily_mean_raw.values, color='blue', linewidth=2, label='Daily Average')

        plt.title('Raw Bias Index Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Raw Bias Score', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/bias_index_raw.png', dpi=300)
        plt.close()
        print("Saved visualizations/bias_index_raw.png")

        # --- Plot 3: Saturation Scatter (Raw vs Norm) ---
        plt.figure(figsize=(10, 8))
        
        if len(df) > 50000:
            scatter_df = df.sample(50000)
        else:
            scatter_df = df
            
        plt.scatter(scatter_df['raw_score'], scatter_df['norm_score'], alpha=0.1, s=2, label='Data Points', color='purple')
        
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=-1.0, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=p_ref, color='green', linestyle=':', label=f'p_ref (99th): {p_ref:.1f}')
        plt.axvline(x=-p_ref, color='green', linestyle=':')
        
        plt.title(f"Saturation Analysis: Raw vs Norm\n{clamped_pct:.2f}% Clamped, {high_sat_pct:.2f}% > 0.9", fontsize=16)
        plt.xlabel("Raw Bias Score", fontsize=12)
        plt.ylabel("Normalized Bias Score (asinh)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/bias_saturation_analysis.png', dpi=300)
        plt.close()
        print("Saved visualizations/bias_saturation_analysis.png")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    visualize_bias_index()
