import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats

# Add pipelines to path to import helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))
from helpers.db_helper import get_db_connection

# CONFIGURATION
TOP_N_COMPANIES = None

def analyze_impact_rigorous():
    conn = get_db_connection()
    try:
        print("Fetching Companies by Article Count...")
        # 1. Get Tickers (Ordered by volume)
        top_q = "SELECT symbol FROM companies ORDER BY current_page DESC"
        with conn.cursor() as cur:
            cur.execute(top_q)
            top_tickers = [r[0] for r in cur.fetchall()]
        
        if TOP_N_COMPANIES:
            top_tickers = top_tickers[:TOP_N_COMPANIES]
            print(f"Limiting to Top {TOP_N_COMPANIES} companies.")
            
        print(f"Selected {len(top_tickers)} companies (e.g., {top_tickers[:5]})...")
        
        print("Fetching Price and Bias Data for Statistical Analysis...")
        # Fetching necessary columns for these tickers
        t_tuple = tuple(top_tickers)
        query = f"""
            SELECT ticker, date, close, norm_bias_score 
            FROM bias_index 
            WHERE close IS NOT NULL 
            AND norm_bias_score IS NOT NULL
            AND ticker IN %s
            ORDER BY ticker, date
        """
        df = pd.read_sql(query, conn, params=(t_tuple,))
        
        if df.empty:
            print("No data found.")
            return

        print(f"Loaded {len(df)} rows. Preprocessing...")
        
        # 1. Calculate Next Day Returns
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(['ticker', 'date'], inplace=True)
        
        # Calculate daily return
        df['ret'] = df.groupby('ticker')['close'].pct_change()
        
        # Shift return to get Next Day Return (t+1)
        # We want to predict Ret(t+1) using Bias(t)
        df['ret_next'] = df.groupby('ticker')['ret'].shift(-1)
        
        # Calculate Cumulative Returns (5-day, 14-day, 30-day)
        indexer_5 = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
        df['ret_5d'] = df.groupby('ticker')['ret'].shift(-1).rolling(window=indexer_5).sum()

        indexer_14 = pd.api.indexers.FixedForwardWindowIndexer(window_size=14)
        df['ret_14d'] = df.groupby('ticker')['ret'].shift(-1).rolling(window=indexer_14).sum()

        indexer_30 = pd.api.indexers.FixedForwardWindowIndexer(window_size=30)
        df['ret_30d'] = df.groupby('ticker')['ret'].shift(-1).rolling(window=indexer_30).sum()
        
        # Drop NaNs created by shift/pct_change
        data = df.dropna(subset=['ret_next', 'ret_5d', 'ret_14d', 'ret_30d', 'norm_bias_score'])
        
        # Filter out 0 bias (or very close to 0) to avoid hugging x-axis
        print(f"Rows before 0-filter: {len(data)}")
        data_filtered = data[data['norm_bias_score'].abs() > 0.001].copy()
        print(f"Rows after 0-filter: {len(data_filtered)}")
        
        # Split into Positive and Negative Bias regimes
        pos_data = data_filtered[data_filtered['norm_bias_score'] > 0].copy()
        neg_data = data_filtered[data_filtered['norm_bias_score'] < 0].copy()
        
        print(f"Positive Bias Samples: {len(pos_data)}")
        print(f"Negative Bias Samples: {len(neg_data)}")
        
        # Winsorize Returns (Remove extreme outliers that might skew OLS)
        def winsorize(d):
            lower = d['ret_next'].quantile(0.01)
            upper = d['ret_next'].quantile(0.99)
            return d[(d['ret_next'] >= lower) & (d['ret_next'] <= upper)].copy()
            
        pos_data = winsorize(pos_data)
        neg_data = winsorize(neg_data)
        
        os.makedirs('visualizations/impact_analysis', exist_ok=True)
        
        # --- Study 0: Global Correlation Matrix (Pos vs Neg) ---
        print("\n--- 0. Correlation Analysis ---")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, (d, name) in enumerate([(data_filtered, "Overall"), (pos_data, "Positive Bias"), (neg_data, "Negative Bias")]):
            corr_cols = ['norm_bias_score', 'ret_next', 'ret_5d', 'ret_14d', 'ret_30d']
            corr_matrix = d[corr_cols].corr(method='spearman')
            print(f"{name} Spearman Corr:\n", corr_matrix)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-0.1, vmax=0.1, fmt='.4f', ax=axes[i])
            axes[i].set_title(f"{name} Correlations")
            
        plt.tight_layout()
        plt.savefig('visualizations/impact_analysis/impact_correlation_matrix.png', dpi=300)
        print("Saved visualizations/impact_analysis/impact_correlation_matrix.png")
        
        # --- Study 1: Information Coefficient (IC) Analysis (Signal Quality) ---
        print("\n--- 1. Information Coefficient (IC) Analysis ---")
        
        # Daily IC
        ic_series = data_filtered.groupby('date').apply(lambda x: x['norm_bias_score'].corr(x['ret_next'], method='spearman'))
        # 5-Day IC
        ic_series_5d = data_filtered.groupby('date').apply(lambda x: x['norm_bias_score'].corr(x['ret_5d'], method='spearman'))
        
        print(f"1-Day IC: Mean={ic_series.mean():.4f}, IR={ic_series.mean()/ic_series.std():.4f}")
        print(f"5-Day IC: Mean={ic_series_5d.mean():.4f}, IR={ic_series_5d.mean()/ic_series_5d.std():.4f}")
        
        # Plot Cumulative IC
        plt.figure(figsize=(12, 6))
        ic_series.cumsum().plot(color='blue', linewidth=2, label='1-Day Forward')
        ic_series_5d.cumsum().plot(color='orange', linewidth=2, linestyle='--', label='5-Day Forward')
        plt.title(f"Cumulative Information Coefficient (Signal Quality)", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Rank Correlation")
        plt.axhline(0, color='black', linestyle='--')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/impact_analysis/impact_cumulative_ic.png', dpi=300)
        print("Saved visualizations/impact_analysis/impact_cumulative_ic.png")
        
        # --- Study 2: Panel Regression Splitted ---
        print("\n--- 2. Panel Regression (Split Pos/Neg) ---")
        
        results_txt = "REGRESSION RESULTS\n\n"
        results_txt += "| Regime | Forecast Horizon | Beta | P-Value | Significance |\n"
        results_txt += "| :--- | :--- | :--- | :--- | :--- |\n"
        
        models_data = []
        
        for d, label in [(data_filtered, "OVERALL"), (pos_data, "POSITIVE (>0)"), (neg_data, "NEGATIVE (<0)")]:
            # Demean Data for FE
            d['bias_demean'] = d.groupby('ticker')['norm_bias_score'].transform(lambda x: x - x.mean())
            d['ret_next_demean'] = d.groupby('ticker')['ret_next'].transform(lambda x: x - x.mean())
            d['ret_5d_demean'] = d.groupby('ticker')['ret_5d'].transform(lambda x: x - x.mean())
            d['ret_14d_demean'] = d.groupby('ticker')['ret_14d'].transform(lambda x: x - x.mean())
            d['ret_30d_demean'] = d.groupby('ticker')['ret_30d'].transform(lambda x: x - x.mean())
            
            # Models
            X = sm.add_constant(d['bias_demean'])
            
            # 1-Day
            model_1d = sm.OLS(d['ret_next_demean'], X).fit(cov_type='cluster', cov_kwds={'groups': d['ticker']})
            beta_1d, p_1d = model_1d.params['bias_demean'], model_1d.pvalues['bias_demean']
            
            # 5-Day
            model_5d = sm.OLS(d['ret_5d_demean'], X).fit(cov_type='cluster', cov_kwds={'groups': d['ticker']})
            beta_5d, p_5d = model_5d.params['bias_demean'], model_5d.pvalues['bias_demean']

            # 14-Day
            model_14d = sm.OLS(d['ret_14d_demean'], X).fit(cov_type='cluster', cov_kwds={'groups': d['ticker']})
            beta_14d, p_14d = model_14d.params['bias_demean'], model_14d.pvalues['bias_demean']

            # 30-Day
            model_30d = sm.OLS(d['ret_30d_demean'], X).fit(cov_type='cluster', cov_kwds={'groups': d['ticker']})
            beta_30d, p_30d = model_30d.params['bias_demean'], model_30d.pvalues['bias_demean']
            
            def get_sig(p): return "Significant" if p < 0.05 else "Insignificant"

            results_txt += f"| {label} | 1-Day | {beta_1d:.4f} | {p_1d:.4f} | {get_sig(p_1d)} |\n"
            results_txt += f"| {label} | 5-Day | {beta_5d:.4f} | {p_5d:.4f} | {get_sig(p_5d)} |\n"
            results_txt += f"| {label} | 14-Day | {beta_14d:.4f} | {p_14d:.4f} | {get_sig(p_14d)} |\n"
            results_txt += f"| {label} | 30-Day | {beta_30d:.4f} | {p_30d:.4f} | {get_sig(p_30d)} |\n"
            
            models_data.append({'label': f'{label} 1D', 'beta': beta_1d, 'err': model_1d.bse['bias_demean']})
            models_data.append({'label': f'{label} 5D', 'beta': beta_5d, 'err': model_5d.bse['bias_demean']})
            models_data.append({'label': f'{label} 14D', 'beta': beta_14d, 'err': model_14d.bse['bias_demean']})
            models_data.append({'label': f'{label} 30D', 'beta': beta_30d, 'err': model_30d.bse['bias_demean']})

        print(results_txt)
        
        with open('visualizations/impact_analysis/regression_results.txt', 'w') as f:
            f.write(results_txt)

        # Visualize Coefficients Comparison
        labels = [m['label'] for m in models_data]
        betas = [m['beta'] for m in models_data]
        errors = [m['err'] for m in models_data]
        
        plt.figure(figsize=(14, 6))
        # Colors not critical just differentiation
        plt.bar(labels, betas, yerr=[e*1.96 for e in errors], capsize=5, alpha=0.8)
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0, color='black', linewidth=1)
        plt.title("Impact Sensitivities: Across Time Horizons\n")
        plt.ylabel("Beta (Impact Coefficient)")
        plt.tight_layout()
        plt.savefig('visualizations/impact_analysis/impact_beta_plot.png', dpi=300)
        print("Saved visualizations/impact_analysis/impact_beta_plot.png")

        # --- Study 3: Aggregated Granger Causality ---
        print("\n--- 3. Aggregated Granger Causality ---")
        top_tickers = data['ticker'].value_counts().head(50).index.tolist()
        
        p_values = []
        significant_count = 0
        
        for ticker in top_tickers:
            sub = data[data['ticker'] == ticker]
            if len(sub) > 50:
                try:
                    # Test: Bias -> Return (Lag 3)
                    gc_res = grangercausalitytests(sub[['ret_next', 'norm_bias_score']], maxlag=[3], verbose=False)
                    p_val = gc_res[3][0]['ssr_ftest'][1]
                    p_values.append(p_val)
                    if p_val < 0.05:
                        significant_count += 1
                except:
                    pass
        
        prevalence = (significant_count / len(p_values)) * 100 if p_values else 0
        print(f"Prevalence of Causality: {prevalence:.2f}% of top companies show significant relation (p<0.05)")
        
        # Plot P-Value Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(p_values, bins=20, kde=True, color='purple')
        plt.axvline(0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
        plt.title(f"Distribution of Granger Causality P-Values\n{prevalence:.1f}% Significant", fontsize=14)
        plt.xlabel("P-Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/impact_analysis/impact_granger_dist.png', dpi=300)
        print("Saved visualizations/impact_analysis/impact_granger_dist.png")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_impact_rigorous()
