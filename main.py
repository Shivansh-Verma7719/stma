import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.console import Console

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipelines'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from helpers.db_helper import get_db_connection
from bias_calculator import run_bias_calculation
from normalize_bias_scores import normalize_bias_scores
from impact_analysis import analyze_impact_rigorous
from visualize_bias_index import visualize_bias_index

console = Console()

def plot_article_distribution(conn):
    """Plot distribution of total article counts per company."""
    
    console.print("\n[cyan]Generating Article Count Distribution...[/cyan]")
    query = "SELECT current_page FROM companies WHERE current_page > 0"
    df = pd.read_sql(query, conn)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['current_page'], bins=50, kde=True, color='blue')
    plt.title("Distribution of Article Counts per Company")
    plt.xlabel("Number of Articles")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/article_count_distribution.png', dpi=300)
    console.print("[green]Saved visualizations/article_count_distribution.png[/green]")

def plot_top_articles(conn):
    """Plot Top 20 companies by article count."""
    console.print("\n[cyan]Generating Top 20 Companies by Article Volume...[/cyan]")
    query = "SELECT symbol, current_page FROM companies ORDER BY current_page DESC LIMIT 20"
    df = pd.read_sql(query, conn)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='symbol', y='current_page', palette='viridis')
    plt.title("Top 20 Companies by Article Count")
    plt.xlabel("Ticker")
    plt.ylabel("Article Count")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/top_20_articles.png', dpi=300)
    console.print("[green]Saved visualizations/top_20_articles.png[/green]")

def main():
    console.print("[bold green]Starting End-to-End Pipeline[/bold green]")
    os.makedirs('visualizations/impact_analysis', exist_ok=True)
    
    # 1. Visualization Standard
    console.print("\n[bold]Step 1: Generating Standard Visualizations[/bold]")
    visualize_bias_index()
    
    # 2. Exploratory Plots
    console.print("\n[bold]Step 2: Generating Exploratory Plots[/bold]")
    conn = get_db_connection()
    try:
        plot_article_distribution(conn)
        plot_top_articles(conn)
    finally:
        conn.close()
        
    # 5. Impact Analysis
    console.print("\n[bold]Step 5: Impact Analysis (Rigorous)[/bold]")
    analyze_impact_rigorous()
    
    # 6. Anomaly Detection Pipeline
    console.print("\n[bold]Step 6: Anomaly Detection Pipeline[/bold]")
    
    # Add anomaly_det to path so its internal imports (helper, etc.) work
    ad_path = os.path.join(os.path.dirname(__file__), 'anomaly_det')
    if ad_path not in sys.path:
        sys.path.insert(0, ad_path)
    
    try:
        # Load anomaly_det/main.py as a module named 'anomaly_pipeline_module'
        import importlib.util
        spec = importlib.util.spec_from_file_location("anomaly_pipeline_module", os.path.join(ad_path, "main.py"))
        ad_module = importlib.util.module_from_spec(spec)
        sys.modules["anomaly_pipeline_module"] = ad_module
        spec.loader.exec_module(ad_module)
        
        # Run the pipeline
        ad_module.run_anomaly_pipeline()
    except Exception as e:
        console.print(f"[red]Error running Anomaly Detection Pipeline: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    console.print("\n[bold green]Pipeline Completed Successfully![/bold green]")

if __name__ == "__main__":
    main()
