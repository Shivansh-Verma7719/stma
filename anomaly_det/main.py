import warnings
import time
import pandas as pd

# Suppress FutureWarnings from pandas (deprecation warnings that don't affect functionality)
# These are just warnings about future pandas versions and don't affect current functionality
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ChainedAssignmentError.*")
warnings.filterwarnings("ignore", message=".*Setting an item of incompatible dtype.*")
warnings.filterwarnings("ignore", message=".*fillna with 'method' is deprecated.*")
warnings.filterwarnings("ignore", message=".*behaviour will change in pandas 3.0.*")

# Also suppress pandas warnings via pandas options
pd.options.mode.chained_assignment = None  # Suppress chained assignment warnings

from helper import fetch_data_from_db
from helper2 import process_market_data
from feature_engineering import process_dataframes_in_memory
from statistical import process_dataframes_in_memory as process_statistical
from inference import (
    aggregate_statistical_results_in_memory,
    generate_silent_shock_plot,
    generate_brief_report,
)
from media_follow import (
    process_dataframes_in_memory as process_media_follow,
    generate_pie_chart,
)
from study import run_event_study_in_memory
import os


def run_anomaly_pipeline():
    """
    Main pipeline that runs all processing steps in order.
    """
    start_time = time.time()
    print("=" * 60)
    print("ANOMALY DETECTION PIPELINE - IN-MEMORY MODE")
    print("=" * 60)

    # Step 1: Fetch data from database
    print("\n[Step 1/7] Fetching data from database...")
    data_dict = fetch_data_from_db()
    if not data_dict:
        print("ERROR: No data fetched from database. Exiting.")
        return
    print(f"✓ Fetched {len(data_dict)} tickers from database")

    # Step 2: Process market data (rename norm_bias_score to bias_index)
    print("\n[Step 2/7] Processing market data...")
    processed_data = process_market_data(data_dict)
    if not processed_data:
        print("ERROR: Market data processing failed. Exiting.")
        return
    print(f"✓ Processed {len(processed_data)} tickers")

    # Step 3: Feature engineering (add indicators)
    print("\n[Step 3/7] Running feature engineering...")
    feature_data = process_dataframes_in_memory(processed_data)
    if not feature_data:
        print("ERROR: Feature engineering failed. Exiting.")
        return
    print(f"✓ Added features to {len(feature_data)} tickers")

    # Step 4: Statistical anomaly detection
    print("\n[Step 4/7] Running statistical anomaly detection...")
    statistical_data = process_statistical(feature_data)
    if not statistical_data:
        print("ERROR: Statistical analysis failed. Exiting.")
        return
    print(f"✓ Analyzed {len(statistical_data)} tickers")

    # Step 5: Aggregate statistical results and generate reports
    print("\n[Step 5/7] Aggregating statistical results...")
    aggregated_df = aggregate_statistical_results_in_memory(statistical_data)
    if aggregated_df.empty:
        print("WARNING: No data to aggregate.")
    else:
        print(f"✓ Aggregated {len(aggregated_df)} rows")

        # Generate plots and reports (these still save files, but that's OK for final output)
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "final_study_output"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generate_silent_shock_plot(aggregated_df, output_dir)
        generate_brief_report(aggregated_df, output_dir)

    # Step 6: Media follow analysis
    print("\n[Step 6/7] Running media follow analysis...")
    media_results = process_media_follow(feature_data)
    if media_results:
        print(f"✓ Analyzed {len(media_results)} tickers")

        # Generate pie chart (saves file, but that's OK for final output)
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "narrative_test"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        master_df = pd.DataFrame(media_results)
        generate_pie_chart(master_df, output_dir)

        # Console summary
        follower_pct = (master_df["behavior"] == "Trend Follower").mean() * 100
        print("\n=== MEDIA FOLLOW VERDICT ===")
        print(f"Stocks Analyzed: {len(master_df)}")
        print(f"Trend Followers: {follower_pct:.1f}%")
    else:
        print("WARNING: No media follow results generated.")

    # Step 7: Event study
    print("\n[Step 7/7] Running event study...")
    event_result = run_event_study_in_memory(feature_data)
    if event_result[0] is not None:
        stats_df, pos_shocks, neg_shocks, pos_endpoints, neg_endpoints = event_result
        print(f"✓ Event study complete")
        print(f"  Positive shocks: {len(pos_shocks)}")
        print(f"  Negative shocks: {len(neg_shocks)}")
        print("\n=== EVENT STUDY STATISTICS ===")
        print(stats_df.to_string(index=False))

        # Generate plots (saves files, but that's OK for final output)
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "event_study"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        from study import generate_event_study_plots

        generate_event_study_plots(
            stats_df, pos_shocks, neg_shocks, pos_endpoints, neg_endpoints, output_dir
        )
    else:
        print("WARNING: Event study did not generate results.")

    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total execution time: {duration:.2f} seconds")
    print(f"Tickers processed: {len(processed_data)}")
    print("\nNote: Final output files (plots, reports) were saved to:")
    print("  - final_study_output/")
    print("  - narrative_test/")
    print("  - event_study/")


if __name__ == "__main__":
    run_anomaly_pipeline()
