import pandas as pd
import numpy as np
import os
import glob
import concurrent.futures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set visual style
sns.set_theme(style="whitegrid")

# --- CONFIGURATION ---
INPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/fin_process/"
OUTPUT_DIR = "/Users/arnav/Desktop/workspaces/stma/stma/anomaly_det/narrative_test/"
MAX_WORKERS = 12  # CPU optimized


def run_momentum_test(input_file, output_file):
    """
    Tests if Price Returns today affect Media Bias for the next 28 days.
    """
    df = pd.read_csv(input_file)
    required_cols = ["bias_index", "intc", "time", "stock_name"]
    if not all(col in df.columns for col in required_cols):
        return "Skipped: Missing columns"

    # 1. Calculate Today's Price Return
    df["log_returns"] = np.log(df["intc"] / df["intc"].shift(1))

    # 2. Prepare Future Media Bias Windows
    # We check if Price(t) predicts Bias(t+1 ... t+N)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=7)
    df["future_bias_7d"] = df["bias_index"].rolling(window=indexer).mean()

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=14)
    df["future_bias_14d"] = df["bias_index"].rolling(window=indexer).mean()

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=28)
    df["future_bias_28d"] = df["bias_index"].rolling(window=indexer).mean()

    # Drop NaNs
    data = df.dropna(subset=["log_returns", "future_bias_28d"]).copy()

    if len(data) < 50:
        return "Skipped: Not enough data"

    # 3. Run Regressions
    X = data[["log_returns"]]

    # Model 7 Days
    model_7 = LinearRegression().fit(X, data["future_bias_7d"])
    r2_7 = model_7.score(X, data["future_bias_7d"])
    slope_7 = model_7.coef_[0]

    # Model 14 Days
    model_14 = LinearRegression().fit(X, data["future_bias_14d"])
    r2_14 = model_14.score(X, data["future_bias_14d"])
    slope_14 = model_14.coef_[0]

    # Model 28 Days
    model_28 = LinearRegression().fit(X, data["future_bias_28d"])
    r2_28 = model_28.score(X, data["future_bias_28d"])
    slope_28 = model_28.coef_[0]

    # 4. Save Summary
    summary_df = pd.DataFrame(
        [
            {
                "stock_name": data["stock_name"].iloc[0],
                # Strength (Does it stick?)
                "strength_7d": r2_7,
                "strength_14d": r2_14,
                "strength_28d": r2_28,
                # Direction (Do they follow?)
                "slope_7d": slope_7,
                "slope_28d": slope_28,
                # Classification (Based on 28-day lingering effect)
                # Slope > 0: Price Up -> Bias Positive (Follower)
                # Slope < 0: Price Up -> Bias Negative (Contrarian)
                "behavior": "Trend Follower" if slope_28 > 0 else "Contrarian",
            }
        ]
    )

    summary_df.to_csv(output_file, index=False)
    return "Success"


def process_single_file(file_path):
    try:
        filename = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, f"MOMENTUM_{filename}")
        status = run_momentum_test(file_path, output_path)
        return f"{status}: {filename}"
    except Exception as e:
        return f"Error: {e}"


def generate_momentum_plots(master_df, output_dir):
    """
    Generates visualizations for Media Hangover and Direction.
    """
    print("Generating Narrative Momentum Plots...")

    # --- PLOT 1: The Hangover Curve (Strength) ---
    means = {
        "7 Days": master_df["strength_7d"].mean(),
        "14 Days": master_df["strength_14d"].mean(),
        "28 Days": master_df["strength_28d"].mean(),
    }

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        means.keys(), means.values(), color=["#ff9999", "#66b3ff", "#99ff99"]
    )
    plt.title(
        "The 'Media Hangover': How Long Does Price Impact Sentiment?", fontsize=14
    )
    plt.ylabel("Predictive Strength (R-Squared)", fontsize=12)
    plt.xlabel("Time after Price Move", fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.5f}",
            va="bottom",
            ha="center",
        )

    plt.savefig(os.path.join(output_dir, "plot_1_hangover_strength.png"))
    plt.close()

    # --- PLOT 2: The Follower Ratio (Direction) ---
    # We count how many stocks are "Trend Followers" vs "Contrarians"
    counts = master_df["behavior"].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=["#ffcc99", "#99ff99"],
        startangle=140,
    )
    plt.title("Is the Media a 'Trend Follower'?", fontsize=14)

    plt.savefig(os.path.join(output_dir, "plot_2_media_behavior.png"))
    plt.close()

    print("Plots saved.")


if __name__ == "__main__":
    start_time = time.time()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Testing Narrative Momentum on {len(csv_files)} stocks...")

    # 1. Parallel Execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_single_file, csv_files))

    # 2. Aggregation
    print("Aggregating results...")
    all_summaries = []
    result_files = glob.glob(os.path.join(OUTPUT_DIR, "MOMENTUM_*.csv"))

    for f in result_files:
        all_summaries.append(pd.read_csv(f))

    if all_summaries:
        master_df = pd.concat(all_summaries)
        master_path = os.path.join(OUTPUT_DIR, "_MASTER_MOMENTUM_REPORT.csv")
        master_df.to_csv(master_path, index=False)

        # 3. Generate Visuals
        generate_momentum_plots(master_df, OUTPUT_DIR)

        # Console Report
        avg_7 = master_df["strength_7d"].mean()
        avg_28 = master_df["strength_28d"].mean()
        follower_pct = (master_df["behavior"] == "Trend Follower").mean() * 100

        print("\n=== NARRATIVE MOMENTUM RESULTS ===")
        print(f"Momentum Strength (7 Days):  {avg_7:.5f}")
        print(f"Momentum Strength (28 Days): {avg_28:.5f}")
        print(f"Media Behavior: {follower_pct:.1f}% Trend Followers")

        if follower_pct > 50:
            print("VERDICT: The Media generally FOLLOWS the Price Trend.")
        else:
            print("VERDICT: The Media generally FIGHTS the Price Trend.")

    print(f"Done in {time.time() - start_time:.2f}s")
