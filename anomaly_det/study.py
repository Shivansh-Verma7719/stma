import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- CONFIGURATION ---
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "fin_process")
OUTPUT_DIR = os.path.join(BASE_DIR, "event_study")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

sns.set_theme(style="whitegrid")


def run_event_study_in_memory(data_dict):
    """
    Runs event study in-memory on a dictionary of DataFrames.
    Returns: tuple of (stats_df, positive_shock_bias, negative_shock_bias, pos_endpoints, neg_endpoints)
    """
    # Store trajectories
    positive_shock_bias = []
    negative_shock_bias = []
    pos_endpoints = []
    neg_endpoints = []

    # Parameters
    LOOK_BACK = 5
    LOOK_AHEAD = 10
    PCT_THRESHOLD = 0.02

    files_processed = 0

    for ticker, df in data_dict.items():
        try:
            df = df.copy()

            # Prep Data - Use only bias_index (from norm_bias_score)
            if "bias_index" in df.columns:
                bias_col = "bias_index"
            elif "norm_bias_score" in df.columns:
                df["bias_index"] = df["norm_bias_score"]
                bias_col = "bias_index"
            else:
                continue

            if "intc" not in df.columns:
                if "close" in df.columns:
                    df["intc"] = df["close"]
                else:
                    continue

            # Calculate Returns
            df["log_ret"] = np.log(df["intc"] / df["intc"].shift(1))

            # Reset index to ensure integer positions for iloc operations
            # This handles cases where the DataFrame has non-contiguous indices
            df = df.reset_index(drop=True)

            # Identify "Events" (Shocks) - now using integer positions
            pos_events = df[df["log_ret"] > PCT_THRESHOLD].index
            neg_events = df[df["log_ret"] < -PCT_THRESHOLD].index

            # Extract Trajectories
            # Process Positive Shocks
            for date_idx in pos_events:
                # date_idx is now an integer position after reset_index
                if date_idx < LOOK_BACK or date_idx > len(df) - LOOK_AHEAD - 1:
                    continue

                window = (
                    df[bias_col]
                    .iloc[date_idx - LOOK_BACK : date_idx + LOOK_AHEAD + 1]
                    .values
                )

                # Normalize (Day 0 = 0.0)
                day_0_bias = window[LOOK_BACK]
                normalized_window = window - day_0_bias

                positive_shock_bias.append(normalized_window)
                pos_endpoints.append(normalized_window[-1])

            # Process Negative Shocks
            for date_idx in neg_events:
                # date_idx is now an integer position after reset_index
                if date_idx < LOOK_BACK or date_idx > len(df) - LOOK_AHEAD - 1:
                    continue

                window = (
                    df[bias_col]
                    .iloc[date_idx - LOOK_BACK : date_idx + LOOK_AHEAD + 1]
                    .values
                )

                day_0_bias = window[LOOK_BACK]
                normalized_window = window - day_0_bias

                negative_shock_bias.append(normalized_window)
                neg_endpoints.append(normalized_window[-1])

            files_processed += 1

        except Exception:
            continue

    print(f"Processed {files_processed} stocks.")
    print(f"Found {len(positive_shock_bias)} Positive Shocks (>2%)")
    print(f"Found {len(negative_shock_bias)} Negative Shocks (<-2%)")

    if not positive_shock_bias or not negative_shock_bias:
        print("Not enough events found.")
        return (
            None,
            positive_shock_bias,
            negative_shock_bias,
            pos_endpoints,
            neg_endpoints,
        )

    # Statistical Calculations
    from scipy import stats

    t_pos, p_pos = stats.ttest_1samp(pos_endpoints, 0, alternative="greater")
    mean_pos = np.mean(pos_endpoints)
    verdict_pos = "Significant" if p_pos < 0.05 else "Noise"

    t_neg, p_neg = stats.ttest_1samp(neg_endpoints, 0, alternative="greater")
    mean_neg = np.mean(neg_endpoints)
    verdict_neg = "Significant" if p_neg < 0.05 else "Noise"

    # Create Summary DataFrame
    stats_data = {
        "Metric": ["Positive Price Shock (>2%)", "Negative Price Shock (<-2%)"],
        "Sample Size": [len(pos_endpoints), len(neg_endpoints)],
        "Mean Media Change (10 Days)": [mean_pos, mean_neg],
        "T-Statistic": [t_pos, t_neg],
        "P-Value": [p_pos, p_neg],
        "Verdict (95% Conf.)": [verdict_pos, verdict_neg],
    }
    stats_df = pd.DataFrame(stats_data)

    return (
        stats_df,
        positive_shock_bias,
        negative_shock_bias,
        pos_endpoints,
        neg_endpoints,
    )


def generate_event_study_plots(
    stats_df,
    positive_shock_bias,
    negative_shock_bias,
    pos_endpoints,
    neg_endpoints,
    output_dir,
):
    """
    Generates event study plots from in-memory data.
    """
    LOOK_BACK = 5
    LOOK_AHEAD = 10

    # Save CSV
    csv_path = os.path.join(output_dir, "event_study_stats.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"Stats exported to: {csv_path}")

    # Visualization 1: The Trajectory Graph
    avg_pos_path = np.mean(positive_shock_bias, axis=0)
    avg_neg_path = np.mean(negative_shock_bias, axis=0)
    sem_pos = np.std(positive_shock_bias, axis=0) / np.sqrt(len(positive_shock_bias))
    sem_neg = np.std(negative_shock_bias, axis=0) / np.sqrt(len(negative_shock_bias))

    days = range(-LOOK_BACK, LOOK_AHEAD + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(
        days,
        avg_pos_path,
        color="green",
        linewidth=3,
        label="Bias Path after Price PUMP (>2%)",
    )
    plt.fill_between(
        days,
        avg_pos_path - 1.96 * sem_pos,
        avg_pos_path + 1.96 * sem_pos,
        color="green",
        alpha=0.1,
    )

    plt.plot(
        days,
        avg_neg_path,
        color="red",
        linewidth=3,
        label="Bias Path after Price DUMP (<-2%)",
    )
    plt.fill_between(
        days,
        avg_neg_path - 1.96 * sem_neg,
        avg_neg_path + 1.96 * sem_neg,
        color="red",
        alpha=0.1,
    )

    plt.axvline(
        0, color="black", linestyle="--", linewidth=1, label="Event Day (Price Shock)"
    )
    plt.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    plt.xlabel("Days Relative to Price Shock (0 = Day of Shock)", fontsize=12)
    plt.ylabel("Change in Media Bias Score", fontsize=12)
    plt.title("The Event Study: How Media Reacts to Market Explosions", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "event_study_result.png")
    plt.savefig(plot_path)
    print(f"Graph saved to: {plot_path}")
    plt.close()

    # Visualization 2: The Stats Table
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    # Prepare table text
    table_data = []
    headers = stats_df.columns.tolist()

    for row in stats_df.itertuples(index=False):
        # Format numbers for prettiness
        formatted_row = [
            row[0],
            f"{row[1]:,}",
            f"{row[2]:.4f}",
            f"{row[3]:.2f}",
            f"{row[4]:.2e}",  # Scientific notation for P-value
            row[5],
        ]
        table_data.append(formatted_row)

    # Draw table
    the_table = ax.table(
        cellText=table_data, colLabels=headers, loc="center", cellLoc="center"
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 1.5)

    plt.title(
        "Statistical Significance of Media Reaction (Day 0 to Day 10)",
        fontsize=12,
        weight="bold",
    )

    table_path = os.path.join(output_dir, "event_study_stats_table.png")
    plt.savefig(table_path, bbox_inches="tight", dpi=300)
    print(f"Stats Table saved to: {table_path}")
    plt.close()


def run_event_study(input_dir):
    print("Running Event Study (Focusing ONLY on Extreme Price Moves)...")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # Store trajectories
    positive_shock_bias = []  # Bias path after Price Goes UP
    negative_shock_bias = []  # Bias path after Price Goes DOWN

    # Store just the endpoint changes (Day 0 to Day 10) for easier stats
    pos_endpoints = []
    neg_endpoints = []

    # Parameters
    LOOK_BACK = 5  # Days to look before the shock
    LOOK_AHEAD = 10  # Days to look after the shock
    PCT_THRESHOLD = 0.02  # Define a "Shock" as a 2% move

    files_processed = 0

    for f in all_files:
        try:
            df = pd.read_csv(f)

            # 1. Prep Data - Use only bias_index (from norm_bias_score)
            if "bias_index" in df.columns:
                bias_col = "bias_index"
            elif "norm_bias_score" in df.columns:
                df["bias_index"] = df["norm_bias_score"]
                bias_col = "bias_index"
            else:
                continue

            if "intc" not in df.columns:
                if "close" in df.columns:
                    df["intc"] = df["close"]
                else:
                    continue

            # Calculate Returns
            df["log_ret"] = np.log(df["intc"] / df["intc"].shift(1))

            # 2. Identify "Events" (Shocks)
            pos_events = df[df["log_ret"] > PCT_THRESHOLD].index
            neg_events = df[df["log_ret"] < -PCT_THRESHOLD].index

            # 3. Extract Trajectories
            # -- Process Positive Shocks --
            for date_idx in pos_events:
                if date_idx < LOOK_BACK or date_idx > len(df) - LOOK_AHEAD - 1:
                    continue

                window = (
                    df[bias_col]
                    .iloc[date_idx - LOOK_BACK : date_idx + LOOK_AHEAD + 1]
                    .values
                )

                # Normalize (Day 0 = 0.0)
                day_0_bias = window[LOOK_BACK]
                normalized_window = window - day_0_bias

                positive_shock_bias.append(normalized_window)
                pos_endpoints.append(normalized_window[-1])  # Value at Day 10

            # -- Process Negative Shocks --
            for date_idx in neg_events:
                if date_idx < LOOK_BACK or date_idx > len(df) - LOOK_AHEAD - 1:
                    continue

                window = (
                    df[bias_col]
                    .iloc[date_idx - LOOK_BACK : date_idx + LOOK_AHEAD + 1]
                    .values
                )

                day_0_bias = window[LOOK_BACK]
                normalized_window = window - day_0_bias

                negative_shock_bias.append(normalized_window)
                neg_endpoints.append(normalized_window[-1])  # Value at Day 10

            files_processed += 1

        except Exception:
            continue

    print(f"Processed {files_processed} stocks.")
    print(f"Found {len(positive_shock_bias)} Positive Shocks (>2%)")
    print(f"Found {len(negative_shock_bias)} Negative Shocks (<-2%)")

    if not positive_shock_bias or not negative_shock_bias:
        print("Not enough events found. Try lowering PCT_THRESHOLD.")
        return

    # --- 4. STATISTICAL CALCULATIONS ---
    print("\nCalculating Statistics...")

    # Positive Shocks Stats (Testing for Increase > 0)
    t_pos, p_pos = stats.ttest_1samp(pos_endpoints, 0, alternative="greater")
    mean_pos = np.mean(pos_endpoints)
    verdict_pos = "Significant" if p_pos < 0.05 else "Noise"

    # Negative Shocks Stats (Testing for Rebound > 0)
    t_neg, p_neg = stats.ttest_1samp(neg_endpoints, 0, alternative="greater")
    mean_neg = np.mean(neg_endpoints)
    verdict_neg = "Significant" if p_neg < 0.05 else "Noise"

    # Create Summary DataFrame
    stats_data = {
        "Metric": ["Positive Price Shock (>2%)", "Negative Price Shock (<-2%)"],
        "Sample Size": [len(pos_endpoints), len(neg_endpoints)],
        "Mean Media Change (10 Days)": [mean_pos, mean_neg],
        "T-Statistic": [t_pos, t_neg],
        "P-Value": [p_pos, p_neg],
        "Verdict (95% Conf.)": [verdict_pos, verdict_neg],
    }
    stats_df = pd.DataFrame(stats_data)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "event_study_stats.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"Stats exported to: {csv_path}")

    # --- 5. VISUALIZATION 1: The Trajectory Graph ---
    avg_pos_path = np.mean(positive_shock_bias, axis=0)
    avg_neg_path = np.mean(negative_shock_bias, axis=0)
    sem_pos = np.std(positive_shock_bias, axis=0) / np.sqrt(len(positive_shock_bias))
    sem_neg = np.std(negative_shock_bias, axis=0) / np.sqrt(len(negative_shock_bias))

    days = range(-LOOK_BACK, LOOK_AHEAD + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(
        days,
        avg_pos_path,
        color="green",
        linewidth=3,
        label="Bias Path after Price PUMP (>2%)",
    )
    plt.fill_between(
        days,
        avg_pos_path - 1.96 * sem_pos,
        avg_pos_path + 1.96 * sem_pos,
        color="green",
        alpha=0.1,
    )

    plt.plot(
        days,
        avg_neg_path,
        color="red",
        linewidth=3,
        label="Bias Path after Price DUMP (<-2%)",
    )
    plt.fill_between(
        days,
        avg_neg_path - 1.96 * sem_neg,
        avg_neg_path + 1.96 * sem_neg,
        color="red",
        alpha=0.1,
    )

    plt.axvline(
        0, color="black", linestyle="--", linewidth=1, label="Event Day (Price Shock)"
    )
    plt.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    plt.xlabel("Days Relative to Price Shock (0 = Day of Shock)", fontsize=12)
    plt.ylabel("Change in Media Bias Score", fontsize=12)
    plt.title("The Event Study: How Media Reacts to Market Explosions", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "event_study_result.png")
    plt.savefig(plot_path)
    print(f"Graph saved to: {plot_path}")
    plt.close()

    # --- 6. VISUALIZATION 2: The Stats Table ---
    # Create a clean graphical table of the stats
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    # Prepare table text
    table_data = []
    headers = stats_df.columns.tolist()

    for row in stats_df.itertuples(index=False):
        # Format numbers for prettiness
        formatted_row = [
            row[0],
            f"{row[1]:,}",
            f"{row[2]:.4f}",
            f"{row[3]:.2f}",
            f"{row[4]:.2e}",  # Scientific notation for P-value
            row[5],
        ]
        table_data.append(formatted_row)

    # Draw table
    the_table = ax.table(
        cellText=table_data, colLabels=headers, loc="center", cellLoc="center"
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 1.5)

    plt.title(
        "Statistical Significance of Media Reaction (Day 0 to Day 10)",
        fontsize=12,
        weight="bold",
    )

    table_path = os.path.join(OUTPUT_DIR, "event_study_stats_table.png")
    plt.savefig(table_path, bbox_inches="tight", dpi=300)
    print(f"Stats Table saved to: {table_path}")
    plt.close()


if __name__ == "__main__":
    run_event_study(INPUT_DIR)
