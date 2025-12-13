import pandas as pd
import numpy as np


def calculate(df, lookback, ema_length):
    velocity_list = []
    for i in range(1, lookback + 1):
        # OLD LOGIC (Dollar Change):
        # velocity_i = (df['intc'] - df['intc'].shift(i)) / i

        # NEW LOGIC (Percentage Change):
        # We calculate the % move over 'i' days, then normalize by time 'i'
        # Formula: ((Price_t / Price_t-i) - 1) / i
        velocity_i = ((df["intc"] / df["intc"].shift(i)) - 1) / i

        velocity_list.append(velocity_i)

    # Summing average daily % velocities
    df.loc[:, "velocity"] = pd.concat(velocity_list, axis=1).sum(axis=1) / lookback

    # Smooth the result
    df.loc[:, "smooth_velocity"] = (
        df["velocity"].ewm(span=ema_length, adjust=False).mean()
    )

    # Scale up by 100 so the values are readable (e.g., 0.01 becomes 1.0)
    # This helps the Machine Learning model converge faster
    df.loc[:, "smooth_velocity"] = df.loc[:, "smooth_velocity"] * 100

    return df


def calculate_float(df, lookback, ema_length):
    velocity_list = []
    steps = np.linspace(1, lookback, num=int(lookback))

    for step in steps:
        step_int = int(step)
        if step_int == 0:
            continue

        # NEW LOGIC (Percentage Change)
        velocity_i = ((df["intc"] / df["intc"].shift(step_int)) - 1) / step
        velocity_list.append(velocity_i)

    df.loc[:, "velocity"] = pd.concat(velocity_list, axis=1).sum(axis=1) / len(steps)
    df.loc[:, "smooth_velocity"] = (
        df["velocity"].ewm(span=ema_length, adjust=False).mean()
    )

    # Scale up by 100
    df.loc[:, "smooth_velocity"] = df.loc[:, "smooth_velocity"] * 100

    return df
