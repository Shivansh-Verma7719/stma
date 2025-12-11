import pandas as pd
import numpy as np


def calculate(df, lookback, ema_length):
    velocity_list = []
    for i in range(1, lookback + 1):
        velocity_i = (df['intc'] - df['intc'].shift(i)) / i
        velocity_list.append(velocity_i)

    df.loc[:, 'velocity'] = pd.concat(velocity_list, axis=1).sum(axis=1) / lookback
    df.loc[:, 'smooth_velocity'] = df['velocity'].ewm(span=ema_length, adjust=False).mean()

    return df



def calculate_float(df, lookback, ema_length):
    velocity_list = []
    steps = np.linspace(1, lookback, num=int(lookback))
    
    for step in steps:
        velocity_i = (df['intc'] - df['intc'].shift(int(step))) / step
        velocity_list.append(velocity_i)
    
    df.loc[:, 'velocity'] = pd.concat(velocity_list, axis=1).sum(axis=1) / len(steps)
    df.loc[:, 'smooth_velocity'] = df['velocity'].ewm(span=ema_length, adjust=False).mean()

    return df
