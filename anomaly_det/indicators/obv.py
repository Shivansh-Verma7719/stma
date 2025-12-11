import pandas as pd
import numpy as np


def calcSlope(src, length):
    sumX, sumY, sumXSqr, sumXY = 0.0, 0.0, 0.0, 0.0
    for i in range(1, length + 1):
        val = src.iloc[length - i]
        per = i + 1.0
        sumX += per
        sumY += val
        sumXSqr += per * per
        sumXY += val * per
    slope = (length * sumXY - sumX * sumY) / (length * sumXSqr - sumX ** 2)
    average = sumY / length
    intercept = average - slope * sumX / length + slope
    return slope, average, intercept

def add_b5_column(df, column_name="macd", length=2, p=1):
    src5 = df[column_name]
    s, _, i = calcSlope(src5, length)
    tt1 = i + s * (length - 0) # assuming offset is 0
    
    df['n5'] = range(len(df))
    df['a15'] = abs(src5 - src5.shift(1)).cumsum() / df['n5'] * p
    df['b5'] = np.where(src5 > src5.shift(1) + df['a15'], src5,
                        np.where(src5 < src5.shift(1) - df['a15'], src5, src5.shift(1)))
    df['b5'] = df['b5'].fillna(method='ffill')
    
    df.drop(columns=['n5', 'a15'], inplace=True) # Cleanup, not required for the final result
    
    return df

def calculate_custom_indicators(df, window_len, v_len, len10, slow_length):
    # Ensure the DataFrame has the necessary columns: 'close', 'high', 'low', 'volume'
        
    # Calculating price spread and volume modified OBV
    price_spread = (df['inth'] - df['intl']).rolling(window=window_len).std(ddof=0)
    df['v1'] = np.sign(df['intc'].diff()).multiply(df['v']).cumsum()
    smooth = df['v1'].rolling(window=v_len).mean()
    v_spread = (df['v1'] - smooth).rolling(window=window_len).std()
    shadow = (df['v1'] - smooth) / v_spread * price_spread
    
    df['out'] = np.where(shadow > 0, df['inth'] + shadow, df['intl'] + shadow)
    
    # Calculating OBV EMA
    df['obvema'] = df['out'].ewm(span=len10, adjust=False).mean()
    
    # Calculating DEMA
    def dema(series, length):
        ema1 = series.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return 2 * ema1 - ema2
    
    df['dema'] = dema(df['obvema'], 9)  # Using 9 as an example length for DEMA
    
    # MACD calculation
    slow_ma = df['intc'].ewm(span=slow_length, adjust=False).mean()
    df['macd'] = df['dema'] - slow_ma
    
    df = add_b5_column(df)
    return df