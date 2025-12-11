import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from numba import jit


def squeeze_original(df, conv, length, col='intc'):
    df['max'] = df[col]
    df['min'] = df[col]

    for i in df.index[1:]:
        df.loc[i, 'max'] = max(df.loc[i, col], df.loc[i-1, 'max'] - (df.loc[i-1, 'max'] - df.loc[i, col]) / conv)
        df.loc[i, 'min'] = min(df.loc[i, col], df.loc[i-1, 'min'] + (df.loc[i, col] - df.loc[i-1, 'min']) / conv)
    
    df['diff'] = df['max'] - df['min']
    df['diff'] = np.where(df['diff'] <= 0, np.nan, df['diff']) 
    df['log_diff'] = np.log(df['diff'])

    df['psi'] = np.nan
    for i in range(length-1, len(df)):
        if df['log_diff'].iloc[i-length+1:i+1].isnull().any():
            continue
        corr, _ = pearsonr(df['log_diff'].iloc[i-length+1:i+1], np.arange(length))
        df.loc[i, 'psi'] = -50 * corr + 50

    df.drop(['max', 'min', 'diff', 'log_diff'], axis=1, inplace=True)
    
    return df

# def compute_max_min(df_col, conv):
#     max_col = df_col.copy()
#     min_col = df_col.copy()
#     for i in range(1, len(df_col)):
#         max_col[i] = max(max_col[i], max_col[i-1] - (max_col[i-1] - max_col[i]) / conv)
#         min_col[i] = min(min_col[i], min_col[i-1] + (min_col[i] - min_col[i-1]) / conv)
#     return max_col, min_col

# def squeeze_index2(df, conv, length, col='intc'):
#     df_col = df[col].values
#     max_col, min_col = compute_max_min(df_col, conv)
    
#     diff = max_col - min_col
#     diff[diff <= 0] = np.nan
#     log_diff = np.log(diff)
    
#     seq = np.arange(length)
#     log_diff_series = pd.Series(log_diff)
#     rolling_windows = log_diff_series.rolling(window=length)
    
#     def compute_psi(window):
#         if window.isnull().any():
#             return np.nan
#         corr = np.corrcoef(window, seq)[0, 1]
#         return -50 * corr + 50
    
#     psi = rolling_windows.apply(compute_psi, raw=False)
#     df['psi'] = psi
    
#     return df



def compute_max_min(df_col, conv):
    max_col = df_col.copy()
    min_col = df_col.copy()
    for i in range(1, len(df_col)):
        max_col[i] = max(max_col[i], max_col[i-1] - (max_col[i-1] - max_col[i]) / conv)
        min_col[i] = min(min_col[i], min_col[i-1] + (min_col[i] - min_col[i-1]) / conv)
    return max_col, min_col

def compute_psi_array(log_diff, length):
    psi = np.full(log_diff.shape, np.nan)
    seq = np.arange(length)
    for i in range(length - 1, len(log_diff)):
        window = log_diff[i - length + 1: i + 1]
        if np.any(np.isnan(window)):
            continue
        corr = np.corrcoef(window, seq)[0, 1]
        psi[i] = -50 * corr + 50
    return psi

def squeeze_index2(df, conv, length, col='intc'):
    df_col = df[col].values
    max_col, min_col = compute_max_min(df_col, conv)
    
    diff = max_col - min_col
    diff[diff <= 0] = np.nan
    log_diff = np.log(diff)
    
    psi = compute_psi_array(log_diff, length)
    df['psi'] = psi
    
    return df






def compute_max_min_float(df_col, conv):
    max_col = df_col.copy()
    min_col = df_col.copy()
    for i in range(1, len(df_col)):
        max_col[i] = max(max_col[i], max_col[i-1] - (max_col[i-1] - max_col[i]) / conv)
        min_col[i] = min(min_col[i], min_col[i-1] + (min_col[i] - min_col[i-1]) / conv)
    return max_col, min_col

def compute_psi_array_float(log_diff, length):
    psi = np.full(log_diff.shape, np.nan)
    int_length = int(length)
    seq = np.linspace(0, int_length - 1, int_length)
    for i in range(int_length - 1, len(log_diff)):
        window = log_diff[int(i - int_length + 1): int(i + 1)]
        if np.any(np.isnan(window)):
            continue
        corr = np.corrcoef(window, seq)[0, 1]
        psi[i] = -50 * corr + 50
    return psi

def squeeze_index2_float(df, conv, length, col='intc'):
    df_col = df[col].values
    max_col, min_col = compute_max_min_float(df_col, conv)
    
    diff = max_col - min_col
    diff[diff <= 0] = np.nan
    log_diff = np.log(diff)
    
    psi = compute_psi_array_float(log_diff, length)
    df['psi'] = psi
    
    return df