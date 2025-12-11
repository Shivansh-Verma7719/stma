import pandas as pd
import numpy as np


def calculate_sma_numpy(src: np.ndarray, length: int) -> np.ndarray:
    """
    Replacement for talib.SMA using Pandas rolling mean.
    """
    return pd.Series(src).rolling(window=length).mean().to_numpy()


def calculate_ema_numpy(src: np.ndarray, length: int) -> np.ndarray:
    """
    Replacement for talib.EMA using Pandas ewm (Exponential Weighted Functions).
    """
    return pd.Series(src).ewm(span=length, adjust=False).mean().to_numpy()


def calc_smma(src: np.ndarray, length: int) -> np.ndarray:
    """
    Calculates the Smoothed Moving Average (SMMA).
    """
    smma = np.full_like(src, fill_value=np.nan)

    # Calculate SMA for initialization purposes
    sma = calculate_sma_numpy(src, length)

    for i in range(1, len(src)):
        # Initialize SMMA with SMA when the first valid SMA value is available
        if np.isnan(smma[i - 1]):
            smma[i] = sma[i]
        else:
            # Recursive SMMA formula
            smma[i] = (smma[i - 1] * (length - 1) + src[i]) / length

    return smma


def calc_zlema(src: np.ndarray, length: int) -> np.ndarray:
    """
    Calculates the Zero-Lag Exponential Moving Average (ZLEMA).
    """
    ema1 = calculate_ema_numpy(src, length)
    ema2 = calculate_ema_numpy(ema1, length)
    d = ema1 - ema2
    return ema1 + d


def macd(data, lengthMA, lengthSignal):
    """
    Calculates the Impulse MACD.
    """
    # Calculate source (Average of High, Low, Close)
    src = (
        data["inth"].to_numpy(dtype=np.double)
        + data["intl"].to_numpy(dtype=np.double)
        + data["intc"].to_numpy(dtype=np.double)
    ) / 3

    # Calculate indicators
    hi = calc_smma(data["inth"].to_numpy(dtype=np.double), lengthMA)
    lo = calc_smma(data["intl"].to_numpy(dtype=np.double), lengthMA)
    mi = calc_zlema(src, lengthMA)

    # Determine Impulse MACD value based on range position
    conditions = [mi > hi, mi < lo]
    choices = [mi - hi, mi - lo]
    md = np.select(conditions, choices, default=0)

    # Signal line (SMA of the Impulse MACD)
    sb = calculate_sma_numpy(md, lengthSignal)

    # Histogram
    sh = md - sb

    # Compile results
    res = pd.DataFrame(
        {
            "open_time": data["time"],
            "ImpulseMACD": md,
            "ImpulseHisto": sh,
            "ImpulseMACDCDSignal": sb,
        }
    )
    return res
