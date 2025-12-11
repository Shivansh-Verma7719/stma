import pandas as pd
import numpy as np


def EMA(df, base, target, period, alpha=False):
    con = pd.concat(
        [
            df[: int(period)][base].rolling(window=int(period)).mean(),
            df[int(period) :][base],
        ]
    )

    if alpha:
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        df[target] = con.ewm(span=period, adjust=False).mean()

    # FIX 1: Avoid inplace fillna warning
    df[target] = df[target].fillna(0)
    return df


def ATR(df, period, ohlc=["open", "high", "low", "close"]):
    atr = "ATR_" + str(period)

    # Compute true range only if it is not computed and stored earlier in the df
    if "TR" not in df.columns:
        df["h-l"] = df[ohlc[1]] - df[ohlc[2]]
        df["h-yc"] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df["l-yc"] = abs(df[ohlc[2]] - df[ohlc[3]].shift())

        df["TR"] = df[["h-l", "h-yc", "l-yc"]].max(axis=1)

        df.drop(["h-l", "h-yc", "l-yc"], inplace=True, axis=1)

    # Compute EMA of true range using ATR formula after ignoring first row
    EMA(df, "TR", atr, period, alpha=True)

    return df


def SuperTrend(df1, period, multiplier):
    ohlc = ["into", "inth", "intl", "intc"]
    df = df1.copy()
    multiplier = float(multiplier)

    ATR(df, period, ohlc=ohlc)
    atr = "ATR_" + str(period)
    st = "ST" + str(period) + "_" + str(multiplier)
    stx = "STX"

    df["basic_ub"] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
    df["basic_lb"] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]

    df["final_ub"] = 0.00
    df["final_lb"] = 0.00

    # Calculate Final Upper/Lower Bands
    # Note: Using .iat loop is slow but kept to preserve original logic/math exactness
    for i in range(int(period), len(df)):
        df["final_ub"].iat[i] = (
            df["basic_ub"].iat[i]
            if df["basic_ub"].iat[i] < df["final_ub"].iat[i - 1]
            or df[ohlc[3]].iat[i - 1] > df["final_ub"].iat[i - 1]
            else df["final_ub"].iat[i - 1]
        )
        df["final_lb"].iat[i] = (
            df["basic_lb"].iat[i]
            if df["basic_lb"].iat[i] > df["final_lb"].iat[i - 1]
            or df[ohlc[3]].iat[i - 1] < df["final_lb"].iat[i - 1]
            else df["final_lb"].iat[i - 1]
        )

    df[st] = 0.00
    for i in range(int(period), len(df)):
        df[st].iat[i] = (
            df["final_ub"].iat[i]
            if df[st].iat[i - 1] == df["final_ub"].iat[i - 1]
            and df[ohlc[3]].iat[i] <= df["final_ub"].iat[i]
            else (
                df["final_lb"].iat[i]
                if df[st].iat[i - 1] == df["final_ub"].iat[i - 1]
                and df[ohlc[3]].iat[i] > df["final_ub"].iat[i]
                else (
                    df["final_lb"].iat[i]
                    if df[st].iat[i - 1] == df["final_lb"].iat[i - 1]
                    and df[ohlc[3]].iat[i] >= df["final_lb"].iat[i]
                    else (
                        df["final_ub"].iat[i]
                        if df[st].iat[i - 1] == df["final_lb"].iat[i - 1]
                        and df[ohlc[3]].iat[i] < df["final_lb"].iat[i]
                        else 0.00
                    )
                )
            )
        )

    # FIX 2: Replacement for the crashing np.where line
    # Instead of mixing strings and NaN in one numpy call, we assign by condition
    df[stx] = np.nan  # Initialize with NaN

    # Condition 1: Trend is valid (non-zero) AND Close < SuperTrend -> DOWN
    cond_down = (df[st] > 0.00) & (df[ohlc[3]] < df[st])
    df.loc[cond_down, stx] = "down"

    # Condition 2: Trend is valid (non-zero) AND Close >= SuperTrend -> UP
    cond_up = (df[st] > 0.00) & (df[ohlc[3]] >= df[st])
    df.loc[cond_up, stx] = "up"

    # Cleanup intermediate columns
    df.drop(
        ["basic_ub", "basic_lb", "final_ub", "final_lb", "TR", atr, st],
        inplace=True,
        axis=1,
    )

    df[stx] = df[stx].fillna(
        "nan"
    )  # Optional: ensure no raw NaNs remain if preferred, or leave as NaN

    return df
