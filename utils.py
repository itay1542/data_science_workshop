from datetime import timedelta
import pandas as pd
import numpy as np
from pandas._libs.tslibs.offsets import BDay
from scipy.signal import find_peaks


def get_daily_df(minute_df, agg_dict):
    df_daily = minute_df.groupby('symbol').resample('D').agg(agg_dict).dropna()
    df_daily["symbol"] = df_daily.index.get_level_values(0)
    df_daily["datetime"] = df_daily.index.get_level_values(1)
    df_daily = df_daily.reset_index(drop=True)
    df_daily["date"] = df_daily["datetime"].dt.date
    return df_daily

def get_hourly_df(minute_df, agg_dict):
    df_hourly = minute_df.groupby('symbol').resample('H').agg(agg_dict).dropna()
    df_hourly["symbol"] = df_hourly.index.get_level_values(0)
    df_hourly["datetime"] = df_hourly.index.get_level_values(1)
    df_hourly = df_hourly.reset_index(drop=True)
    df_hourly["date"] = df_hourly["datetime"].dt.date
    df_hourly["hour"] = df_hourly["datetime"].dt.hour
    return df_hourly


def get_five_minute_df(minute_df, agg_dict):
    df_five_minute = minute_df.groupby('symbol').resample('5T').agg(agg_dict).dropna()
    df_five_minute["symbol"] = df_five_minute.index.get_level_values(0)
    df_five_minute["datetime"] = df_five_minute.index.get_level_values(1)
    df_five_minute = df_five_minute.reset_index(drop=True)
    df_five_minute["date"] = df_five_minute["datetime"].dt.date
    return df_five_minute

def get_next_trading_day(datetime):
    bdays = BDay()
    the_day_after = datetime + timedelta(days=1)
    is_business_day = bdays.is_on_offset(the_day_after)
    while not is_business_day:
        the_day_after = the_day_after + timedelta(days=1)
        is_business_day = bdays.is_on_offset(the_day_after)
    return the_day_after


def historic_resistances(df, symbol, date=None, resistance_min_pivot_rank=2, strong_peak_prominence_pct=0.1,
                         strong_peak_distance=22, peak_distance=15, peak_rank_w_pct=0.02, include_high=True):
    if "symbol" in df.columns:
        symbol_df = df.copy()[df['symbol'] == symbol].reset_index(drop=True)
    else:
        symbol_df = df.copy().reset_index(drop=True)
    if date:
        symbol_df_past = symbol_df[symbol_df['date'] < date].reset_index()
    else:
        symbol_df_past = symbol_df
    peaks, _ = find_peaks(symbol_df_past['high'], distance=peak_distance)
    strong_peaks, _ = find_peaks(symbol_df_past['high'], distance=strong_peak_distance,
                                 prominence=symbol_df_past['close'].mean() * strong_peak_prominence_pct)
    strong_peaks_values = symbol_df_past.iloc[strong_peaks]["high"].values.tolist()

    yearly_high = symbol_df_past[symbol_df_past["date"] >= (date - pd.Timedelta(days=365))]["high"].max()
    if include_high:
        strong_peaks_values.extend([yearly_high])

    peak_to_rank = {peak: 0 for peak in peaks}
    for i, current_peak in enumerate(peaks):
        peak_rank_w = symbol_df_past[:current_peak].close.mean() * peak_rank_w_pct
        current_high = symbol_df_past.at[current_peak, "high"]
        for previous_peak in peaks[:i]:
            if abs(current_high - symbol_df_past.at[previous_peak, "high"]) <= peak_rank_w:
                peak_to_rank[current_peak] += 1

    resistances = strong_peaks_values
    for peak, rank in peak_to_rank.items():
        if rank >= resistance_min_pivot_rank:
            resistances.append(symbol_df_past.at[peak, "high"] + 1e-3)

    if not resistances:
        return []
    resistances.sort()
    # reduce resistances if they are too close to each other
    resistance_bins = []

    current_bin = [resistances[0]]

    for r in resistances:
        if r - current_bin[-1] < peak_rank_w_pct * r:
            current_bin.append(r)
        else:
            resistance_bins.append(current_bin)
            current_bin = [r]

    resistance_bins.append(current_bin)
    means = [np.mean(bin) for bin in resistance_bins]

    return means


def historic_supports(df, symbol, date=None, support_min_pivot_rank=2, strong_trough_prominence_pct=0.1,
                      strong_trough_distance=22, trough_distance=15, trough_rank_w_pct=0.02, include_low=True):
    if "symbol" in df.columns:
        symbol_df = df.copy()[df['symbol'] == symbol].reset_index(drop=True)
    else:
        symbol_df = df.copy().reset_index(drop=True)
    if date:
        symbol_df_past = symbol_df[symbol_df['date'] < date].reset_index()
    else:
        symbol_df_past = symbol_df

    # Identify troughs (lows) instead of peaks (highs)
    troughs, _ = find_peaks(-symbol_df_past['low'], distance=trough_distance)
    strong_troughs, _ = find_peaks(-symbol_df_past['low'], distance=strong_trough_distance,
                                   prominence=symbol_df_past['close'].mean() * strong_trough_prominence_pct)
    strong_troughs_values = symbol_df_past.iloc[strong_troughs]["low"].values.tolist()

    yearly_low = symbol_df_past[symbol_df_past["date"] >= (date - pd.Timedelta(days=365))]["low"].min()
    if include_low:
        strong_troughs_values.extend([yearly_low])

    trough_to_rank = {trough: 0 for trough in troughs}
    for i, current_trough in enumerate(troughs):
        trough_rank_w = symbol_df_past[:current_trough].close.mean() * trough_rank_w_pct
        current_low = symbol_df_past.at[current_trough, "low"]
        for previous_trough in troughs[:i]:
            if abs(current_low - symbol_df_past.at[previous_trough, "low"]) <= trough_rank_w:
                trough_to_rank[current_trough] += 1

    supports = strong_troughs_values
    for trough, rank in trough_to_rank.items():
        if rank >= support_min_pivot_rank:
            supports.append(symbol_df_past.at[trough, "low"] - 1e-3)

    if not supports:
        return []
    supports.sort()

    # Reduce supports if they are too close to each other
    support_bins = []
    current_bin = [supports[0]]

    for s in supports:
        if s - current_bin[-1] < trough_rank_w_pct * s:
            current_bin.append(s)
        else:
            support_bins.append(current_bin)
            current_bin = [s]

    support_bins.append(current_bin)
    means = [np.mean(bin) for bin in support_bins]

    return means