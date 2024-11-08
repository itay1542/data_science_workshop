import mplfinance as mpf
import numpy as np
import matplotlib.dates as mdates

import matplotlib.pyplot as plt

def plot_bars_with_indicators(ohlc, ax, title: str, addplots=[]):
    mpf.plot(
        ohlc.rename({'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, axis=1),
        type='candle',
        ax=ax,
        addplot=addplots,
        axtitle=title,
        ylabel='Price',
        style="yahoo"
    )


def plot_with_resistances_and_supports(df, ax):
    # Create a list to hold resistance line data
    addplot = []

    last_resistances = df.iloc[-1]["resistances"]
    last_supports = df.iloc[-1]["supports"]
    for resistance in last_resistances:
        addplot.append(mpf.make_addplot([resistance] * len(df), color='red', ax=ax))
    for support in last_supports:
        addplot.append(mpf.make_addplot([support] * len(df), color='green', ax=ax))


    # Plot the candlestick chart with resistance lines
    mpf.plot(df, type='candle', addplot=addplot, style='charles',
             volume=False, ax=ax)

def plot_intraday_with_key_level(df, is_pivot=False):
    # Create a list to hold resistance line data
    stock = df.iloc[0]["symbol"]
    addplot = []
    key_level_in_range = df["key_level_in_daily_range"]
    addplot.append(mpf.make_addplot(key_level_in_range, color='red'))
    mpf.plot(df, type='candle', addplot=addplot, style='charles', volume=True, title=f"{stock} Intraday with {'Pivot on' if is_pivot else 'Breakthrough'} Key Level")
    #todo: add volume

def plot_with_rsi(df, bar_ax, rsi_ax):
    # plot intraday data for a stock with RSI in a seperate panel
    mpf.plot(df, type='candle', volume=False, style='charles', ax=bar_ax)

    # Plot the RSI
    rsi_ax.plot(df["RSI"], color='red')
    rsi_ax.axhline(30, color='green', linestyle='--')  # Lower RSI threshold
    rsi_ax.axhline(70, color='green', linestyle='--')  # Upper RSI threshold
    rsi_ax.set_ylabel('RSI')

    # Set up the y-axis limits for better visual context
    rsi_ax.set_ylim(0, 100)
    rsi_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.show()
