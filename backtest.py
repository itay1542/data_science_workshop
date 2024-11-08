from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mplfinance as mpf
from pandas._libs.tslibs.offsets import Minute


@dataclass
class Position:
    shares: int
    entry_price: float
    entry_bar: pd.Series = None
    stop_loss: float = None
    take_profit: float = None
    ttl: int = float("inf")

@dataclass
class Trade:
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    is_long: bool
    entry_bar: pd.Series
    exit_bar: pd.Series


def total_pnl(pnl_list):
    return sum(pnl_list)


def maximum_drawdown(pnl_list):
    max_pnl = 0
    max_drawdown = 0
    for pnl in pnl_list:
        max_pnl = max(max_pnl, pnl)
        max_drawdown = max(max_drawdown, max_pnl - pnl)
    return max_drawdown


def sharpe_ratio(pnl_list):
    returns_std = np.std(pnl_list)
    return np.mean(pnl_list) / returns_std


def winning_trades_ratio(pnl_list):
    if not pnl_list:
        return 0
    return len([pnl for pnl in pnl_list if pnl > 0]) / len(pnl_list)


def backtest_ema_strategy(symbol_df, start_date, trade_value=1000, plot=False):
    trades = []
    symbol_df["standard_deviation"] = symbol_df["close"].rolling(window=30).std()  # half hour
    in_trade = False
    position = None
    for i, row in symbol_df.loc[start_date:].iterrows():
        if row["hour"] == 9:
            continue  # skip trading in the first 30 mins to let EMA stabilize
        if not in_trade and (row["hour"] < 15 or (row["hour"] == 15 and row["minute"] < 30)):
            if row["close"] <= row["EMA"] - row["standard_deviation"]:
                in_trade = True
                position = Position(trade_value // row["close"], row["close"])
            elif row["close"] >= row["EMA"] + row["standard_deviation"]:
                in_trade = True
                position = Position(-trade_value // row["close"], row["close"])
        else:
            if position.shares > 0 and row["close"] >= row["EMA"]:
                in_trade = False
                trades.append(
                    Trade(
                        entry_price=position.entry_price,
                        exit_price=row["close"],
                        shares=position.shares,
                        pnl=(row["close"] - position.entry_price) * position.shares,
                        is_long=True,
                        entry_bar=position.entry_bar,
                        exit_bar=row
                    )
                )
            elif position.shares < 0 and row["close"] <= row["EMA"]:
                in_trade = False
                trades.append(
                    Trade(
                        entry_price=position.entry_price,
                        exit_price=row["close"],
                        shares=position.shares,
                        pnl=(position.entry_price - row["close"]) * position.shares,
                        is_long=False,
                        entry_bar=position.entry_bar,
                        exit_bar=row
                    )
                )
            elif row["hour"] == 15 and row["minute"] == 30:  # close position at the end of the day
                in_trade = False
                trades.append(
                    Trade(
                        entry_price=position.entry_price,
                        exit_price=row["close"],
                        shares=position.shares,
                        pnl=(row["close"] - position.entry_price) * position.shares,
                        is_long=True,
                        entry_bar=position.entry_bar,
                        exit_bar=row
                    )
                )
    trades_pnls = [trade.pnl for trade in trades]
    if plot:
        plot_pnls(trades_pnls)

    return {
        "total_pnl": total_pnl(trades_pnls),
        "max_drawdown": maximum_drawdown(trades_pnls),
        "sharpe_ratio": sharpe_ratio(trades_pnls),
        "winning_trades_ratio": winning_trades_ratio(trades_pnls),
        "total_trades": len(trades_pnls),
        "trades_pnls": trades_pnls,
    }


def backtest_bnh_strategy(symbol_df, start_date, trade_value=1000):
    buy_price = symbol_df.loc[symbol_df["date"] == start_date, "open"].values[0]
    sell_price = symbol_df.loc[symbol_df.index[-1], "close"]
    trade_shares = trade_value // buy_price
    return {
        "total_pnl": trade_shares * (sell_price - buy_price),
        "max_drawdown": 0,
        "sharpe_ratio": 0,
        "winning_trades_ratio": (buy_price > sell_price) * 1,
        "total_trades": 1,
        "trades_pnls": [trade_shares * (sell_price - buy_price)],
    }


def backtest_rsi_strategy(symbol_df, start_date, trade_value=1000, plot=False):
    trades_pnls = []
    in_trade = False
    position = None
    for i, row in symbol_df.loc[start_date:].iterrows():
        if row["hour"] == 9 and row["minute"] < 50:
            continue  # skip trading in the first 20 mins to let RSI stabilize
        if not in_trade and (row["hour"] < 15 or (row["hour"] == 15 and row["minute"] < 30)):
            if row["RSI"] <= 30:
                in_trade = True
                position = Position(trade_value // row["close"], row["close"])
            elif row["RSI"] >= 70:
                in_trade = True
                position = Position(-trade_value // row["close"], row["close"])
        elif in_trade:
            if position.shares > 0 and row["RSI"] >= 50:
                in_trade = False
                trades_pnls.append((row["close"] - position.entry_price) * position.shares)
            elif position.shares < 0 and row["RSI"] <= 50:
                in_trade = False
                trades_pnls.append((position.entry_price - row["close"]) * position.shares)
            elif row["hour"] == 15 and row["minute"] == 30:  # close position at the end of the day
                in_trade = False
                trades_pnls.append((row["close"] - position.entry_price) * position.shares)

    if plot:
        plot_pnls(trades_pnls)

    return {
        "total_pnl": total_pnl(trades_pnls),
        "max_drawdown": maximum_drawdown(trades_pnls),
        "sharpe_ratio": sharpe_ratio(trades_pnls),
        "winning_trades_ratio": winning_trades_ratio(trades_pnls),
        "total_trades": len(trades_pnls),
        "trades_pnls": trades_pnls,
    }


def backtest_tft_strategy(symbol_df, start_date, trade_value=1000, plot=False, prediction_length=10):
    trades = []
    in_trade = False
    position = None
    for i, row in symbol_df.loc[start_date:].iterrows():
        if not in_trade:
            prediction_quantiles_30 = row["predictions"][:, 0]
            prediction_quantiles_50 = row["predictions"][:, 1]
            prediction_quantiles_70 = row["predictions"][:, 2]
            if prediction_quantiles_30[5:].min() >= row["open"]:
                in_trade = True
                position = Position(
                    trade_value // row["open"],
                    row["open"],
                    stop_loss=min(prediction_quantiles_30.min(), row["open"]-0.05),
                    take_profit=prediction_quantiles_50.max(),
                    ttl=prediction_length,
                    entry_bar=row
                )
            elif prediction_quantiles_70[5:].max() <= row["open"]:
                in_trade = True
                position = Position(
                    -trade_value // row["open"],
                    row["open"],
                    stop_loss=max(prediction_quantiles_70.max(), row["open"]),
                    take_profit=prediction_quantiles_50.min(),
                    ttl=prediction_length,
                    entry_bar=row
                )
        elif in_trade:
            if position.shares > 0:
                if row["high"] >= position.take_profit:
                    in_trade = False
                    trades.append(
                        Trade(
                            entry_price=position.entry_price,
                            exit_price=position.take_profit,
                            shares=position.shares,
                            pnl=(position.take_profit - position.entry_price) * position.shares,
                            is_long=True,
                            entry_bar=position.entry_bar,
                            exit_bar=row
                        )
                    )
                elif row["low"] <= position.stop_loss:
                    in_trade = False
                    trades.append(
                        Trade(
                            entry_price=position.entry_price,
                            exit_price=position.stop_loss,
                            shares=position.shares,
                            pnl=(position.stop_loss - position.entry_price) * position.shares,
                            is_long=True,
                            entry_bar=position.entry_bar,
                            exit_bar=row
                        )
                    )
                elif position.ttl == 1:
                    in_trade = False
                    trades.append(
                        Trade(
                            entry_price=position.entry_price,
                            exit_price=row["close"],
                            shares=position.shares,
                            pnl=(row["close"] - position.entry_price) * position.shares,
                            is_long=True,
                            entry_bar=position.entry_bar,
                            exit_bar=row
                        )
                    )
                if not in_trade:
                    symbol_df.at[row.name, "trade"] = "sell_to_close"
            else:
                if row["low"] <= position.take_profit:
                    in_trade = False
                    trades.append(
                        Trade(
                            entry_price=position.entry_price,
                            exit_price=position.take_profit,
                            shares=position.shares,
                            pnl=(position.entry_price - position.take_profit) * -position.shares,
                            is_long=False,
                            entry_bar=position.entry_bar,
                            exit_bar=row
                        )
                    )
                elif row["high"] >= position.stop_loss:
                    in_trade = False
                    trades.append(
                        Trade(
                            entry_price=position.entry_price,
                            exit_price=position.stop_loss,
                            shares=position.shares,
                            pnl=(position.entry_price - position.stop_loss) * -position.shares,
                            is_long=False,
                            entry_bar=position.entry_bar,
                            exit_bar=row
                        )
                    )
                elif position.ttl == 1:
                    in_trade = False
                    trades.append(
                        Trade(
                            entry_price=position.entry_price,
                            exit_price=row["close"],
                            shares=position.shares,
                            pnl=(position.entry_price - row["close"]) * -position.shares,
                            is_long=False,
                            entry_bar=position.entry_bar,
                            exit_bar=row
                        )
                    )
                if not in_trade:
                    symbol_df.at[row.name, "trade"] = "buy_to_close"

            position.ttl -= 1

    if plot and trades:
        trade_sample_index = np.random.choice(len(trades))
        trade_sample = trades[trade_sample_index]
        candles = symbol_df.loc[symbol_df["date"] == trade_sample.entry_bar.date].copy()
        is_long = trade_sample.is_long
        # clip the plot area to be around the trade instead of the whole day
        candles = candles.loc[trade_sample.entry_bar.name - 20 * Minute(): trade_sample.exit_bar.name + 20 * Minute()]

        entry_price = pd.Series([np.nan] * candles.shape[0], index=[candles.index])
        entry_price[trade_sample.entry_bar.name] = trade_sample.entry_price
        exit_price = pd.Series([np.nan] * candles.shape[0], index=[candles.index])
        exit_price[trade_sample.exit_bar.name] = trade_sample.exit_price
        quantile_30 = trade_sample.entry_bar.predictions[:, 0]
        quantile_50 = trade_sample.entry_bar.predictions[:, 1]
        quantile_50_plot = pd.Series([np.nan] * len(candles), index=[candles.index])
        quantile_50_plot[trade_sample.entry_bar.name: trade_sample.entry_bar.name + 9 * Minute()] = quantile_50
        quantile_70 = trade_sample.entry_bar.predictions[:, 2]

        fig, ax = plt.subplots()
        ax.set_title(f"Trade on {symbol_df.symbol[0]}, pnl: {trades[trade_sample_index].pnl:.2f}")
        addplots = [
            mpf.make_addplot(
                entry_price,
                type='scatter',
                markersize=100,
                marker='v' if not is_long else '^',
                color='red' if not is_long else 'green',
                label='Entry',
                ax=ax
            ),  # entry price
            mpf.make_addplot(
                exit_price,
                type='scatter',
                markersize=100,
                marker='^' if not is_long else 'v',
                color='green' if not is_long else 'red',
                label='Exit',
                ax=ax
            ),  # exit price
            mpf.make_addplot(
                quantile_50_plot,
                color='orange',
                linestyle='dashed',
                ax=ax
            ),  # 50% quantile
        ]

        mpf.plot(
            candles,
            ax=ax,
            type="candle",
            addplot=addplots,
        )
        prediction_start_index = candles.index.get_loc(trade_sample.entry_bar.name)
        ax.fill_between(range(prediction_start_index, prediction_start_index+10), quantile_30, quantile_70, color='orange', alpha=0.2)

        plt.show()

    trades_pnls = [trade.pnl for trade in trades]
    return {
        "total_pnl": total_pnl(trades_pnls),
        "max_drawdown": maximum_drawdown(trades_pnls),
        "sharpe_ratio": sharpe_ratio(trades_pnls),
        "winning_trades_ratio": winning_trades_ratio(trades_pnls),
        "total_trades": len(trades_pnls),
        "trades_pnls": trades_pnls,
    }

def plot_pnls(trades_pnls, title=None):
    cumulative_pnls = pd.Series(trades_pnls).cumsum()

    plt.figure(figsize=(6, 4))
    plt.plot(cumulative_pnls, label='Cumulative PnL')
    plt.title(title or 'Cumulative PnL Over Time')
    plt.xlabel('Trade #')
    plt.ylabel('Cumulative PnL')
    plt.axhline(0, color='red', linestyle='--', label='Break-even')
    plt.legend()
    plt.grid(True)
    plt.show()


def describe_strategy_result(symbol_strategy_results: dict, plot=False):
    results_df = pd.DataFrame.from_dict(symbol_strategy_results, orient='index')
    print(f"Total unique stocks traded: {len(symbol_strategy_results)}")
    print(f"Total money used: {1000 * len(symbol_strategy_results)}")
    print(f"Total PnL across all stocks: {results_df['total_pnl'].sum():.2f}")
    print(f"Total PnL % of money used: {results_df['total_pnl'].sum() / (1000 * len(symbol_strategy_results)):.2%}")
    print(f"Average PnL per stock: {results_df['total_pnl'].mean():.2f}")
    print(f"Max drawdown: {results_df['max_drawdown'].max():.2f}")
    print(f"Average Sharpe ratio: {results_df['sharpe_ratio'].mean():.2f}")
    print(f"Average winning trades ratio: {results_df['winning_trades_ratio'].mean():.2%}")
    print(f"Total trades: {results_df['total_trades'].sum()}")
    best_stock = results_df['total_pnl'].idxmax()
    print(f"Best stock PnL: {results_df['total_pnl'].max():.2f} - {best_stock}")
    best_stock_trades_pnls = symbol_strategy_results[best_stock]["trades_pnls"]
    if len(best_stock_trades_pnls) > 1 and plot:
        plot_pnls(best_stock_trades_pnls, title=f'Best Cumulative PnL Over Trade # - {best_stock}')
    worst_stock = results_df['total_pnl'].idxmin()
    print(f"Worst stock PnL: {results_df['total_pnl'].min():.2f} - {worst_stock}")
    worst_stock_trades_pnls = symbol_strategy_results[worst_stock]["trades_pnls"]
    if len(worst_stock_trades_pnls) > 1 and plot:
        plot_pnls(worst_stock_trades_pnls, title=f'Worst Cumulative PnL Over Trade # - {worst_stock}')
    pnl_per_stock = results_df.groupby(results_df.index)["total_pnl"].sum()
    print(f"PnL Per Stock: {pnl_per_stock.sort_values(ascending=False)}")