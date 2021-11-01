from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from pandas import DataFrame, to_datetime
from src.Performance import get_performance_stats
from src.Position import Position
from src.util.Features import Features, PositionType
from typing import List, Tuple
from src.Cointegrator2 import CointPair


class PairHolding:
    def __init__(self, ticker_l: str, ticker_s: str, pl: float, ps: float, nl: float, ns: float):
        self.ticker_l: str = ticker_l
        self.ticker_s: str = ticker_s
        self.pl: float = pl
        self.ps: float = ps
        self.nl: float = nl
        self.ns: float = ns

    def __repr__(self):
        return f"PairHolding[Long({self.ticker_l}), Short({self.ticker_s})]"


class Portfolio2:

    def __init__(self, cash: float = 1_000_000.0):
        self.cur_cash: float = cash
        self.capital_invested: float = 0
        self.cur_holding = None
        self.realised_pnl: float = 0
        self.outstanding_pnl: float = 0
        self.t_cost: float = 0.0005

    def rebalance(self, coint_pairs, today):
        # this one want to change the holdings (long/short). Maybe useful to create holding object.
        # maybe call here something like __examine single pair
        for coint_pair in coint_pairs:
            trade_action = self.evaluate_trade_action(coint_pair)


    def evaluate_trade_action(self, coint_pair):
        prev_sign, curr_sign = coint_pair.previous_pair_signal, coint_pair.current_pair_signal
        if curr_sign == 1 and prev_sign == 0: return "OpenLong"
        elif curr_sign == -1 and prev_sign == 0: return "OpenShort"
        elif curr_sign == 0 and prev_sign == -1: return "CloseShort"
        elif curr_sign == 0 and prev_sign == 1: return "CloseLong"
        elif curr_sign == 0 and prev_sign == 0: return "Pass"
        else: return "Hold"

    def make_trade_action(self, coint_pair, today):
        if coint_pair.current_pair_signal == 1 and coint_pair.previous_pair_signal == 0:
            # long pair means buy 1 unit of y sell hedgeratio units of x
            hedge_ratio = coint_pair.hedge_ratio
            px, py = self.__get_todays_price_x_y(coint_pair, today)
            nx, ny = self.__units_finder(py, coint_pair.hedge_ratio)
            ticker_x, ticker_y = self.__get_ticker_x_y(coint_pair)
            pair_holding = PairHolding(ticker_y, ticker_x, py, px, ny, nx)
            a=10

    @staticmethod
    def __get_todays_price_x_y(coint_pair: CointPair, today: date) -> Tuple:
        px = coint_pair.stock_x.window_prices.loc[pd.to_datetime(today)]
        py = coint_pair.stock_y.window_prices.loc[pd.to_datetime(today)]
        return px, py

    @staticmethod
    def __get_ticker_x_y(coint_pair: CointPair) -> Tuple:
        return coint_pair.stock_x.ticker, coint_pair.stock_y.ticker

    @staticmethod
    def __units_finder(py, hedge_ratio, min_notional=48_000, max_notional=52_000):
        notionals = np.linspace(min_notional, max_notional, 101)
        errors = []
        for notional in notionals:
            ny = round(notional/py)
            nx = round(ny * hedge_ratio)
            abs_err = abs(nx/ny - hedge_ratio)
            errors.append(abs_err)
        idx_min_error = np.argmin(errors)
        notional_w_min_err = notionals[idx_min_error]
        ny_star = round(notional_w_min_err / py)
        nx_star = round(ny_star * hedge_ratio)
        return nx_star, ny_star


    def open_position(self, position: Position):
        pass
        # cur_price = self.current_window.get_data(tickers=[position.asset1, position.asset2],
        #                                          features=[Features.CLOSE])
        # # notional reference amount for each pair. Actual positions are scaled accordingly with respect to
        # # maximum weight as per below formula
        # pair_dedicated_cash = self.init_cash * self.single_pair_loading / max(abs(position.weight1), abs(position.weight2))
        # position.quantity1 = int(pair_dedicated_cash * position.weight1 / cur_price.iloc[-1, 0])
        # position.quantity2 = int(pair_dedicated_cash * position.weight2 / cur_price.iloc[-1, 1])
        # asset1_value = cur_price.iloc[-1, 0] * position.quantity1
        # asset2_value = cur_price.iloc[-1, 1] * position.quantity2
        # commission = self.generate_commission(asset1_value, asset2_value)
        # pair_dedicated_cash = asset1_value + asset2_value
        # position.set_position_value(pair_dedicated_cash)
        # if pair_dedicated_cash > self.cur_cash:
        #     print('No sufficient cash to open position')
        # else:
        #     print(f"{position.asset1}, {position.asset2} "
        #           f"are cointegrated and zscore is in trading range. Opening position....")
        #     self.cur_positions.append(position)
        #     self.hist_positions.append(position)
        #
        #     self.cur_cash -= (pair_dedicated_cash + commission)
        #     self.active_port_value += pair_dedicated_cash
        #     print(f'Asset 1: {position.asset1} @${round(cur_price.iloc[-1, 0], 2)} '
        #           f'Quantity: {round(position.quantity1, 2)} Value: {round(asset1_value, 2)}')
        #     print(f'Asset 2: {position.asset2} @${round(cur_price.iloc[-1, 1], 2)} '
        #           f'Quantity: {round(position.quantity2, 2)} Value: {round(asset2_value, 2)}')
        #     print(f'Cash balance: ${self.cur_cash}')

    def close_position(self, position: Position):
        pass
        # cur_price = self.current_window.get_data(tickers=[position.asset1, position.asset2],
        #                                          features=[Features.CLOSE])
        # if not (position in self.cur_positions):
        #     print("do not have this position open")
        # else:
        #     print(f"{position.closingtype} threshold is passed for active pair {position.asset1}, "
        #           f"{position.asset2}. Closing position...")
        #     self.cur_positions.remove(position)
        #
        #     asset1_value = cur_price.iloc[-1, 0] * position.quantity1
        #     asset2_value = cur_price.iloc[-1, 1] * position.quantity2
        #     commission = self.generate_commission(asset1_value, asset2_value)
        #     pair_residual_cash = asset1_value + asset2_value
        #
        #     position.close_trade(pair_residual_cash, self.current_window)
        #     self.cur_cash += pair_residual_cash - commission
        #     self.active_port_value -= pair_residual_cash
        #     self.realised_pnl += position.pnl
        #     print(f'Asset 1: {position.asset1} @${round(cur_price.iloc[-1, 0], 2)}'
        #           f' Quantity: {int(position.quantity1)}')
        #     print(f'Asset 2: {position.asset2} @${round(cur_price.iloc[-1, 1], 2)} '
        #           f'Quantity: {int(position.quantity2)}')
        #     print(f'Realised PnL for position: {round(position.pnl, 2)}')

    def generate_commission(self, asset1_value, asset2_value):
        pass
        # transaction costs as % of notional amount
        #return self.t_cost * (abs(asset1_value) + abs(asset2_value))

    def update_portfolio(self, today: date):
        pass
        # cur_port_val = 0
        #
        # for pair in self.cur_positions:
        #     todays_prices = self.current_window.get_data(tickers=[pair.asset1, pair.asset2],
        #                                                  features=[Features.CLOSE]).loc[today]
        #
        #     asset_value = todays_prices[0] * pair.quantity1 + todays_prices[1] * pair.quantity2
        #     pair.update_position_pnl(asset_value, self.current_window)
        #     cur_port_val += asset_value
        #
        # # Compute portfolio stats
        # self.active_port_value = cur_port_val
        # self.total_capital.append(self.cur_cash + self.active_port_value)
        # self.daily_return = self.total_capital[-1]/self.total_capital[-2] - 1
        # self.cum_return = self.total_capital[-1]/self.total_capital[0] - 1
        # self.port_hist.append([self.current_window.window_end, self.cur_cash, self.active_port_value,
        #                        self.cur_cash + self.active_port_value, self.realised_pnl, self.daily_return * 100,
        #                        self.cum_return * 100])
        # print(f"Total Capital: {self.total_capital[-1]:.4f}\tCum Return: {self.cum_return:4f}")

    def execute_trades(self, trades_to_execute_list):
        pass
        # for position in trades_to_execute_list:
        #     if position.new_pos is PositionType.NOT_INVESTED:
        #         self.close_position(position)
        #     else:
        #         self.open_position(position)


    def get_port_summary(self):
        pass
        # data = list()
        # for pair in self.cur_positions:
        #     data.append([pair.asset1, pair.quantity1, pair.asset2, pair.quantity2, pair.pnl])
        #
        # df = DataFrame(data, columns=['Asset 1', 'Quantity 1', 'Asset 2', 'Quantity 2', 'PnL'])
        #
        # print('------------------Portfolio Summary------------------')
        # print('Current cash balance: \n %s' % self.cur_cash)
        # if len(data) != 0:
        #     print('Current Positions: ')
        #     print(df)
        # else:
        #     print('No Current Positions')
        # print('Realised PnL: \n %s' % self.realised_pnl)
        # print('-----------------------------------------------------')
        # return [self.cur_cash, df, self.realised_pnl]




