from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from pandas import DataFrame, to_datetime
from src.Performance import get_performance_stats
from src.Position import Position
from src.util.Features import Features, PositionType
from typing import List

class Portfolio2:

    def __init__(self, cash: float = 100_000.0):
        self.cur_cash: float = cash
        self.capital_invested: float = 0
        self.cur_positions: List[Position]
        self.realised_pnl: float = 0
        self.outstanding_pnl: float = 0
        self.t_cost: float = 0.0005
        #self.timestamp = datetime.now().strftime("%Y%m%d%H%M")


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

    # noinspection PyTypeChecker
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




