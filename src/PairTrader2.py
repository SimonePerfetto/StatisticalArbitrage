from datetime import date, timedelta, datetime
import pandas as pd
from src.DataRepository2 import DataRepository2
from src.Cointegrator2 import Cointegrator2, CointPair
from src.Portfolio2 import Portfolio2
from src.Window2 import Window2
from typing import  List

class PairTrader2:
    # cointegration check done only once every full window, then we trade on old info
    def __init__(self,
                 coint_window_length: int,
                 backtest_start_date: date,
                 max_active_pairs: int):
        #self.n_coint_pairs = n_coint_pairs
        self.max_active_pairs = max_active_pairs
        self.data_repository: DataRepository2 = self.get_data_repository(backtest_start_date, coint_window_length)
        self.final_backtest_date =  self.data_repository.window.get_backtest_end_date()
        self.today = self.data_repository.window.get_today()
        self.coint_pairs = None
        self.portfolio = None

    def get_data_repository(self, backtest_start_date, coint_window_length) -> DataRepository2:
        window = Window2(backtest_start_date, coint_window_length)
        return DataRepository2(window)

    def get_coint_pairs(self) -> List[CointPair]:
        cointegrator = Cointegrator2(self.data_repository)
        return cointegrator.cointegrated_pairs

    def get_portfolio(self) -> Portfolio2:
        return Portfolio2()

    def update_pair_signals(self) -> None:
        for cointpair in self.coint_pairs:
            cointpair.update_signal(self.today, self.data_repository.window.no_new_trades_from_date,
                                    self.data_repository.window.trade_window_end_date)

    def rebalance_portfolio(self):
        pass


    def init(self):
        self.coint_pairs = self.get_coint_pairs()
        self.portfolio = self.get_portfolio()
        self.today = self.data_repository.window.get_today()


    def trade(self):
        while self.today < self.final_backtest_date:
            print(f"Today is {self.today}.")
            if self.today > self.data_repository.window.trade_window_end_date:
                self.data_repository.window.update_key_dates()
                self.data_repository.update_data()
                self.coint_pairs = self.get_coint_pairs()
            self.update_pair_signals()
            self.rebalance_portfolio()
            self.today = self.data_repository.window.go_to_next_day(self.today)

        pass
    pass

if __name__ == '__main__':
    pairtrader = PairTrader2(coint_window_length=60, backtest_start_date=date(2008, 1, 2),
                             max_active_pairs=10)
    pairtrader.init()
    pairtrader.trade()
