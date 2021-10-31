from datetime import date, timedelta, datetime
import pandas as pd
from src.DataRepository2 import DataRepository2
from src.Cointegrator2 import Cointegrator2, CointPair
from src.Portfolio2 import Portfolio2
from typing import  List

class PairTrader2:
    # cointegration check done only once every full window, then we trade on old info
    def __init__(self,
                 coint_window_length: int,
                 coint_window_start_date: date,
                 max_active_pairs: int):
        #self.n_coint_pairs = n_coint_pairs
        self.max_active_pairs = max_active_pairs
        self.data_repository = self.__get_data_repository(coint_window_length, coint_window_start_date)
        self.final_backtest_date =  self.__get_final_backtest_date()
        self.today = None
        self.coint_pairs = None
        self.portfolio = None



    def __get_data_repository(self, coint_window_length, coint_window_start_date) -> DataRepository2:
        # get initial coint window data
        return DataRepository2(coint_window_length, coint_window_start_date)

    def __get_todays_date(self):
        return self.data_repository.trade_window_start_date

    def __get_final_backtest_date(self):
        return self.data_repository.all_dates.values[-1]

    def __get_coint_pairs(self) -> List[CointPair]:
        coint_obj = Cointegrator2(self.data_repository)
        return coint_obj.cointegrated_pairs

    def __get_portfolio(self) -> Portfolio2:
        return Portfolio2()

    def update_pair_signals(self) -> None:
        for cointpair in self.coint_pairs:
            no_new_trades_from_date = self.data_repository.no_new_trades_from_date
            trade_window_end_date = self.data_repository.trade_window_end_date
            cointpair.update_signal(self.today, no_new_trades_from_date, trade_window_end_date)

    def rebalance_portfolio(self):
        pass

    def go_to_next_day(self):
        todays_idx = self.data_repository.all_dates[self.data_repository.all_dates==self.today].index[0]
        tomorrows_idx = todays_idx + 1
        self.today = self.data_repository.all_dates[tomorrows_idx]

    def init(self):
        self.coint_pairs = self.__get_coint_pairs()
        self.portfolio = self.__get_portfolio()
        self.today = self.__get_todays_date()


    def trade(self):
        while self.today < self.final_backtest_date:
            print(f"Today is {self.today}.")
            if self.today > self.data_repository.trade_window_end_date:
                self.data_repository.update()
                self.coint_pairs = self.__get_coint_pairs()
                # cointegration already done at initialization... we'll see later when it has to be done again...
            #print("Updating signals for each cointegrated pair...")
            self.update_pair_signals()
            print("Rebalancing Portfolio based on new signals...")
            self.rebalance_portfolio()
            self.go_to_next_day()


        pass
    pass

if __name__ == '__main__':
    pairtrader = PairTrader2(coint_window_length=60, coint_window_start_date=date(2008, 1, 2),
                             max_active_pairs=10)
    pairtrader.init()
    pairtrader.trade()
