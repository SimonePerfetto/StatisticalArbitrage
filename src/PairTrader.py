from datetime import date
import pandas as pd
from src.DataRepository import SPXDataRepository
from src.Cointegrator import Cointegrator, CointPair
from src.Portfolio import Portfolio
from src.DateManager import DateManager
from typing import List, Union
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

class PairTrader:
    def __init__(self, coint_window_length: int, roll_stats_window: int, trade_window_length: int,
                 backtest_start_date: date, max_active_pairs: int, num_std_away: float):
        self.roll_stats_window = roll_stats_window
        self.max_active_pairs = max_active_pairs
        self.num_std_away = num_std_away
        self.data_repository: SPXDataRepository = \
            self.get_data_repository(backtest_start_date, coint_window_length, trade_window_length)
        self.final_backtest_date =  self.data_repository.date_manager.get_backtest_end_date()
        self.today = self.data_repository.date_manager.get_today()
        self.coint_pairs: Union[None, List[CointPair]] = None
        self.portfolio: Union[None, Portfolio] = None
        self.total_pnl_dict: dict = {}
        self.n_bad_trades_dict: dict = {}
        self.n_good_trades_dict: dict = {}

    @staticmethod
    def get_data_repository(backtest_start_date, coint_window_length, trade_window_length) -> SPXDataRepository:
        window = DateManager(backtest_start_date, coint_window_length, trade_window_length)
        return SPXDataRepository(window, "closes.csv")

    def get_coint_pairs(self) -> List[CointPair]:
        cointegrator = Cointegrator(self.data_repository, self.roll_stats_window, self.num_std_away)
        return cointegrator.cointegrated_pairs

    def get_portfolio(self) -> Portfolio:
        return Portfolio(max_active_pairs=self.max_active_pairs)

    def update_pair_signals(self, today) -> None:
        for cointpair in self.coint_pairs:
            cointpair.update_signal(today, self.data_repository.date_manager.no_new_trades_from_date,
                                    self.data_repository.date_manager.trade_end_date)


    def init(self) -> None:
        self.coint_pairs: List[CointPair] = self.get_coint_pairs()
        self.portfolio: Portfolio = self.get_portfolio()
        self.today = self.data_repository.date_manager.get_today()

    def trade(self) -> None:
        while self.today < self.final_backtest_date:
            if self.today > self.data_repository.date_manager.trade_end_date:
                self.data_repository.date_manager.update_key_dates()
                self.data_repository.update_data()
                self.coint_pairs = self.get_coint_pairs()
            self.update_pair_signals(self.today)
            self.portfolio.rebalance(self.today, self.coint_pairs)
            print(f"{self.today}: {self.portfolio}")
            self.total_pnl_dict[self.today] = self.portfolio.total_pnl
            self.n_bad_trades_dict[self.today] = self.portfolio.n_bad_trades
            self.n_good_trades_dict[self.today] = self.portfolio.n_good_trades
            self.today = self.data_repository.date_manager.go_to_next_day(self.today)

        pd.Series(self.total_pnl_dict).plot()
        plt.show()


if __name__ == '__main__':
    pairtrader = PairTrader(coint_window_length=240, roll_stats_window=240, trade_window_length=180,
                            backtest_start_date=date(2008, 1, 2), max_active_pairs=20, num_std_away=3)
    pairtrader.init()
    pairtrader.trade()
