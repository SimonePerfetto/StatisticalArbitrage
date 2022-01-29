from datetime import date
import pandas as pd
from src.DataRepository import SPXDataRepository
from src.Cointegrator import Cointegrator, CointPair
from src.Portfolio import Portfolio
from src.DateManager import DateManager
from typing import List, Union
import matplotlib.pyplot as plt


class PairTrader:
    def __init__(self, coint_window: int, roll_stats_window: int, trade_window: int,
                 backtest_start_date: date, max_active_pairs: int, num_std_away: float):
        self.roll_stats_window = roll_stats_window
        self.max_active_pairs = max_active_pairs
        self.num_std_away = num_std_away
        self.date_manager: DateManager = DateManager(backtest_start_date, coint_window, trade_window)
        self.data_repository: SPXDataRepository = self.init_data_repository()
        self.final_backtest_date =  self.date_manager.get_backtest_end_date()
        self.today = self.date_manager.get_today()
        self.coint_pairs: Union[List[CointPair], None] = None
        self.portfolio: Union[Portfolio, None] = None
        self.total_pnl_dict = {}
        self.n_bad_trades_dict = {}
        self.n_good_trades_dict = {}

    def init_data_repository(self) -> SPXDataRepository:
        data_repos = SPXDataRepository(file_name="closes.csv")
        coint_start, coint_end = self.date_manager.coint_start_date, self.date_manager.coint_end_date
        data_repos.update_train_data(coint_start, coint_end)
        return data_repos

    def get_coint_pairs(self) -> List[CointPair]:
        cointegrator = Cointegrator(self.roll_stats_window, self.num_std_away)
        coint_start, coint_end = self.date_manager.coint_start_date, self.date_manager.coint_end_date
        cointegrated_pairs = cointegrator.create_cointegrated_pairs(self.data_repository, coint_start, coint_end)
        return cointegrated_pairs

    def get_portfolio(self) -> Portfolio:
        return Portfolio(max_active_pairs=self.max_active_pairs)

    def update_pair_signals(self, today) -> None:
        for cointpair in self.coint_pairs:
            cointpair.update_signal(today, self.date_manager.no_new_trades_from_date,
                                    self.date_manager.trade_end_date)

    def init(self) -> None:
        self.coint_pairs: List[CointPair] = self.get_coint_pairs()
        self.portfolio: Portfolio = self.get_portfolio()
        self.today = self.date_manager.get_today()

    def trade(self) -> None:
        while self.today < self.final_backtest_date:
            if self.today > self.date_manager.trade_end_date:
                self.date_manager.update_key_dates()
                self.data_repository.update_train_data(start=self.date_manager.coint_start_date,
                                                       end=self.date_manager.coint_end_date)
                self.coint_pairs = self.get_coint_pairs()
            self.update_pair_signals(self.today)
            self.portfolio.rebalance(self.today, self.coint_pairs)
            print(f"{self.today}: {self.portfolio}")
            self.total_pnl_dict[self.today] = self.portfolio.total_pnl
            self.n_bad_trades_dict[self.today] = self.portfolio.n_bad_trades
            self.n_good_trades_dict[self.today] = self.portfolio.n_good_trades
            self.today = self.date_manager.go_to_next_day(self.today)

        pd.Series(self.total_pnl_dict).plot()
        plt.show()


if __name__ == '__main__':
    pairtrader = PairTrader(coint_window=240, roll_stats_window=240, trade_window=180,
                            backtest_start_date=date(2008, 1, 2), max_active_pairs=20, num_std_away=3)
    pairtrader.init()
    pairtrader.trade()
