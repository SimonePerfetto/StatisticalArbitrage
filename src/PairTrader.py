from datetime import date
import pandas as pd
from src.DataRepository import DataRepository
from src.Cointegrator import Cointegrator, CointPair
from src.Portfolio import Portfolio
from src.Window import Window
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
        self.data_repository: DataRepository = \
            self.get_data_repository(backtest_start_date, coint_window_length, trade_window_length)
        self.final_backtest_date =  self.data_repository.window.get_backtest_end_date()
        self.today = self.data_repository.window.get_today()
        self.coint_pairs: Union[None, List[CointPair]] = None
        self.portfolio: Union[None, Portfolio] = None
        self.total_pnl_dict: dict = {}
        self.n_bad_trades_dict: dict = {}
        self.n_good_trades_dict: dict = {}

    @staticmethod
    def get_data_repository(backtest_start_date, coint_window_length, trade_window_length) -> DataRepository:
        window = Window(backtest_start_date, coint_window_length, trade_window_length)
        return DataRepository(window)

    def get_coint_pairs(self) -> List[CointPair]:
        cointegrator = Cointegrator(self.data_repository, self.roll_stats_window, self.num_std_away)
        return cointegrator.cointegrated_pairs

    def get_portfolio(self) -> Portfolio:
        return Portfolio(max_active_pairs=self.max_active_pairs)

    def update_pair_signals(self, today) -> None:
        for cointpair in self.coint_pairs:
            cointpair.update_signal(today, self.data_repository.window.no_new_trades_from_date,
                                    self.data_repository.window.trade_window_end_date)


    #TODO: not implemented yet
    def compute_sharpe_ratio(self):
        rates = pd.read_csv(Path(f"../data/DTB3.csv"), parse_dates=["Date"], dayfirst=True).set_index("Date")
        #returns = pd.Series(self.total_pnl_dict).pct_change().iloc[2:]  # lazy, need to fix
        #avg_y_return = returns.mean() * 252
        #std_y = np.std(returns) * 252 ** 0.5
        #avg_rf_return = rates.mean() * 360  # days-per-year convention for risk-free rate benchmark
        #print(f"Sharpe Ratio: {(avg_y_return - avg_rf_return)/std_y}")


    def init(self) -> None:
        self.coint_pairs: List[CointPair] = self.get_coint_pairs()
        self.portfolio: Portfolio = self.get_portfolio()
        self.today = self.data_repository.window.get_today()

    def trade(self) -> None:
        while self.today < self.final_backtest_date:
            if self.today > self.data_repository.window.trade_window_end_date:
                self.data_repository.window.update_key_dates()
                self.data_repository.update_data()
                self.coint_pairs = self.get_coint_pairs()
            self.update_pair_signals(self.today)
            self.portfolio.rebalance(self.today, self.coint_pairs)
            print(f"{self.today}: {self.portfolio}")
            self.total_pnl_dict[self.today] = self.portfolio.total_pnl
            self.n_bad_trades_dict[self.today] = self.portfolio.n_bad_trades
            self.n_good_trades_dict[self.today] = self.portfolio.n_good_trades
            self.today = self.data_repository.window.go_to_next_day(self.today)

        pd.Series(self.total_pnl_dict).plot()
        plt.show()


if __name__ == '__main__':
    #pairtrader = PairTrader(coint_window_length=60, roll_stats_window=20, trade_window_length=180, backtest_start_date=date(2008, 1, 2), max_active_pairs=10, num_std_away=2)
    pairtrader = PairTrader(coint_window_length=60, roll_stats_window=20, trade_window_length=180, backtest_start_date=date(2008, 1, 2), max_active_pairs=10, num_std_away=1.8)
    pairtrader.init()
    pairtrader.trade()
