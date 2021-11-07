from datetime import date
import pandas as pd
from src.DataRepository import DataRepository
from src.Cointegrator import Cointegrator, CointPair
from src.Portfolio import Portfolio
from src.Window import Window
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

class PairTrader:
    def __init__(self, coint_window_length: int, backtest_start_date: date, max_active_pairs: int):
        self.max_active_pairs = max_active_pairs
        self.data_repository: DataRepository = self.get_data_repository(backtest_start_date, coint_window_length)
        self.final_backtest_date =  self.data_repository.window.get_backtest_end_date()
        self.today = self.data_repository.window.get_today()
        self.coint_pairs = None
        self.portfolio = None
        self.total_pnl_dict: dict = {}

    @staticmethod
    def get_data_repository(backtest_start_date, coint_window_length) -> DataRepository:
        window = Window(backtest_start_date, coint_window_length)
        return DataRepository(window)

    def get_coint_pairs(self) -> List[CointPair]:
        cointegrator = Cointegrator(self.data_repository)
        return cointegrator.cointegrated_pairs

    def get_portfolio(self) -> Portfolio:
        return Portfolio(max_active_pairs=self.max_active_pairs)

    def update_pair_signals(self) -> None:
        for cointpair in self.coint_pairs:
            cointpair.update_signal(self.today, self.data_repository.window.no_new_trades_from_date,
                                    self.data_repository.window.trade_window_end_date)

    def compute_sharpe_ratio(self):
        rates = pd.read_csv(Path(f"../data/DTB3.csv"), parse_dates=["Date"], dayfirst=True).set_index("Date")
        returns = pd.Series(self.total_pnl_dict).pct_change().dropna()
        avg_y_return = returns.mean() * 252
        std_y = np.std(returns) * 252 ** 0.5
        avg_rf_return = rates.mean() * 360  # days-per-year convention for risk-free rate benchmark
        print(f"Sharpe Ratio: {(avg_y_return - avg_rf_return)/std_y}")


    def init(self) -> None:
        self.coint_pairs = self.get_coint_pairs()
        self.portfolio = self.get_portfolio()
        self.today = self.data_repository.window.get_today()

    def trade(self) -> None:
        while self.today < self.final_backtest_date:
            if self.today > self.data_repository.window.trade_window_end_date:
                self.data_repository.window.update_key_dates()
                self.data_repository.update_data()
                self.coint_pairs = self.get_coint_pairs()

            self.update_pair_signals()
            self.portfolio.rebalance(self.coint_pairs, self.today)
            print(f"{self.today}: {self.portfolio}")
            self.total_pnl_dict[self.today] = self.portfolio.total_pnl
            self.today = self.data_repository.window.go_to_next_day(self.today)

        pd.Series(self.total_pnl_dict).plot()
        plt.show()
        self.compute_sharpe_ratio()


if __name__ == '__main__':
    pairtrader = PairTrader(coint_window_length=60, backtest_start_date=date(2008, 1, 2), max_active_pairs=10)
    pairtrader.init()
    pairtrader.trade()
