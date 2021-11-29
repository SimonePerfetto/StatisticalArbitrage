from datetime import date
from typing import Union
import pandas as pd
from pathlib import Path


class Window:

    def __init__(self, backtest_start_date: date,  coint_window_length: int, trading_window_length: int):
        self.backtest_start_date: date = backtest_start_date
        self.coint_window_length: int = coint_window_length
        self.trade_window_length: int = trading_window_length
        self.all_dates: pd.Series = self.__load_all_available_dates()
        self.backtest_end_date: date = self.get_backtest_end_date()
        self.no_new_trades_days: int = 15  # might become parameter later
        self.coint_window_start_date: Union[date, None] = None
        self.coint_window_end_date: Union[date, None] = None
        self.trade_window_start_date: Union[date, None] = None
        self.trade_window_end_date: Union[date, None] = None
        self.no_new_trades_from_date: Union[date, None] = None
        self.__define_key_dates()

    @staticmethod
    def __load_all_available_dates() -> pd.Series:
        all_avail_dates: pd.DataFrame = pd.read_csv(Path(f"../data/closes.csv"), usecols=[0],
                                                    parse_dates=True, dayfirst=True)
        all_avail_dates: pd.Series = pd.to_datetime(all_avail_dates.Date, dayfirst=True).dt.date
        return all_avail_dates

    def add_date(self, dt: date, n_days: int) -> date:
        start_dt_idx = self.all_dates[self.all_dates == dt].index[0]
        target_dt_idx = start_dt_idx + n_days
        return self.all_dates[target_dt_idx]

    def __define_key_dates(self) -> None:
        self.coint_window_start_date = self.backtest_start_date
        self.coint_window_end_date = self.add_date(self.coint_window_start_date, self.coint_window_length - 1)
        self.trade_window_start_date = self.add_date(self.coint_window_end_date, 1)
        self.trade_window_end_date = self.add_date(self.trade_window_start_date, self.trade_window_length - 1)
        self.no_new_trades_from_date = self.add_date(self.trade_window_end_date, -self.no_new_trades_days)

    def update_key_dates(self) -> None:
        #TODO: need to parametrize the below
        days_left = (self.backtest_end_date - self.add_date(self.trade_window_end_date, 1)).days
        if days_left >= self.trade_window_length * 1.5 - 1:  # 1.5 times trading window,
            self.coint_window_start_date = self.add_date(self.trade_window_end_date, -self.coint_window_length + 1)
            self.coint_window_end_date = self.trade_window_end_date
            self.trade_window_start_date = self.add_date(self.trade_window_end_date, 1)
            self.trade_window_end_date = self.add_date(self.trade_window_start_date, self.trade_window_length - 1)
            self.no_new_trades_from_date = self.add_date(self.trade_window_end_date, -self.no_new_trades_days)
        else:
            self.trade_window_end_date = self.backtest_end_date
            self.no_new_trades_from_date = self.add_date(self.trade_window_end_date, -self.no_new_trades_days)

    def get_today(self) -> date:
        return self.trade_window_start_date

    def get_backtest_end_date(self) -> date:
        # return datetime(2010,7,16).date()
        return self.all_dates.iloc[-1]

    def go_to_next_day(self, today) -> date:
        return self.add_date(today, 1)
