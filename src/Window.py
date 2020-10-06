import sys
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd
from pandas import DataFrame

from src.DataRepository import DataRepository, Universes
from src.util.Features import Features
from src.util.Tickers import Tickers


class Window:

    def __init__(self,
                 window_start: date,
                 trading_win_len: timedelta,
                 repository: DataRepository):

        # Window object contains information about timings for the window as well as SNP and ETF data for that period.
        # After construction of the object we also have self.etf_data nd self.snp_data and the live tickers for each
        self.window_start: date = window_start
        self.window_length: timedelta = trading_win_len
        self.repository: DataRepository = repository

        self.window_end = self.__get_nth_working_day_ahead(window_start, self.window_length.days - 1)
        self.window_trading_days: pd.Series = self.__get_window_trading_days(window_start, self.window_end)
        self.__update_window_data(self.window_trading_days)


    def __get_window_trading_days(self, window_start: date, window_end: date):
        window_trading_days = self.repository.all_dates[window_start:window_end]
        return window_trading_days

    def __update_window_data(self, trading_dates_to_get_data_for: pd.Series):
        if self.repository.all_data is None:
            # ie need to load the new window from disk for first time
            self.repository.get(Universes.SNP, trading_dates_to_get_data_for)

        next_load_start_date = max(self.repository.all_data.index)
        if self.window_end > next_load_start_date:
            # i.e., need to load the new window from disk
            read_ahead_win_start = self.__get_nth_working_day_ahead(next_load_start_date,
                                                                    self.window_length.days - 1)
            look_forward_win_dates = self.__get_window_trading_days(next_load_start_date,
                                                                    read_ahead_win_start)

            self.repository.get(Universes.SNP, look_forward_win_dates)

        trading_dates_to_get_data_for = self.repository.all_data.index.intersection(trading_dates_to_get_data_for)
        lookback_temp_snp_data = self.repository.all_data.loc[trading_dates_to_get_data_for]

        _, self.snp_data = self.repository.remove_dead_tickers(lookback_temp_snp_data)

    def roll_forward_one_day(self) -> None:

        self.window_start = self.__get_nth_working_day_ahead(self.window_start, 1)
        self.window_end = self.__get_nth_working_day_ahead(self.window_end, 1)

        self.window_trading_days = self.__get_window_trading_days(self.window_start, self.window_end)
        # last window trading date should be today + 1 because today gets updated after this function gets called


    def __get_nth_working_day_ahead(self, starting_date: date, n: int):
        for idx, d in enumerate(self.repository.all_dates):
            if d == starting_date:
                return self.repository.all_dates[idx + n]

        print("The window start date was not in the list of all dates.")
        print("Ensure backtest is started with a day that is in the datset.")

    def get_data(self,
                 tickers: Optional[List[Tuple[Tickers]]] = None,
                 features: Optional[List[Features]] = None) -> DataFrame:
        """
        function to get data, with tickers and features specified
        universe: Universe.SNP
        tickers: a list of Tickers or None, if None, return all tickers
        features: a list of Features or None, if None, return all features
        Note it takes lists of Tickers and Features but must be called with:
            - lists of SnpTickers and SnpFeatures
        """
        if tickers is None and features is None:   # both none
            return self.snp_data
        if tickers is not None and features is None:  # tickers not None
            return self.snp_data.loc[:, pd.IndexSlice[tickers, :]]
        elif tickers is None and features is not None:  # features not None
            return self.snp_data.loc[:, pd.IndexSlice[:, features]]
        elif tickers is not None and features is not None:  # both not None
            return self.snp_data.loc[:, pd.IndexSlice[tickers, features]]

    def get_fundamental(self):
        return self.repository.get_fundamental(self.window_end)
