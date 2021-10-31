import itertools
from datetime import date, timedelta, datetime
from enum import Enum, unique
from pathlib import Path
from typing import Set, List, Dict, Tuple

import numpy as np
import pandas as pd

import yfinance as yf



class DataRepository2:
    def __init__(self, coint_window_length: int,
                 coint_window_start_date: date):
        self.coint_window_length = coint_window_length
        self.coint_window_start_date = coint_window_start_date
        self.snp_info = self.__get_company_data_from_yfinance()
        #self.__get_price_data_from_yfinance()
        self.all_dates = self.__load_all_available_dates()
        self.idx_coint_start_date, self.idx_coint_end_date, \
        self.idx_trade_start_date, self.idx_trade_end_date, \
        self.idx_no_new_trades_date = self.__get_window_boundaries()
        self.coint_window_end_date = self.all_dates[self.idx_coint_end_date]
        self.trade_window_start_date = self.all_dates[self.idx_trade_start_date]
        self.trade_window_end_date = self.all_dates[self.idx_trade_end_date]
        self.no_new_trades_from_date = self.all_dates[self.idx_no_new_trades_date]
        self.current_price_data = self.__get_price_data_from_local_csv() # might useful to be called outside for understanding
        self.current_cluster_dict = self.__get_current_cluster_dict()
        self.allowed_couples = self.__get_allowed_couples()


    def __load_all_available_dates(self) -> pd.Series:
        all_avail_dates = pd.read_csv(Path(f"../resources/all_snp2.csv"), usecols=[0],
                                      parse_dates=True, dayfirst=True).iloc[2:, :]
        all_avail_dates.columns = ["Date"]
        all_avail_dates = pd.to_datetime(all_avail_dates.Date, dayfirst=True).dt.date
        return all_avail_dates.reset_index(drop=True)

    def __get_window_boundaries(self) -> List[int]:
        idx_coint_start_date = self.all_dates[self.all_dates == self.coint_window_start_date].index[0]
        idx_coint_end_date = idx_coint_start_date + self.coint_window_length - 1
        idx_trade_start_date = idx_coint_end_date + 1
        idx_trade_end_date = idx_trade_start_date + 3 * self.coint_window_length - 1
        idx_no_new_trades_date = idx_trade_end_date - 15
        return [idx_coint_start_date, idx_coint_end_date, idx_trade_start_date, idx_trade_end_date, idx_no_new_trades_date]

    def __get_price_data_from_yfinance(self) -> None:
        start, end = datetime(2008, 1, 2), datetime(2021, 10, 22)
        # s&p500 components price data
        data = yf.download(self.snp_info.ticker.to_list(), start=start, end=end)
        idx = pd.IndexSlice
        data.loc[:, idx["Adj Close", :]].droplevel(0, axis=1).dropna(axis=0, how="all")\
            .to_parquet(Path(f"../resources/closes.parquet"))
        data.loc[:, idx["Volume", :]].droplevel(0, axis=1).dropna(axis=0, how="all").\
            to_parquet(Path(f"../resources/volumes.parquet"))

    def __get_company_data_from_yfinance(self) -> None:
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_info = wiki_table[0].loc[:, ["Symbol", "GICS Sub-Industry"]]
        sp_info.columns = ["ticker", "sector"]
        return sp_info

    def __get_price_data_from_local_csv(self) -> pd.DataFrame:
        if self.idx_coint_start_date == 0:
            snp_price_data = pd.read_csv(Path(f"../resources/closes.csv"), squeeze=True, header=0,
                                              index_col="Date", nrows=4 * self.coint_window_length,
                                              low_memory=False, parse_dates=True, dayfirst=True)
        else:
            snp_price_data = pd.read_csv(Path(f"../resources/closes.csv"), squeeze=True, header=0,
                                         index_col=0, skiprows=range(1, self.idx_coint_start_date + 1),
                                         nrows=4 * self.coint_window_length,
                                         low_memory=False, parse_dates=True, dayfirst=True)

        return snp_price_data.dropna(axis=1, how="any")

    def __get_current_cluster_dict(self) -> Dict:
        existing_tickers = self.current_price_data.columns
        current_cluster_dict = {}
        for ticker in existing_tickers:
            ticker_sector = self.snp_info.loc[self.snp_info["ticker"]==ticker, "sector"].item()
            if current_cluster_dict.get(ticker_sector) is None:
                current_cluster_dict[ticker_sector] = [ticker]
            else:
                current_cluster_dict[ticker_sector].append(ticker)
        return current_cluster_dict

    def __get_allowed_couples(self) -> List:
        couples = []
        for sect, ticker_list in self.current_cluster_dict.items():
            if len(ticker_list) > 1:
                couples += [couple for couple in itertools.combinations(ticker_list, 2)]
        return couples

    def __update_window_boundaries(self):
        self.idx_coint_start_date = self.idx_trade_end_date + 1
        self.idx_coint_end_date = self.idx_coint_start_date + self.coint_window_length - 1
        self.idx_trade_start_date = self.idx_coint_end_date + 1
        self.idx_trade_end_date = self.idx_trade_start_date + 3 * self.coint_window_length - 1
        self.idx_no_new_trades_date = self.idx_trade_end_date - 15

    def __update_key_dates(self):
        try:
            self.coint_window_start_date = self.all_dates[self.idx_coint_start_date]
            self.coint_window_end_date = self.all_dates[self.idx_coint_end_date]
            self.trade_window_start_date = self.all_dates[self.idx_trade_start_date]
            self.trade_window_end_date = self.all_dates[self.idx_trade_end_date]
            self.no_new_trades_from_date = self.all_dates[self.idx_no_new_trades_date]
        except:
            print(self.all_dates.iloc[-1], "end of bt", self.idx_coint_start_date,
                  self.idx_coint_end_date, self.idx_trade_start_date, self.idx_trade_end_date)

    def __update_price_data_and_cluster(self):
        self.current_price_data = self.__get_price_data_from_local_csv() # might useful to be called outside for understanding
        self.current_cluster_dict = self.__get_current_cluster_dict()
        self.allowed_couples = self.__get_allowed_couples()

    def update(self):
        self.__update_window_boundaries()
        self.__update_key_dates()
        self.__update_price_data_and_cluster()