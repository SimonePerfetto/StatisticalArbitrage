import itertools
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import dask.dataframe as dd
import pandas as pd
from src.Window import Window
import yfinance as yf



class DataRepository:
    def __init__(self, window: Window):
        self.window = window
        self.snp_info = self.__get_company_data_from_yfinance()
        #self.__get_price_data_from_yfinance()
        self.lazy_dask_price_data = self.__get_lazy_dask_price_data_from_csv()
        self.current_price_data = self.__get_current_price_data_from_dask_df() # might useful to be called outside for understanding
        self.current_cluster_dict = self.__get_current_cluster_dict()
        self.allowed_couples = self.__get_allowed_couples()

    def __get_company_data_from_yfinance(self) -> pd.DataFrame:
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_info = wiki_table[0].loc[:, ["Symbol", "GICS Sub-Industry"]]
        sp_info.columns = ["ticker", "sector"]
        return sp_info

    def __get_price_data_from_yfinance(self) -> None:
        start, end = datetime(2008, 1, 2), datetime(2021, 10, 29)
        # s&p500 components price data
        data = yf.download(self.snp_info.ticker.to_list(), start=start, end=end)
        idx = pd.IndexSlice
        data.loc[:, idx["Adj Close", :]].droplevel(0, axis=1).dropna(axis=0, how="all")\
            .to_csv(Path(f"../data/closes.csv"))
        data.loc[:, idx["Volume", :]].droplevel(0, axis=1).dropna(axis=0, how="all").\
            to_csv(Path(f"../data/volumes.csv"))

    @staticmethod
    def __get_lazy_dask_price_data_from_csv() -> dd.DataFrame:
        close_dd = dd.read_csv(Path(f"../data/closes.csv"), parse_dates=["Date"], dayfirst=True).set_index("Date")
        return close_dd

    def __get_current_price_data_from_dask_df(self) -> pd.DataFrame:
        start = self.window.coint_window_start_date
        end = self.window.trade_window_end_date
        snp_price_data = self.lazy_dask_price_data.loc[start:end].compute()
        return snp_price_data.dropna(axis=1, how="any")

    def __get_current_cluster_dict(self) -> Dict:
        existing_tickers = self.current_price_data.columns
        current_cluster_dict = {}
        for ticker in existing_tickers:
            ticker_sector = self.snp_info.loc[self.snp_info["ticker"] == ticker, "sector"].item()
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

    def __update_price_data_and_cluster(self) -> None:
        self.current_price_data = self.__get_current_price_data_from_dask_df()
        self.current_cluster_dict = self.__get_current_cluster_dict()
        self.allowed_couples = self.__get_allowed_couples()

    def update_data(self) -> None:
        self.__update_price_data_and_cluster()