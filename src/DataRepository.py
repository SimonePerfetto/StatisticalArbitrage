import itertools
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import dask.dataframe as dd
import pandas as pd
from src.DateManager import DateManager
import yfinance as yf
from abc import ABC, abstractmethod


class DataRepositoryBase(ABC):
    def __init__(self, date_manager: DateManager, file_name):
        self.date_manager = date_manager
        self.file_name = file_name

    @abstractmethod
    def get_lazy_price_data(self) -> dd.DataFrame:
        """"""

    @abstractmethod
    def get_current_price_data_from_lazy_df(self, start, end) -> pd.DataFrame:
        """"""


class SPXDataRepository(DataRepositoryBase):
    def __init__(self, date_manager: DateManager, file_name: str, data_load=False):
        super().__init__(date_manager, file_name)
        self.snp_info = self.__get_misc_data_from_web()
        if data_load: self.__download_price_data_from_yfinance()
        self.lazy_price_data = self.get_lazy_price_data()
        self.current_price_data = self.get_current_price_data_from_lazy_df(date_manager.coint_start_date,
                                                                           date_manager.trade_end_date)  # might useful to be called outside for understanding
        self.current_cluster_dict = self.__get_current_cluster_dict()
        self.allowed_couples = self.__get_allowed_couples()

    def get_lazy_price_data(self) -> dd.DataFrame:
        return dd.read_csv(Path(f"../data/{self.file_name}"), parse_dates=["Date"], dayfirst=True).set_index("Date")

    def get_current_price_data_from_lazy_df(self, start, end) -> pd.DataFrame:
        snp_price_data = self.lazy_price_data.loc[start:end].compute()
        return snp_price_data.dropna(axis=1, how="any")

    @staticmethod
    def __get_misc_data_from_web() -> pd.DataFrame:
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_info = wiki_table[0].loc[:, ["Symbol", "GICS Sub-Industry"]]
        sp_info.columns = ["ticker", "sector"]
        sp_info = sp_info.loc[~sp_info.ticker.isin(['GOOG', 'NWS'])].reset_index(drop=True)
        return sp_info

    def __download_price_data_from_yfinance(self) -> None:
        start, end = datetime(2008, 1, 2), datetime(2021, 10, 29)
        # s&p500 components price data
        data = yf.download(self.snp_info.ticker.to_list(), start=start, end=end)
        idx = pd.IndexSlice
        data.loc[:, idx["Adj Close", :]].droplevel(0, axis=1).dropna(axis=0, how="all") \
            .to_csv(Path(f"../data/closes.csv"))
        data.loc[:, idx["Volume", :]].droplevel(0, axis=1).dropna(axis=0, how="all"). \
            to_csv(Path(f"../data/volumes.csv"))

    def __get_current_cluster_dict(self) -> Dict:
        existing_tickers = self.current_price_data.columns
        existing_snp_info = self.snp_info.loc[self.snp_info.ticker.isin(existing_tickers)]
        return {sector: list(tickers) for sector, tickers in existing_snp_info.groupby('sector')['ticker']}

    def __get_allowed_couples(self) -> List:
        return [couple for ticker_list in self.current_cluster_dict.values()
                for couple in itertools.combinations(ticker_list, 2) if len(ticker_list) > 1]

    def __update_price_data_and_cluster(self) -> None:
        self.current_price_data = self.get_current_price_data_from_lazy_df(self.date_manager.coint_start_date,
                                                                           self.date_manager.trade_end_date)
        self.current_cluster_dict = self.__get_current_cluster_dict()
        self.allowed_couples = self.__get_allowed_couples()

    #TODO: might become an abstract method to be implemented
    def update_data(self) -> None:
        self.__update_price_data_and_cluster()



class CryptoDataRepository(DataRepositoryBase):
    def get_current_price_data_from_lazy_df(self, start, end) -> pd.DataFrame:
        raise NotImplementedError("Not implemented yet")

    def get_lazy_price_data(self) -> dd.DataFrame:
        raise NotImplementedError("Not implemented yet")

