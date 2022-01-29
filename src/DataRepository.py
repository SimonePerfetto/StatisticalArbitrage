import itertools
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import dask.dataframe as dd
import pandas as pd
import yfinance as yf
from abc import ABC, abstractmethod
from datetime import date


class DataRepositoryBase(ABC):
    def __init__(self, file_name):
        self.file_name = file_name

    @abstractmethod
    def get_price_data(self): pass


class SPXDataRepository(DataRepositoryBase):
    def __init__(self, file_name: str, data_download: bool = False):
        super().__init__(file_name)
        self.snp_info = self._get_misc_data_from_web()
        if data_download:
            self._download_price_data_from_yfinance()
        self.price_data = self.get_price_data()
        self.current_cluster_dict = None
        self.allowed_couples = None

    def get_price_data(self) -> pd.DataFrame:
        return pd.read_csv(Path(f"../data/{self.file_name}"), parse_dates=["Date"], dayfirst=True).set_index("Date")

    def filter_price_data(self, start: date, end: date, tickers_list=None):
        if tickers_list is None: return self.price_data.loc[start:end, :]
        else: return self.price_data.loc[start:end, tickers_list]

    @staticmethod
    def _get_misc_data_from_web() -> pd.DataFrame:
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_info = wiki_table[0].loc[:, ["Symbol", "GICS Sub-Industry"]]
        sp_info.columns = ["ticker", "sector"]
        sp_info = sp_info.loc[~sp_info.ticker.isin(['GOOG', 'NWS'])].reset_index(drop=True)
        return sp_info

    def _download_price_data_from_yfinance(self) -> None:
        start, end = datetime(2008, 1, 2), datetime(2021, 10, 29)
        # s&p500 components price data
        data = yf.download(self.snp_info.ticker.to_list(), start=start, end=end)
        idx = pd.IndexSlice
        data.loc[:, idx["Adj Close", :]].droplevel(0, axis=1).dropna(axis=0, how="all") \
            .to_csv(Path(f"../data/closes.csv"))
        data.loc[:, idx["Volume", :]].droplevel(0, axis=1).dropna(axis=0, how="all"). \
            to_csv(Path(f"../data/volumes.csv"))

    def _get_existing_tickers(self, start, end) -> List:
        return self.price_data.loc[start:end].dropna(axis=1).columns

    def _get_current_cluster_dict(self, start, end) -> Dict:
        existing_tickers = self._get_existing_tickers(start, end)
        existing_snp_info = self.snp_info.loc[self.snp_info.ticker.isin(existing_tickers)]
        return {sector: list(tickers) for sector, tickers in existing_snp_info.groupby('sector')['ticker']}

    def _get_allowed_couples(self) -> List:
        return [couple for ticker_list in self.current_cluster_dict.values()
                for couple in itertools.combinations(ticker_list, 2) if len(ticker_list) > 1]

    # TODO: might become an abstract method to be implemented
    def update_train_data(self, start: date, end: date) -> None:
        self.current_cluster_dict = self._get_current_cluster_dict(start, end)
        self.allowed_couples = self._get_allowed_couples()


class CryptoDataRepository(DataRepositoryBase):
    def get_price_data(self) -> dd.DataFrame:
        raise NotImplementedError("Not implemented yet")
