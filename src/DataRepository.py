import math
import re
from datetime import date, timedelta
from enum import Enum, unique
from pathlib import Path
from typing import Set, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import IndexSlice
from pandas import Series as Se

from src.util.Features import Features
from src.util.Tickers import SnpTickers


@unique
class Universes(Enum):
    # ETFs = Path(f"../resources/all_etfs2.csv")
    SNP = Path(f"../resources/all_snp2.csv")
    ETFs = Path(f"../resources/all_etfs2_no_vol.csv")

    # ETFs = Path(f"../resources/etf_test.csv")
    # SNP = Path(f"../resources/snp_test.csv")


class DataRepository:
    def __init__(self, window_length: timedelta):
        self.window_length: timedelta = window_length
        self.all_data: DataFrame = None
        self.tickers: Set[SnpTickers]
        self.features: Set[Features]
        self.all_dates = self.__load_dates()
        self.fundamental_data = None

    def get(self,
            snp_data: Universes,
            trading_dates: pd.Series):
        self.__get_from_disk_and_store(snp_data, trading_dates)

    def get_fundamental(self, trading_date: date):
        if self.fundamental_data is None:
            self.__get_fundamental_from_disk()
        return self.fundamental_data.loc[trading_date,]

    def remove_dead_tickers(self, alive_and_dead_ticker_data: DataFrame):
        # Just gets the first column of data (don't care which feature) of the ticker to see if theyre all nan [dead]
        # If they're all nan, we assume the ticker didnt exist then, and so remove from the window
        # If there are some (or no) nans then the ticker is live

        alive_tickers = [tick for tick in self.tickers]
        junk_val = 'XXXXX'
        for idx, ticker in enumerate(self.tickers):
            column = alive_and_dead_ticker_data.loc[:, ticker].iloc[:, 0]
            is_nans = [True if math.isnan(i) else False for i in column]
            if any(is_nans):
                # we have no Nans, thus ticker is alive for this window
                alive_tickers[idx] = junk_val  # placeholder for tickers in the alive list which are actually not alive

        alive_tickers = [tick for tick in alive_tickers if tick != junk_val]

        return alive_tickers, alive_and_dead_ticker_data.loc[:, IndexSlice[alive_tickers, :]]

    def __load_dates(self) -> List[date]:
        dates = pd.read_csv(Universes.SNP.value, index_col=0, usecols=[0],
                            parse_dates=True, dayfirst=True)
        dates["dates_col"] = dates.index
        return dates["dates_col"]


    def __get_from_disk_and_store(self, snp_data: Universes, trading_dates: pd.Series):
        """
        :param snp_data:
        :param trading_dates:
        """
        idxs_to_read = []
        for d1 in trading_dates:
            for idx, d2 in enumerate(self.all_dates):
                if d1==d2:
                    idxs_to_read.append(idx)
                    break

        temp_snp_data = pd.read_csv(snp_data.value, squeeze=True, header=0, index_col=0,
                        skiprows=range(1, idxs_to_read[0] + 1), nrows=len(idxs_to_read),
                        low_memory=False, parse_dates=True, dayfirst=True)

        # create a list of lists, each of them including ["ticker", "feature"]
        match_results = [re.findall(r"(\w+)", col) for col in temp_snp_data.columns]
        tickers = [SnpTickers(r[0].upper()) for r in match_results]
        features = [Features(r[-1].upper()) for r in match_results]
        self.tickers = set(tickers)
        self.features = set(features)

        temp_snp_data.columns = pd.MultiIndex.from_tuples(
            tuples=list(zip(tickers, features)),
            names=['Ticker', 'Feature'])
        # assume same day-before price for NaN values
        temp_snp_data = temp_snp_data.fillna(method='ffill')
        temp_snp_data = temp_snp_data.drop_duplicates(keep='first')
        self.all_data = pd.concat([self.all_data, temp_snp_data], axis=0).drop_duplicates(keep='first')
        self.all_data = self.all_data.drop_duplicates(keep='first')


    @staticmethod
    def __normalised_true_range(row: Se):
        try:
            # https://www.investopedia.com/terms/a/atr.asp -
            # we use the formula for TR and then divide by close
            return max(row[0] - row[1], abs(row[0] - row[2]), abs(row[1] - row[2])) / row[2]
        except RuntimeError:
            return np.nan()

    def __get_fundamental_from_disk(self):
        data = pd.read_csv(Path(f"../resources/fundamental_snp.csv"),
                           index_col=0)
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        fundamental_start = date(2016, 3, 31)
        fundamental_date = [dt for dt in self.all_dates if dt > fundamental_start]
        df = pd.DataFrame(index=fundamental_date)
        df = df.join(data, how='outer')
        match_results = [re.findall(r"(\w+)", col) for col in df.columns]
        funda_tickers = [SnpTickers(r[0]) for r in match_results]
        funda_features = [r[1] for r in match_results]
        df.columns = pd.MultiIndex.from_tuples(
            tuples=list(zip(funda_tickers, funda_features)),
            names=['ticker', 'feature'])
        df = df.fillna(method='ffill')
        self.fundamental_data = df
        return




