import math
import re
from datetime import date, timedelta
from enum import Enum, unique
from pathlib import Path
from typing import Dict, Optional, Set, List

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
        self.all_dates: List[date] = self.__load_dates()
        self.fundamental_data = None

    def get(self,
            datatype: Universes,
            trading_dates: List[date]):
        self.__get_from_disk_and_store(datatype, trading_dates)

    def get_fundamental(self, trading_date: date):
        if self.fundamental_data is None:
            self.__get_fundamental_from_disk()
        return self.fundamental_data.loc[trading_date,]

    def remove_dead_tickers(self, datatype: Universes, alive_and_dead_ticker_data: DataFrame):
        # Just gets the first column of data (don't care which feature) of the ticker to see if theyre all nan [dead]
        # If they're all nan, we assume the ticker didnt exist then, and so remove from the window
        # If there are some (or no) nans then the ticker is live

        alive_tickers = [i for i in self.tickers]
        junk_val = 'XXXXX'
        for idx, ticker in enumerate(self.tickers):

            column = alive_and_dead_ticker_data.loc[:, ticker].iloc[:, 0]
            is_nans = [True if math.isnan(i) else False for i in column]

            if any(is_nans):
                # ticker is alive for this window
                alive_tickers[idx] = junk_val

        alive_tickers = [i for i in alive_tickers if i != junk_val]

        return alive_tickers, alive_and_dead_ticker_data.loc[:, IndexSlice[alive_tickers, :]]

    def __load_dates(self) -> List[date]:

        def _f(datatype: List):
            d = pd.read_csv(datatype.value,
                            squeeze=True,
                            header=0,
                            index_col=0,
                            usecols=[0])
            return [i.date() for i in pd.to_datetime(d.index, format='%d/%m/%Y')]

        common_dates = set(_f(Universes.SNP))#.intersection(set(_f(Universes.ETFs)))
        return sorted(common_dates)

    def check_date_equality(self, d1: date, d2: date):
        return (d1.day == d2.day and
                d1.month == d2.month and
                d1.year == d2.year)

    def __get_from_disk_and_store(self, datatype: Universes, trading_dates: List[date]):
        idxs_to_read = []
        for d1 in trading_dates:
            for idx, d2 in enumerate(self.all_dates):
                if self.check_date_equality(d1, d2):
                    idxs_to_read.append(idx)
                    break

        d = pd.read_csv(datatype.value,
                        squeeze=True,
                        header=0,
                        index_col=0,
                        skiprows=range(1, idxs_to_read[0] + 1),
                        low_memory=False,
                        nrows=len(idxs_to_read))

        d.index = pd.to_datetime(d.index, format='%d/%m/%Y')

        match_results = [re.findall(r"(\w+)", col) for col in d.columns]
        tickers = [SnpTickers(r[0].upper()) for r in match_results]
        features = [Features(r[-1].upper()) for r in match_results]
        self.tickers = set(tickers)
        self.features = set(features)

        d.columns = pd.MultiIndex.from_tuples(
            tuples=list(zip(tickers, features)),
            names=['Ticker', 'Feature']
        )

        d = self.forward_fill(d)

        # weekday_data_for_window = data_for_all_time[data_for_all_time.index.isin(trading_dates)]

        d = d[d.index.isin(self.all_dates)]
        d = d.drop_duplicates(keep='first')

        self.all_data = pd.concat([self.all_data, d], axis=0).drop_duplicates(keep='first')

        self.all_data = self.all_data.drop_duplicates(keep='first')


    def __normalised_true_range(self, row: Se):
        try:
            # https://www.investopedia.com/terms/a/atr.asp - we use the formula for TR and then divide by close
            return max(row[0] - row[1], abs(row[0] - row[2]), abs(row[1] - row[2])) / row[2]
        except RuntimeError:
            return np.nan()

    def __get_fundamental_from_disk(self):
        data = pd.read_csv(Path(f"../resources/fundamental_snp.csv"),
                           index_col=0)
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        fundamental_start = date(2016, 3, 31)
        fundamental_date = [date for date in self.all_dates if date > fundamental_start]
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

    def forward_fill(self, df: DataFrame):
        return pd.DataFrame(df).fillna(method='ffill')



