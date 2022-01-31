import pandas as pd

from src.DataRepository import SPXDataRepository


class Stock:
    def __init__(self, ticker: str, repos: SPXDataRepository):
        self._ticker = ticker
        self._price_ts = self.get_price_ts(repos)

    def __repr__(self):
        return f'{self._ticker}'

    @property
    def ticker(self) -> str:
        return self._ticker

    @property
    def price_ts(self) -> pd.Series:
        return self._price_ts

    def get_price_ts(self, repos) -> pd.Series:
        return repos.price_data.loc[:, self.ticker]

    def get_todays_price(self, today) -> float:
        return self.price_ts.loc[pd.to_datetime(today)]