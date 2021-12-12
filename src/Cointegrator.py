import random
from typing import Tuple, List, Union
import pandas as pd
from statsmodels.tsa.api import adfuller
from src.DataRepository import DataRepository
import statsmodels.api as sm
from src.util.OnlineRollingStats import OnlineRollingStats
from hurst import compute_Hc
from datetime import date
import numpy as np
import cufflinks

class Stock:
    def __init__(self, ticker: str, repository: DataRepository):
        self.ticker = ticker
        self.window_prices = self.get_window_prices(repository.current_price_data)

    def __repr__(self):
        return f'{self.ticker}'

    def get_ticker(self):
        return self.ticker

    def get_window_prices(self, current_price_data: pd.DataFrame) -> pd.Series:
        return current_price_data[self.ticker]

    def get_todays_price(self, today) -> float:
        return self.window_prices.loc[pd.to_datetime(today)]


class CointPair:
    def __init__(self, stock_x: Stock, stock_y: Stock, cointegration_result: List, roll_stats_window: int,
                 num_std_away: float):
        self.roll_stats_window: int = roll_stats_window
        self.num_std_away: float = num_std_away
        self.stock_x: Stock = stock_x
        self.stock_y: Stock = stock_y
        self.hedge_ratio, self.residuals = cointegration_result
        self.residuals_mavg: pd.Series = self.residuals.rolling(self.roll_stats_window).mean()
        self.residuals_mstdv: pd.Series = self.residuals.rolling(self.roll_stats_window).std()
        self.upper_band: pd.Series = self.residuals_mavg + self.num_std_away * self.residuals_mstdv
        self.lower_band: pd.Series = self.residuals_mavg - self.num_std_away * self.residuals_mstdv
        self.signals: pd.Series = self.initialize_signals()
        self.last_residual, self.last_roll_mean, self.last_roll_std = None, None, None
        self.last_lower_band, self.last_upper_band = None, None
        self.last_signal = None

    def __repr__(self):
        return f"CointPair({self.stock_x.ticker}, {self.stock_y.ticker})"

    def update_signal(self, today, no_new_trades_from_date, trade_window_end_date) -> None:
        self.last_residual, self.last_roll_mean, self.last_roll_std = self.__build_last_resid_mean_std(today)
        self.last_lower_band, self.last_upper_band = self.__build_last_upper_lower_bound()
        self.last_signal = self.__get_last_pair_signal(today, no_new_trades_from_date, trade_window_end_date)
        self.__update_coint_pair_series(today)

    def initialize_signals(self):
        return pd.Series(np.zeros(len(self.residuals)), index=self.residuals.index).astype("int32")

    def get_hedge_ratio(self):
        return self.hedge_ratio

    def get_todays_price_x_y(self, today: date) -> Tuple:
        px = self.stock_x.get_todays_price(today)
        py = self.stock_y.get_todays_price(today)
        return px, py

    def get_ticker_x_y(self):
        return self.stock_x.get_ticker(), self.stock_y.get_ticker()

    def plot_residuals_and_bb_bands(self, trade_action) -> None:
        residual_features_df = pd.concat([self.residuals, self.residuals_mavg,
                                          self.upper_band, self.lower_band], axis=1).dropna(axis=0)
        residual_features_df.columns = ["Residuals", "Residuals MA", "Residuals Upper BB", "Residuals Lower BB"]
        figure = residual_features_df.iplot(colorscale="polar", theme="white", asFigure=True,
                                            title=f"{trade_action} {self} - Residuals, BolBands, Residuals MA",
                                            xTitle="Time")
        figure.update_layout(font=dict(family="Computer Modern"))
        # figure.write_image("images/img.pdf", format="pdf")
        figure.show()

    def __build_last_resid_mean_std(self, today) -> Tuple:
        last_residual = self.__compute_last_residual(today)
        last_roll_mean, last_roll_std = self.__compute_last_mean_and_std(last_residual)
        return last_residual, last_roll_mean, last_roll_std

    def __build_last_upper_lower_bound(self) -> Tuple:
        last_upper = self.last_roll_mean + self.num_std_away * self.last_roll_std
        last_lower = self.last_roll_mean - self.num_std_away * self.last_roll_std
        return last_lower, last_upper

    def __update_coint_pair_series(self, today) -> None:
        self.__update_series(self.residuals, self.last_residual, today)
        self.__update_series(self.residuals_mavg, self.last_roll_mean, today)
        self.__update_series(self.residuals_mstdv, self.last_roll_std, today)
        self.__update_series(self.upper_band, self.last_upper_band, today)
        self.__update_series(self.lower_band, self.last_lower_band, today)
        self.__update_series(self.signals, self.last_signal, today)

    def __compute_last_residual(self, today) -> float:
        last_price_y = self.stock_y.window_prices[today.strftime('%Y-%m-%d')]
        last_price_x = self.stock_x.window_prices[today.strftime('%Y-%m-%d')]
        last_residual = last_price_y - self.hedge_ratio * last_price_x
        return last_residual

    def __compute_last_mean_and_std(self, last_residual) -> Tuple:
        mean_old = self.residuals_mavg.values[-1]
        std_old = self.residuals_mstdv.values[-1]
        first_residual = self.residuals.values[-self.roll_stats_window]
        online_roll = OnlineRollingStats(window_size=self.roll_stats_window, mean=mean_old, stdv=std_old)
        mean_new, std_new = online_roll.update(new=last_residual, old=first_residual)
        return mean_new, std_new

    @staticmethod
    def __update_series(series, last_item, today) -> None:
        series.loc[pd.to_datetime(today)] = last_item

    def __get_last_pair_signal(self, today, no_new_trades_from_date, trade_window_end_date) -> int:
        previous_signal = self.signals.values[-1]
        if today < no_new_trades_from_date:
            if previous_signal == 0: return self.__evaluate_entry()
            else: return self.__evaluate_exit(previous_signal)

        elif today < trade_window_end_date:
            if previous_signal == 0: return 0
            else: return self.__evaluate_exit(previous_signal)

        return 0

    def __evaluate_entry(self) -> int:
        if self.last_residual > self.last_upper_band:
            return -1
        elif self.last_residual < self.last_lower_band:
            return 1
        else:
            return 0

    def __evaluate_exit(self, previous_signal) -> int:
        if previous_signal == 1:
            is_exit = self.last_residual > self.last_roll_mean
        else:
            is_exit = self.last_residual < self.last_roll_mean
        return previous_signal * (1 - is_exit)

    def set_signal(self, signal_value):
        self.signals.iloc[-1] = signal_value

    def get_last_signal(self):
        return self.signals.iloc[-1]

    def get_penultimate_signal(self):
        return self.signals.iloc[-2]

class Cointegrator:

    def __init__(self, repository: DataRepository, roll_stats_window: int, num_std_away: float):
        self.repository: DataRepository = repository
        self.roll_stats_window: int = roll_stats_window
        self.num_std_away: float = num_std_away
        self.cointegrated_pairs = self.create_cointegrated_pairs()

    @staticmethod
    def __get_hedge_ratio_and_residuals(x: pd.Series, y: pd.Series) -> Tuple:
        reg = sm.OLS(y, x).fit()
        hedge_ratio = reg.params[0]
        residuals = y - hedge_ratio * x
        return hedge_ratio, residuals

    @staticmethod
    def __get_hurst(res: pd.Series):
        h, _, _ = compute_Hc(res)
        return h

    def __coint_check(self, residuals: pd.Series) -> bool:
        adf_results = adfuller(residuals)
        adf_pvalue = adf_results[1]
        # hurst = self.__get_hurst(residuals)
        return adf_pvalue < 0.01  # and hurst < 0.35

    def cointegrate(self, stock_x: Stock, stock_y: Stock) -> Union[Tuple, None]:
        stock_x_coint_window_prices = stock_x.window_prices[:self.repository.window.coint_window_end_date]
        stock_y_coint_window_prices = stock_y.window_prices[:self.repository.window.coint_window_end_date]
        hedge_ratio, residuals = self.__get_hedge_ratio_and_residuals(stock_x_coint_window_prices,
                                                                      stock_y_coint_window_prices)
        if self.__coint_check(residuals):
            return hedge_ratio, residuals

    def create_cointegrated_pairs(self) -> List[CointPair]:
        stock_obj_dict, cointpair_stocks, cointpair_list = {}, [], []
        random.seed(5)
        shuffled_allowed_couples_raw = random.sample(self.repository.allowed_couples,
                                                     k=len(self.repository.allowed_couples))
        shuffled_allowed_couples = self.__remove__class_a_b_share_tickers(shuffled_allowed_couples_raw)
        for tick_x, tick_y in shuffled_allowed_couples:
            if tick_x in cointpair_stocks or tick_y in cointpair_stocks: continue
            if tick_x not in stock_obj_dict: stock_obj_dict[tick_x] = Stock(tick_x, self.repository)
            if tick_y not in stock_obj_dict: stock_obj_dict[tick_y] = Stock(tick_y, self.repository)
            stock_x, stock_y = stock_obj_dict[tick_x], stock_obj_dict[tick_y]
            cointegration_result = self.cointegrate(stock_x, stock_y)
            if cointegration_result is not None:
                cointpair = CointPair(stock_x, stock_y, cointegration_result, self.roll_stats_window, self.num_std_away)
                cointpair_stocks += [tick_x, tick_y]
                cointpair_list.append(cointpair)
        return cointpair_list

    @staticmethod
    def __remove__class_a_b_share_tickers(couples_raw: List[Tuple]):
        couples = [couple for couple in couples_raw if couple not in [('GOOG', 'GOOGL'), ('GOOGL', 'GOOG'),
                                                                      ('NWS', 'NWSA'), ('NWSA', 'NWS')]]
        return couples
