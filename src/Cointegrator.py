import random
from typing import Tuple, List
import pandas as pd
from statsmodels.tsa.api import adfuller
from src.DataRepository import SPXDataRepository
from src.util.OnlineRollingStats import OnlineRollingStats
from hurst import compute_Hc
from datetime import date
import numpy as np
import cufflinks
import time
from src.util.KalmanUtils import KalmanUtils


class Stock:
    def __init__(self, ticker: str, repos: SPXDataRepository):
        self._ticker = ticker
        self._price_ts = self.get_price_ts(repos)

    def __repr__(self):
        return f'{self._ticker}'

    @property
    def ticker(self):
        return self._ticker

    @property
    def price_ts(self):
        return self._price_ts

    def get_price_ts(self, repos) -> pd.Series:
        return repos.price_data.loc[:, self.ticker]

    def get_todays_price(self, today) -> float:
        return self.price_ts.loc[pd.to_datetime(today)]


class CointPair:
    def __init__(self, stock_x: Stock, stock_y: Stock, cointegration_result: List,
                 roll_stats_window: int, num_std_away: float, kf_flag=True):
        self.roll_stats_window = roll_stats_window
        self.num_std_away = num_std_away
        self.stock_x = stock_x
        self.stock_y = stock_y
        self.kf_flag = kf_flag
        self.hedge_ratio, self.intercept, self.residuals = cointegration_result
        if kf_flag:
            self.kalman_utils = KalmanUtils(cointegration_result, stock_x, stock_y)
            self._override_hedge_intercept_and_residuals()
        self.residuals_data = self.build_residuals_data()
        self.signals: pd.Series = self.initialize_signals()
        self.last_signal = 0
        self.last_residual, self.last_roll_mean, self.last_roll_std = None, None, None
        self.last_lower_band, self.last_upper_band = None, None

    def __repr__(self):
        return f"CointPair({self.stock_x.ticker}, {self.stock_y.ticker})"


    def _override_hedge_intercept_and_residuals(self) -> None:
        self.residuals = self.kalman_utils.kf_residuals
        self.hedge_ratio, self.intercept = self.kalman_utils.kf_hedge_ratio, self.kalman_utils.kf_intercept

    def build_residuals_data(self):
        res_data = pd.DataFrame(index=self.residuals.index)
        res_data["Res_MAvg"] = self.residuals.rolling(self.roll_stats_window).mean()
        res_data["Res_MStdv"] = self.residuals.rolling(self.roll_stats_window).std()
        res_data["Upper_BBand"] = res_data["Res_MAvg"] + self.num_std_away * res_data["Res_MStdv"]
        res_data["Lower_BBand"] = res_data["Res_MAvg"] - self.num_std_away * res_data["Res_MStdv"]

        return res_data

    def update_signal(self, today, no_new_trades_from_date, trade_window_end_date) -> None:
        self.last_residual, self.last_roll_mean, self.last_roll_std = self._build_last_resid_mean_std(today)
        self.last_lower_band, self.last_upper_band = self._build_last_upper_lower_bound()
        self.last_signal = self._compute_last_signal(today, no_new_trades_from_date, trade_window_end_date)
        self._update_coint_pair_data(today)

    def initialize_signals(self):
        return pd.Series(np.zeros(len(self.residuals)), index=self.residuals.index).astype("int32")

    def get_hedge_ratio(self):
        return self.hedge_ratio

    def get_todays_price_x_y(self, today: date) -> Tuple:
        px = self.stock_x.get_todays_price(today)
        py = self.stock_y.get_todays_price(today)
        return px, py

    def get_ticker_x_y(self):
        return self.stock_x.ticker, self.stock_y.ticker

    def plot_residuals_and_bb_bands(self, trade_action) -> None:
        residual_features_df = self.residuals_data.loc[:, ["Res_MAvg", "Upper_BBand", "Lower_BBand"]]
        residual_features_df = residual_features_df.insert(0, "Res", self.residuals).dropna(axis=0)
        residual_features_df.columns = ["Residuals", "Residuals MA", "Residuals Upper BB", "Residuals Lower BB"]
        figure = residual_features_df.dropna(axis=0)\
            .iplot(colorscale="polar", theme="white", asFigure=True,
                   title=f"{trade_action} {self} - Residuals, BolBands, Residuals MA", xTitle="Time")
        figure.update_layout(font=dict(family="Computer Modern"))
        # figure.write_image("images/img.pdf", format="pdf")
        figure.show()

    def _compute_last_residual(self, today) -> float:
        last_price_x, last_price_y = self.get_todays_price_x_y(today)

        if self.kf_flag:
            self.kalman_utils.update_kalman_hedge_intercept(last_price_x, last_price_y)
            self.hedge_ratio, self.intercept = self.kalman_utils.kf_hedge_ratio, self.kalman_utils.kf_intercept

        last_residual = last_price_y - self.hedge_ratio * last_price_x - self.intercept
        return last_residual

    def _compute_last_mean_and_std(self, last_residual) -> Tuple:
        previous_mean = self.residuals_data["Res_MAvg"].values[-1]
        previous_std = self.residuals_data["Res_MStdv"].values[-1]
        first_residual = self.residuals.values[-self.roll_stats_window]
        online_roll = OnlineRollingStats(roll_window_size=self.roll_stats_window, mean=previous_mean, stdv=previous_std)
        new_mean, new_std = online_roll.update(new=last_residual, old=first_residual)
        return new_mean, new_std

    def _build_last_resid_mean_std(self, today) -> Tuple:
        last_residual = self._compute_last_residual(today)
        last_roll_mean, last_roll_std = self._compute_last_mean_and_std(last_residual)
        return last_residual, last_roll_mean, last_roll_std

    def _build_last_upper_lower_bound(self) -> Tuple:
        last_upper = self.last_roll_mean + self.num_std_away * self.last_roll_std
        last_lower = self.last_roll_mean - self.num_std_away * self.last_roll_std
        return last_lower, last_upper

    def _update_coint_pair_data(self, today) -> None:
        self._update_series(self.residuals, self.last_residual, today)
        self._update_series(self.signals, self.last_signal, today)
        self._update_df(self.residuals_data,
                        [self.last_roll_mean, self.last_roll_std, self.last_upper_band, self.last_lower_band],
                        today)

    def _compute_last_signal(self, today, no_new_trades_from_date, trade_window_end_date) -> int:
        previous_signal = self.signals.values[-1]

        if today < no_new_trades_from_date:
            if previous_signal == 0: return self._evaluate_entry()
            else: return self._evaluate_exit(previous_signal)

        elif today < trade_window_end_date:
            if previous_signal == 0: return 0
            else: return self._evaluate_exit(previous_signal)

        return 0

    def _evaluate_entry(self) -> int:
        if self.last_residual > self.last_upper_band:
            return -1
        elif self.last_residual < self.last_lower_band:
            return 1
        else:
            return 0

    def _evaluate_exit(self, previous_signal) -> int:
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

    @staticmethod
    def _update_series(series, last_item, today) -> None:
        series.loc[pd.to_datetime(today)] = last_item

    @staticmethod
    def _update_df(df, new_row, today) -> None:
        df.loc[today] = new_row


class Cointegrator:

    def __init__(self, roll_stats_window: int, num_std_away: float):
        self.roll_stats_window: int = roll_stats_window
        self.num_std_away: float = num_std_away

    @staticmethod
    def _get_hurst(res: pd.Series):
        h, _, _ = compute_Hc(res)
        return h

    @staticmethod
    def _coint_check(residuals: np.ndarray) -> bool:
        adf_results = adfuller(residuals)
        adf_pvalue = adf_results[1]
        # hurst = self.__get_hurst(residuals)
        return adf_pvalue < 0.01  # and hurst < 0.35


    def create_cointegrated_pairs(self, repos: SPXDataRepository, coint_start: date, coint_end: date) -> List[CointPair]:
        #st = time.time()
        cointpair_list = []
        random.seed(5)
        shuffled_allowed_couples = random.sample(repos.allowed_couples,
                                                 k=len(repos.allowed_couples))
        x_tickers, y_tickers = self.get_x_y_tickers(shuffled_allowed_couples)
        coint_info_dict = self.cointegrate(repos, x_tickers, y_tickers, coint_start, coint_end)

        already_used =  set()
        for (ticker_x, ticker_y), coint_result in coint_info_dict.items():
            if ticker_x in already_used or ticker_y in already_used: continue
            already_used.add(ticker_x)
            already_used.add(ticker_y)
            stock_x, stock_y = Stock(ticker_x, repos), Stock(ticker_y, repos)
            cointpair = CointPair(stock_x, stock_y, coint_result.values(), self.roll_stats_window, self.num_std_away)
            cointpair_list.append(cointpair)

        #ed = time.time()
        #print(f"Time to cointegrate: {ed - st}")

        return cointpair_list

    @staticmethod
    def get_x_y_tickers(couples):
        return [c[0] for c in couples], [c[1] for c in couples]

    def cointegrate(self, repos, x_tickers, y_tickers, start, end):
        xs = repos.filter_price_data(start, end, x_tickers)
        ys = repos.filter_price_data(start, end, y_tickers)
        xs_mean = np.mean(xs.values, axis=0, keepdims=True)
        xs_norm = xs.values - xs_mean
        ys_mean = np.mean(ys.values, axis=0, keepdims=True)
        ys_norm = ys.values - ys_mean
        hedge_ratio = np.einsum('ij,ij->j', xs_norm, ys_norm) / np.einsum('ij,ij->j', xs_norm, xs_norm)
        intercept = (ys_mean - hedge_ratio * xs_mean).flatten()
        residuals_matrix = ys.values - xs.values * hedge_ratio - intercept * np.ones(xs.shape)
        residuals_df = pd.DataFrame(residuals_matrix, index=xs.index)

        coint_info_dict = {
            (x_tickers[i], y_tickers[i]):
                {
                    "hedgeratio": hedge_ratio[i],
                    "intercept": intercept[i],
                    "residuals": residuals_df.loc[:, i]
                }
            for i in range(xs.shape[1]) if self._coint_check(residuals_df.loc[:, i].values)
        }

        return coint_info_dict


