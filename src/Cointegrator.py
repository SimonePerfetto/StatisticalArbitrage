import random
from typing import Tuple, List, Union, Dict
import pandas as pd
from statsmodels.tsa.api import adfuller
from src.DataRepository import SPXDataRepository
from src.Stock import Stock
from src.util.OnlineRollingStats import OnlineRollingStats
from hurst import compute_Hc
from datetime import date
import numpy as np
from src.util.KalmanUtils import KalmanUtils
from src.util.enumerations import TradingAction
import cufflinks as cf

class OLSParams:
    def __init__(
            self,
            cointegration_result: List,
            stock_x: Stock,
            stock_y: Stock,
            kf_flag: bool = True
    ):
        self._hedge_ratio, self._intercept, self._residuals = cointegration_result
        self._kf_flag = kf_flag
        self._kalman_utils = KalmanUtils(
            cointegration_result=cointegration_result,
            stock_x=stock_x,
            stock_y=stock_y
        ) if kf_flag else None
        if kf_flag: self._override_hedge_intercept_and_residuals()

    @property
    def hedge_ratio(self) -> float: return self._hedge_ratio

    @property
    def intercept(self) -> float: return self._intercept

    @property
    def residuals(self) -> pd.Series: return self._residuals

    @property
    def kf_flag(self) -> pd.Series: return self._kf_flag

    @property
    def kalman_utils(self) -> Union[KalmanUtils, None]: return self._kalman_utils

    @hedge_ratio.setter
    def hedge_ratio(self, value) -> None: self._hedge_ratio = value

    @intercept.setter
    def intercept(self, value) -> None: self._intercept = value

    @residuals.setter
    def residuals(self, value) -> None: self._residuals = value

    def _override_hedge_intercept_and_residuals(self) -> None:
        self.residuals = self.kalman_utils.kf_residuals
        self.hedge_ratio, self.intercept = self.kalman_utils.kf_hedge_ratio, self.kalman_utils.kf_intercept

class SignalBuilder:
    def __init__(
            self,
            roll_stats_window: int,
            num_std_away: float,
            ols_params: OLSParams
    ):
        self._roll_stats_window = roll_stats_window
        self._num_std_away = num_std_away
        self._ols_params = ols_params
        self.residual_data = self._build_residual_data(ols_params=ols_params)
        self._last_residual = ols_params.residuals.values[-1]
        self._last_roll_mean, self._last_roll_std, self._last_upper_band, self._last_lower_band = \
            self.residual_data.iloc[-1, :]
        self._signals = self._initialize_signals()
        self._last_signal = self.signals[-1]

    @property
    def roll_stats_window(self) -> float: return self._roll_stats_window

    @property
    def num_std_away(self) -> float: return self._num_std_away

    @property
    def ols_params(self) -> OLSParams: return self._ols_params

    @property
    def signals(self) -> pd.Series: return self._signals

    @signals.setter
    def signals(self, value) -> None: self._signals = value

    @property
    def last_residual(self) -> float: return self._last_residual

    @last_residual.setter
    def last_residual(self, value) -> None: self._last_residual = value

    @property
    def last_roll_mean(self) -> float: return self._last_roll_mean

    @last_roll_mean.setter
    def last_roll_mean(self, value) -> None: self._last_roll_mean = value

    @property
    def last_roll_std(self) -> float: return self._last_roll_std

    @last_roll_std.setter
    def last_roll_std(self, value) -> None: self._last_roll_std = value

    @property
    def last_upper_band(self) -> float: return self._last_upper_band

    @last_upper_band.setter
    def last_upper_band(self, value) -> None: self._last_upper_band = value

    @property
    def last_lower_band(self) -> float: return self._last_lower_band

    @last_lower_band.setter
    def last_lower_band(self, value) -> None: self._last_lower_band = value

    @property
    def last_signal(self) -> float: return self._last_signal

    @last_signal.setter
    def last_signal(self, value) -> None: self._last_signal = value

    def _build_residual_data(
            self,
            ols_params: OLSParams
    ) -> pd.DataFrame:
        res_data = pd.DataFrame(index=ols_params.residuals.index)
        res_data["Res_MAvg"] = ols_params.residuals.rolling(self.roll_stats_window).mean()
        res_data["Res_MStdv"] = ols_params.residuals.rolling(self.roll_stats_window).std()
        res_data["Upper_BBand"] = res_data["Res_MAvg"] + self.num_std_away * res_data["Res_MStdv"]
        res_data["Lower_BBand"] = res_data["Res_MAvg"] - self.num_std_away * res_data["Res_MStdv"]

        return res_data


    def _initialize_signals(self) -> pd.Series:
        return pd.Series(
            data=np.zeros(len(self.ols_params.residuals)),
            index=self.ols_params.residuals.index
        ).astype("int32")

    def _compute_last_residual(
            self,
            last_px: float,
            last_py: float
    ) -> float:
        if self.ols_params.kf_flag:
            self.ols_params.kalman_utils.update_kalman_hedge_intercept(
                last_price_x=last_px,
                last_price_y=last_py
            )
            self.ols_params.hedge_ratio = self.ols_params.kalman_utils.kf_hedge_ratio
            self.ols_params.intercept = self.ols_params.kalman_utils.kf_intercept
        last_residual = last_py - self.ols_params.hedge_ratio * last_px - self.ols_params.intercept
        return last_residual

    def _compute_last_mean_and_std(
            self,
            last_residual: float
    ) -> Tuple:
        previous_mean = self.residual_data["Res_MAvg"].values[-1]
        previous_std = self.residual_data["Res_MStdv"].values[-1]
        first_residual = self.ols_params.residuals.values[-self.roll_stats_window]
        online_roll = OnlineRollingStats(
            roll_window_size=self.roll_stats_window,
            mean=previous_mean,
            stdv=previous_std
        )
        new_mean, new_std = online_roll.update(
            new=last_residual,
            old=first_residual
        )
        return new_mean, new_std

    def update_residuals_data(
            self,
            last_px: float,
            last_py: float,
            today: date
    ) -> None:
        self.last_residual = self._compute_last_residual(
            last_px=last_px,
            last_py=last_py
        )
        self.last_roll_mean, self.last_roll_std = self._compute_last_mean_and_std(last_residual=self.last_residual)
        self.last_upper_band = self.last_roll_mean + self.num_std_away * self.last_roll_std
        self.last_lower_band = self.last_roll_mean - self.num_std_away * self.last_roll_std

        self.ols_params.residuals = self.ols_params.residuals.append(
            pd.Series(
                data=self.last_residual,
                index=[pd.to_datetime(today)]
            )
        )
        self.residual_data.loc[pd.to_datetime(today)] = [
            self.last_roll_mean,
            self.last_roll_std,
            self.last_upper_band, self.last_lower_band
        ]

    def update_signal(
            self, today: date,
            no_new_trades_from_date: date,
            trade_end_date: date
    ) -> None:

        self.last_signal = self._compute_last_signal(
            today=today,
            no_new_trades_from_date=no_new_trades_from_date,
            trade_window_end_date=trade_end_date
        )
        self.signals = self.signals.append(
            pd.Series(
                data=self.last_signal,
                index=[pd.to_datetime(today)]
            )
        )

    def _compute_last_signal(
            self,
            today: date,
            no_new_trades_from_date: date,
            trade_window_end_date: date
    ) -> int:
        previous_signal = self.signals.values[-1]

        if today < no_new_trades_from_date:
            if previous_signal == 0: return self._evaluate_entry()
            else: return self._evaluate_exit(previous_signal=previous_signal)

        elif today < trade_window_end_date:
            if previous_signal == 0: return 0
            else: return self._evaluate_exit(previous_signal=previous_signal)

        return 0

    def _evaluate_entry(self) -> int:
        if self.last_residual > self.last_upper_band:
            return -1
        elif self.last_residual < self.last_lower_band:
            return 1
        else:
            return 0

    def _evaluate_exit(
            self,
            previous_signal: int
    ) -> int:
        if previous_signal == 1:
            is_exit = self.last_residual > self.last_roll_mean
        else:
            is_exit = self.last_residual < self.last_roll_mean
        return previous_signal * (1 - is_exit)

    def get_last_signal(self) -> int:
        return self.signals.iloc[-1]

    def get_penultimate_signal(self) -> int:
        return self.signals.iloc[-2]

class CointPair:
    def __init__(
            self,
            stock_x: Stock,
            stock_y: Stock,
            signal_builder: SignalBuilder
    ):
        self._stock_x = stock_x
        self._stock_y = stock_y
        self._signal_builder = signal_builder

    def __repr__(self):
        return f"CointPair({self.stock_x.ticker}, {self.stock_y.ticker})"

    @property
    def stock_x(self) -> Stock: return self._stock_x

    @property
    def stock_y(self) -> Stock: return self._stock_y

    @property
    def signal_builder(self) -> SignalBuilder: return self._signal_builder

    @property
    def hedge_ratio(self) -> float: return self.signal_builder.ols_params.hedge_ratio

    def get_todays_price_x_y(self, today: date) -> Tuple:
        px = self.stock_x.get_todays_price(today=today)
        py = self.stock_y.get_todays_price(today=today)
        return px, py

    def get_ticker_x_y(self):
        return self.stock_x.ticker, self.stock_y.ticker

    def override_signal(
            self,
            signal_value: int
    ) -> None:
        self.signal_builder.last_signal = signal_value
        signals_series = self.signal_builder.signals.copy()
        signals_series.iloc[-1] = signal_value
        self.signal_builder.signals = signals_series

    def plot_residuals_and_bb_bands(
            self,
            trade_action: TradingAction
    ) -> None:
        residual_features_df = self.signal_builder.residual_data.loc[:, ["Res_MAvg", "Upper_BBand", "Lower_BBand"]]
        residual_features_df["Res"] = self.signal_builder.ols_params.residuals
        residual_features_df.columns = ["Residuals", "Residuals MA", "Residuals Upper BB", "Residuals Lower BB"]
        figure = residual_features_df.dropna(axis=0)\
            .iplot(colorscale="polar", theme="white", asFigure=True,
                   title=f"{trade_action} {self} - Residuals, BolBands, Residuals MA", xTitle="Time")
        figure.update_layout(font=dict(family="Computer Modern"))
        figure.write_image("images/res.svg", format="svg")
        figure.show()
        f = pd.concat(
            [
                self.stock_x.price_ts,
                self.stock_y.price_ts / self.stock_y.price_ts.values[0] * self.stock_x.price_ts.values[0]
            ],
            axis=1
        ).iloc[:240, :].iplot(asFigure=True, )
        f.show()
        f.write_image("images/cointpair.svg", format="svg")

class Cointegrator:

    def __init__(
            self,
            roll_stats_window: int,
            num_std_away: float
    ):
        self.roll_stats_window: int = roll_stats_window
        self.num_std_away: float = num_std_away

    @staticmethod
    def _get_hurst(res: pd.Series) -> float:
        h, _, _ = compute_Hc(series=res)
        return h

    @staticmethod
    def _coint_check(residuals: np.ndarray) -> bool:
        adf_results = adfuller(x=residuals)
        adf_pvalue = adf_results[1]
        # hurst = self.__get_hurst(residuals)
        return adf_pvalue < 0.01  # and hurst < 0.35


    def create_cointegrated_pairs(
            self,
            repos: SPXDataRepository,
            coint_start: date,
            coint_end: date
    ) -> List[CointPair]:

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
            ols_params = OLSParams(coint_result.values(), stock_x, stock_y)
            signal_builder = SignalBuilder(self.roll_stats_window, self.num_std_away, ols_params)
            cointpair = CointPair(stock_x, stock_y, signal_builder)
            cointpair_list.append(cointpair)
        #ed = time.time()
        #print(f"Time to cointegrate: {ed - st}")

        return cointpair_list

    @staticmethod
    def get_x_y_tickers(couples: List[Tuple]) -> Tuple[List[str], List[str]]:
        return [c[0] for c in couples], [c[1] for c in couples]

    def cointegrate(
            self,
            repos: SPXDataRepository,
            x_tickers: List[str],
            y_tickers: List[str],
            start_date: date,
            end_date: date
    ) -> Dict[Tuple, Dict[str, Union[float, pd.Series]]]:
        xs = repos.filter_price_data(
            start=start_date,
            end=end_date,
            tickers_list=x_tickers
        )
        ys = repos.filter_price_data(
            start=start_date,
            end=end_date,
            tickers_list=y_tickers
        )
        xs_mean = np.mean(xs.values, axis=0, keepdims=True)
        xs_norm = xs.values - xs_mean
        ys_mean = np.mean(ys.values, axis=0, keepdims=True)
        ys_norm = ys.values - ys_mean
        hedge_ratio = np.einsum('ij,ij->j', xs_norm, ys_norm) / np.einsum('ij,ij->j', xs_norm, xs_norm)
        intercept = (ys_mean - hedge_ratio * xs_mean).flatten()
        residuals_matrix = ys.values - xs.values * hedge_ratio - intercept * np.ones(xs.shape)
        residuals_df = pd.DataFrame(data=residuals_matrix, index=xs.index)

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


