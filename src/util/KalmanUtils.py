import numpy as np
from pykalman import KalmanFilter
from typing import Tuple, List, Union
import pandas as pd
from src.Stock import Stock


class KalmanUtils:
    def __init__(
            self,
            cointegration_result: List[Union[float, pd.Series]],
            stock_x: Stock,
            stock_y: Stock
    ):
        self._kf_model, self._state_means, self._state_covs = self.build_kalman_features(
            cointegration_result=cointegration_result,
            stock_x=stock_x,
            stock_y=stock_y
        )
        self._kf_residuals, self._kf_hedge_ratio, self._kf_intercept = self.get_kalman_results(
            coint_res=cointegration_result,
            stock_x=stock_x,
            stock_y=stock_y
        )

    @property
    def kf_model(self) -> KalmanFilter: return self._kf_model

    @property
    def kf_residuals(self) -> np.ndarray: return self._kf_residuals

    @property
    def state_means(self) -> np.ndarray: return self._state_means

    @state_means.setter
    def state_means(self, value) -> None: self._state_means = value

    @property
    def state_covs(self) -> np.ndarray: return self._state_covs

    @state_covs.setter
    def state_covs(self, value) -> None: self._state_covs = value

    @property
    def kf_hedge_ratio(self) -> float: return self._kf_hedge_ratio

    @kf_hedge_ratio.setter
    def kf_hedge_ratio(self, value) -> None: self._kf_hedge_ratio = value

    @ property
    def kf_intercept(self) -> float: return self._kf_intercept

    @kf_intercept.setter
    def kf_intercept(self, value) -> None: self._kf_intercept = value

    @staticmethod
    def build_kalman_features(
            cointegration_result: List[Union[float, pd.Series]],
            stock_x: Stock,
            stock_y: Stock
    ) -> Tuple:
        ols_hedge_ratio, ols_intercept, ols_residuals = cointegration_result
        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.vstack([stock_x.price_ts.loc[ols_residuals.index].values,
                             np.ones(len(ols_residuals))]).T[:, np.newaxis]

        kf_model = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=2,
            initial_state_mean=(ols_hedge_ratio, ols_intercept),
            # initial_state_mean=np.zeros(2),
            initial_state_covariance=np.ones((2, 2)),
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
            observation_covariance=1.0,
            transition_covariance=trans_cov
        )
        state_means, state_covs = kf_model.filter(stock_y.price_ts.loc[ols_residuals.index].values)
        return kf_model, state_means, state_covs

    def get_kalman_results(
            self,
            coint_res: List[Union[float, pd.Series]],
            stock_x: Stock,
            stock_y: Stock
    ) -> Tuple:
        _, _, ols_residuals = coint_res
        kf_residuals = stock_y.price_ts.loc[ols_residuals.index] - \
                   self.state_means[:, 0] * stock_x.price_ts.loc[ols_residuals.index] - \
                   self.state_means[:, 1]
        kf_hedge_ratio, kf_intercept = self.state_means[-1, :]
        return kf_residuals, kf_hedge_ratio, kf_intercept


    def update_kalman_hedge_intercept(
            self,
            last_price_x: float,
            last_price_y: float
    ) -> None:
        obs_mat = np.asarray([[last_price_x, 1]])
        new_state_means, new_state_covs = self.kf_model.filter_update(
            filtered_state_mean=self.state_means[-1],
            filtered_state_covariance=self.state_covs[-1],
            observation=last_price_y,
            observation_matrix=obs_mat
        )
        self.state_means = np.vstack((self.state_means, new_state_means))
        self.state_covs = np.concatenate((self.state_covs, new_state_covs[None, :, :]), axis=0)
        self.kf_hedge_ratio, self.kf_intercept = new_state_means
