import numpy as np
from pykalman import KalmanFilter
from typing import Tuple
import pandas as pd

class KalmanUtils:

    def __init__(self, coint_res, stock_x, stock_y):
        self.kf_model, self.state_means, self.state_covs = self.build_kalman_features(coint_res, stock_x, stock_y)
        self.kf_residuals, self.kf_hedge_ratio, self.kf_intercept = self.get_kalman_results(coint_res, stock_x, stock_y)

    @staticmethod
    def build_kalman_features(coint_res, stock_x, stock_y) -> Tuple:
        ols_hedge_ratio, ols_intercept, ols_residuals = coint_res
        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.vstack([stock_x.window_prices.iloc[:len(ols_residuals)],
                             np.ones(len(ols_residuals))]).T[:, np.newaxis]

        kf_model = KalmanFilter(n_dim_obs=1, n_dim_state=2, initial_state_mean=(ols_hedge_ratio, ols_intercept),
                                # initial_state_mean=np.zeros(2),
                                initial_state_covariance=np.ones((2, 2)), transition_matrices=np.eye(2),
                                observation_matrices=obs_mat, observation_covariance=1.0,
                                transition_covariance=trans_cov)

        state_means, state_covs = kf_model.filter(stock_y.window_prices.iloc[:len(ols_residuals)].values)
        return kf_model, state_means, state_covs

    def get_kalman_results(self, coint_res, stock_x, stock_y) -> Tuple:
        _, _, ols_residuals = coint_res
        kf_residuals_np = stock_y.window_prices.values[:len(ols_residuals)] - \
                          self.state_means[:, 0] * stock_x.window_prices.values[:len(ols_residuals)] - \
                          self.state_means[:, 1]
        kf_residuals = pd.Series(kf_residuals_np, index=ols_residuals.index)
        kf_hedge_ratio, kf_intercept = self.state_means[-1, :]
        return kf_residuals, kf_hedge_ratio, kf_intercept


    def update_kalman_hedge_intercept(self, last_price_x, last_price_y) -> None:
        obs_mat = np.asarray([[last_price_x, 1]])
        new_state_means, new_state_covs = self.kf_model.filter_update(self.state_means[-1], self.state_covs[-1],
                                                                      observation=last_price_y,
                                                                      observation_matrix=obs_mat)
        self.state_means = np.vstack((self.state_means, new_state_means))
        self.state_covs = np.concatenate((self.state_covs, new_state_covs[None, :, :]), axis=0)
        self.kf_hedge_ratio, self.kf_intercept = new_state_means
