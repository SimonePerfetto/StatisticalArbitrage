from datetime import date
import numpy as np
from typing import Tuple, List
from src.Cointegrator import CointPair

from src.TradedPair import LongLeg, ShortLeg, TradedPair


class Portfolio:

    def __init__(self, max_active_pairs: int, cash: float = 1_000_000.0):
        self.max_active_pairs: int = max_active_pairs
        self.pf_free_cash: float = cash
        self.pf_locked_cash: float = 0
        self.pf_committed_capital: float = 0
        self.outstanding_pnl_dict = {}
        self.realised_pnl: float = 0
        self.current_pnl: float = 0
        self.previous_pnl: float = 0
        self.total_pnl: float = 0
        self.current_holdings: dict = {}
        self.t_fee: float = 0.0005
        self.n_good_trades: int = 0
        self.n_bad_trades: int = 0

    def __repr__(self):
        #CurrCash: {self.pf_free_cash}, CommCapital: {self.pf_committed_capital},
        return f"Portfolio(FreeCash: {round(self.pf_free_cash)}, " \
               f"CommCapital: {round(self.pf_committed_capital)}, " \
               f"Good/Bad Trades:{self.n_good_trades}/{self.n_bad_trades}, " \
               f"UnrealisedPnL: {round(self.current_pnl)}, " \
               f"RealisedPnL: {round(self.realised_pnl)}, " \
               f"TotPnL: {round(self.total_pnl)})"

    def rebalance(self, today, coint_pairs):
        for coint_pair in coint_pairs:
            trade_action = self.formulate_trade_action_from_signal(coint_pair)
            if trade_action in ("OpenLong", "OpenShort") and self.is_reached_max_n_active_pairs():
                coint_pair.set_signal(signal_value=0)
                trade_action = "Pass"
            self.execute_trade_action(coint_pair, today, trade_action)

        self.total_pnl = self.current_pnl + self.realised_pnl

    @staticmethod
    def formulate_trade_action_from_signal(coint_pair) -> str:
        prev_sign, curr_sign = coint_pair.get_penultimate_signal(), coint_pair.get_last_signal()
        if curr_sign == 1 and prev_sign == 0:
            return "OpenLong"
        elif curr_sign == -1 and prev_sign == 0:
            return "OpenShort"
        elif curr_sign == 0 and prev_sign == -1:
            return "CloseShort"
        elif curr_sign == 0 and prev_sign == 1:
            return "CloseLong"
        elif curr_sign == 0 and prev_sign == 0:
            return "Pass"
        elif curr_sign == 1 and prev_sign == 1:
            return "HoldLong"
        elif curr_sign == -1 and prev_sign == -1:
            return "HoldShort"
        else:
            raise ValueError(f"Unexpected combination of previous signal: {prev_sign}, current signal: {curr_sign}")

    def execute_trade_action(self, coint_pair: CointPair, today: date, trade_action: str, plot=False) -> None:
        if trade_action in ("OpenLong", "OpenShort"):
            traded_pair = self.set_traded_pair(coint_pair, today, trade_action)
            self.update_portfolio_data(traded_pair)
            self.insert_in_holdings(coint_pair, traded_pair)
            if plot: coint_pair.plot_residuals_and_bb_bands(trade_action)

        elif trade_action in ("CloseLong", "CloseShort", "HoldLong", "HoldShort"):
            traded_pair = self.get_traded_pair_from_holdings(coint_pair)
            self.update_traded_pair(traded_pair, coint_pair, today, trade_action)
            self.update_portfolio_current_pnl(traded_pair, trade_action)

            if trade_action in ("CloseLong", "CloseShort"):
                self.update_portfolio_realized_pnl(closing_pair=traded_pair)
                self.remove_from_holdings(coint_pair)
                if plot: coint_pair.plot_residuals_and_bb_bands(trade_action)

        elif trade_action == "Pass":
            """ do nothing """
        else:
            raise ValueError(f"Trade action: {trade_action} not valid.")

    def set_traded_pair(self, coint_pair: CointPair, today: date, trade_action: str) -> TradedPair:
        hedge_ratio = coint_pair.get_hedge_ratio()
        px, py = coint_pair.get_todays_price_x_y(today)
        nx, ny = self.__units_finder(py, hedge_ratio)
        ticker_x, ticker_y = coint_pair.get_ticker_x_y()

        if trade_action == "OpenLong":  # long pair means buy 1 unit of y, sell hedgeratio units of x
            long_ticker, long_price, long_units = ticker_y, py, ny
            short_ticker, short_price, short_units = ticker_x, px, nx

        else:  # short pair means sell 1 unit of y, buy hedgeratio units of x
            long_ticker, long_price, long_units = ticker_x, px, nx
            short_ticker, short_price, short_units = ticker_y, py, ny

        long_leg = LongLeg(long_ticker, long_price, long_units, self.t_fee)
        short_leg = ShortLeg(short_ticker, short_price, short_units, self.t_fee)
        return TradedPair(long_leg=long_leg, short_leg=short_leg)

    @staticmethod
    def update_traded_pair(traded_pair: TradedPair, coint_pair: CointPair, today: date, trade_action: str) -> None:

        px, py = coint_pair.get_todays_price_x_y(today)
        is_pos_closing = True if trade_action in ("CloseLong", "CloseShort") else False
        # long (short) pair is long (short) one unit of y, short (long) hedgeratio units of x
        long_price, short_price = (py, px) if trade_action in ("HoldLong", "CloseLong") else (px, py)

        traded_pair.update_legs(long_price=long_price, short_price=short_price, is_pos_closing=is_pos_closing)
        traded_pair.update_traded_pair_pnl()

    def update_portfolio_data(self, traded_pair) -> None:
        self.__update_pf_committed_capital(traded_pair)
        self.__update_pf_free_cash(traded_pair)
        self.__update_pf_locked_cash(traded_pair)

    def update_portfolio_current_pnl(self, traded_pair: TradedPair, trade_action: str) -> None:
        self.current_pnl += (traded_pair.pair_current_holding_pnl - traded_pair.pair_prev_holding_pnl)

    def update_portfolio_realized_pnl(self, closing_pair: TradedPair) -> None:
        closing_pair_pnl = closing_pair.pair_current_holding_pnl
        self.realised_pnl += closing_pair_pnl
        self.current_pnl -= closing_pair_pnl
        self.pf_locked_cash -= closing_pair.locked_cash
        self.pf_free_cash += closing_pair.tot_committed_capital + closing_pair_pnl
        self.pf_committed_capital -= closing_pair.tot_committed_capital
        if closing_pair_pnl > 0: self.n_good_trades += 1
        else: self.n_bad_trades += 1

    def insert_in_holdings(self, coint_pair: CointPair, traded_pair: TradedPair) -> None:
        self.current_holdings[coint_pair] = traded_pair

    def get_traded_pair_from_holdings(self, coint_pair: CointPair) -> TradedPair:
        return self.current_holdings[coint_pair]

    def remove_from_holdings(self, coint_pair: CointPair) -> None:
        del self.current_holdings[coint_pair]

    def __update_pf_committed_capital(self, traded_pair) -> None:
        self.pf_committed_capital += traded_pair.tot_committed_capital

    def __update_pf_free_cash(self, traded_pair) -> None:
        self.pf_free_cash -= traded_pair.tot_committed_capital

    def __update_pf_locked_cash(self, traded_pair) -> None:
        self.pf_locked_cash += traded_pair.locked_cash


    @staticmethod
    def __units_finder(py, hedge_ratio, min_notional=48_000, max_notional=52_000) -> Tuple:
        notionals = np.linspace(min_notional, max_notional, 101)
        num_ys = np.round(notionals / py)
        num_xs = np.round(num_ys * hedge_ratio)
        abs_errors = np.abs(num_xs / num_ys - hedge_ratio)
        idx_min_error = np.argmin(abs_errors)
        notional_w_min_err = notionals[idx_min_error]
        ny_star = round(notional_w_min_err / py)
        nx_star = round(ny_star * hedge_ratio)
        return nx_star, ny_star

    def is_reached_max_n_active_pairs(self) -> bool:
        return len(self.current_holdings) >= self.max_active_pairs



