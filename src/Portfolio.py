from datetime import date
import numpy as np
from typing import Tuple, List, Dict
from src.Cointegrator import CointPair
from src.TradedPair import LongLeg, ShortLeg, TradedPair
from src.util.enumerations import TradingAction

class Portfolio:

    def __init__(
            self,
            max_active_pairs: int,
            cash: float = 1_000_000.0
    ):
        self.max_active_pairs: int = max_active_pairs
        self.pf_free_cash: float = cash
        self.pf_locked_cash: float = 0
        self.pf_committed_capital: float = 0
        self.outstanding_pnl_dict = {}
        self.realised_pnl: float = 0
        self.current_pnl: float = 0
        self.previous_pnl: float = 0
        self.total_pnl: float = 0
        self.current_holdings: Dict[CointPair, TradedPair] = {}
        self.t_fee: float = 0.0005
        self.n_good_trades: int = 0
        self.n_bad_trades: int = 0

    def __repr__(self):
        return f"Portfolio(FreeCash: {round(self.pf_free_cash)}, " \
               f"CommCapital: {round(self.pf_committed_capital)}, " \
               f"Good/Bad Trades:{self.n_good_trades}/{self.n_bad_trades}, " \
               f"UnrealisedPnL: {round(self.current_pnl)}, " \
               f"RealisedPnL: {round(self.realised_pnl)}, " \
               f"TotPnL: {round(self.total_pnl)})"

    def rebalance(
            self,
            today: date,
            coint_pairs: List[CointPair]
    ) -> None:
        for coint_pair in coint_pairs:
            trade_action = self.formulate_trade_action_from_signal(coint_pair=coint_pair)
            if trade_action in (TradingAction.OpenLong, TradingAction.OpenShort) and self.is_reached_max_n_active_pairs():
                coint_pair.override_signal(signal_value=0)
                trade_action = TradingAction.Pass
            self.execute_trade_action(
                coint_pair=coint_pair,
                today=today,
                trade_action=trade_action
            )

        self.total_pnl = self.current_pnl + self.realised_pnl

    @staticmethod
    def formulate_trade_action_from_signal(
            coint_pair: CointPair
    ) -> TradingAction:
        prev_sign, curr_sign = \
            coint_pair.signal_builder.get_penultimate_signal(), coint_pair.signal_builder.get_last_signal()
        if curr_sign == 1 and prev_sign == 0:
            return TradingAction.OpenLong
        elif curr_sign == -1 and prev_sign == 0:
            return TradingAction.OpenShort
        elif curr_sign == 0 and prev_sign == -1:
            return TradingAction.CloseShort
        elif curr_sign == 0 and prev_sign == 1:
            return TradingAction.CloseLong
        elif curr_sign == 0 and prev_sign == 0:
            return TradingAction.Pass
        elif curr_sign == 1 and prev_sign == 1:
            return TradingAction.HoldLong
        elif curr_sign == -1 and prev_sign == -1:
            return TradingAction.HoldShort
        else:
            raise ValueError(f"Unexpected combination of previous signal: {prev_sign}, current signal: {curr_sign}")

    def execute_trade_action(
            self,
            coint_pair: CointPair,
            today: date,
            trade_action: str,
            plot=False
    ) -> None:
        if trade_action in (TradingAction.OpenLong, TradingAction.OpenShort):
            traded_pair = self.set_traded_pair(
                coint_pair=coint_pair,
                today=today,
                trade_action=trade_action
            )
            self.update_portfolio_data(traded_pair=traded_pair)
            self.insert_in_holdings(
                coint_pair=coint_pair,
                traded_pair=traded_pair
            )
            if plot:
                coint_pair.plot_residuals_and_bb_bands(trade_action=trade_action)

        elif trade_action in (
                TradingAction.CloseLong, TradingAction.CloseShort, TradingAction.HoldLong, TradingAction.HoldShort
        ):
            traded_pair = self.get_traded_pair_from_holdings(coint_pair=coint_pair)
            self.update_traded_pair(
                traded_pair=traded_pair,
                coint_pair=coint_pair,
                today=today,
                trade_action=trade_action
            )
            self.update_portfolio_current_pnl(traded_pair=traded_pair)

            if trade_action in (TradingAction.CloseLong, TradingAction.CloseShort):
                self.update_portfolio_realized_pnl(closing_pair=traded_pair)
                self.remove_from_holdings(coint_pair=coint_pair)
                if plot:  # and 'ADBE' in coint_pair.get_ticker_x_y():  # and traded_pair.pair_current_holding_pnl < 0:
                    coint_pair.plot_residuals_and_bb_bands(trade_action=trade_action)

        elif trade_action == TradingAction.Pass:
            """ do nothing """
        else:
            raise ValueError(f"Trade action: {trade_action} not valid.")

    def set_traded_pair(
            self,
            coint_pair: CointPair,
            today: date,
            trade_action: str
    ) -> TradedPair:
        px, py = coint_pair.get_todays_price_x_y(today=today)
        nx, ny = self._units_finder(
            py=py,
            hedge_ratio=coint_pair.hedge_ratio
        )
        ticker_x, ticker_y = coint_pair.get_ticker_x_y()

        if trade_action == TradingAction.OpenLong:  # long pair means buy 1 unit of y, sell hedgeratio units of x
            long_ticker, long_price, long_units = ticker_y, py, ny
            short_ticker, short_price, short_units = ticker_x, px, nx

        else:  # short pair means sell 1 unit of y, buy hedgeratio units of x
            long_ticker, long_price, long_units = ticker_x, px, nx
            short_ticker, short_price, short_units = ticker_y, py, ny

        long_leg = LongLeg(
            ticker=long_ticker,
            price=long_price,
            units=long_units,
            transaction_fee=self.t_fee
        )
        short_leg = ShortLeg(
            ticker=short_ticker,
            price=short_price,
            units=short_units,
            transaction_fee=self.t_fee
        )
        return TradedPair(
            long_leg=long_leg,
            short_leg=short_leg
        )

    @staticmethod
    def update_traded_pair(
            traded_pair: TradedPair,
            coint_pair: CointPair,
            today: date,
            trade_action: str
    ) -> None:

        px, py = coint_pair.get_todays_price_x_y(today=today)
        is_pos_closing = True if trade_action in (TradingAction.CloseLong, TradingAction.CloseShort) else False
        # long (short) pair is long (short) one unit of y, short (long) hedgeratio units of x
        long_price, short_price = (py, px) if trade_action in (TradingAction.HoldLong, TradingAction.CloseLong) else (px, py)

        traded_pair.update_legs(
            long_price=long_price,
            short_price=short_price,
            is_pos_closing=is_pos_closing
        )
        traded_pair.update_traded_pair_pnl()

    def update_portfolio_data(
            self,
            traded_pair: TradedPair
    ) -> None:
        self._update_pf_committed_capital(traded_pair=traded_pair)
        self._update_pf_free_cash(traded_pair=traded_pair)
        self._update_pf_locked_cash(traded_pair=traded_pair)

    def update_portfolio_current_pnl(
            self,
            traded_pair: TradedPair
    ) -> None:
        self.current_pnl += (traded_pair.pair_current_holding_pnl - traded_pair.pair_prev_holding_pnl)

    def update_portfolio_realized_pnl(
            self,
            closing_pair: TradedPair
    ) -> None:
        closing_pair_pnl = closing_pair.pair_current_holding_pnl
        self.realised_pnl += closing_pair_pnl
        self.current_pnl -= closing_pair_pnl
        self.pf_locked_cash -= closing_pair.locked_cash
        self.pf_free_cash += closing_pair.tot_committed_capital + closing_pair_pnl
        self.pf_committed_capital -= closing_pair.tot_committed_capital
        if closing_pair_pnl > 0: self.n_good_trades += 1
        else: self.n_bad_trades += 1

    def insert_in_holdings(
            self,
            coint_pair: CointPair,
            traded_pair: TradedPair
    ) -> None:
        self.current_holdings[coint_pair] = traded_pair

    def get_traded_pair_from_holdings(
            self,
            coint_pair: CointPair
    ) -> TradedPair:
        return self.current_holdings[coint_pair]

    def remove_from_holdings(
            self,
            coint_pair: CointPair
    ) -> None:
        del self.current_holdings[coint_pair]

    def _update_pf_committed_capital(
            self,
            traded_pair: TradedPair
    ) -> None:
        self.pf_committed_capital += traded_pair.tot_committed_capital

    def _update_pf_free_cash(
            self,
            traded_pair: TradedPair
    ) -> None:
        self.pf_free_cash -= traded_pair.tot_committed_capital

    def _update_pf_locked_cash(
            self,
            traded_pair: TradedPair
    ) -> None:
        self.pf_locked_cash += traded_pair.locked_cash


    @staticmethod
    def _units_finder(
            py: float,
            hedge_ratio: float,
            min_notional: float = 48_000,
            max_notional: float = 52_000
    ) -> Tuple:
        notionals = np.linspace(
            start=min_notional,
            stop=max_notional,
            num=101
        )
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



