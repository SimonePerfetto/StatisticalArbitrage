from abc import ABC, abstractmethod
from typing import Tuple


class Leg(ABC):
    def __init__(self, ticker: str, price: float, units: int, transaction_fee: float):
        self.ticker: str = ticker
        self.price: float = price
        self.units: int = units
        self.transaction_fee: float = transaction_fee
        self.prev_holding_pnl: float = 0
        self.current_holding_pnl: float = 0

    def adjust_pnl_for_entry_transaction_costs(self) -> None:
        self.prev_holding_pnl = self.current_holding_pnl
        self.current_holding_pnl = - self.transaction_fee * (self.price * self.units)

    @abstractmethod
    def adjust_capital(self):
        """ different behaviour for LongLeg and ShortLeg """

    @abstractmethod
    def update_leg_price_and_pnl(self, new_price, is_pos_closing):
        """ different behaviour for LongLeg and ShortLeg """


class LongLeg(Leg):
    def __init__(
            self,
            ticker: str,
            price:
            float,
            units: int,
            transaction_fee: float
    ):
        super().__init__(
            ticker=ticker,
            price=price,
            units=units,
            transaction_fee=transaction_fee
        )
        self.invested_capital = self.adjust_capital()
        self.adjust_pnl_for_entry_transaction_costs()

    def adjust_capital(self) -> float:
        return self.price * self.units

    def update_leg_price_and_pnl(
            self,
            new_price: float,
            is_pos_closing: bool
    ) -> None:
        self.prev_holding_pnl = self.current_holding_pnl
        self.current_holding_pnl += self.units * (new_price - self.price)
        if is_pos_closing:
            self.current_holding_pnl -= self.transaction_fee * (self.units * new_price)
        self.price = new_price

    def __repr__(self):
        return f"Long({self.ticker})"


class ShortLeg(Leg):
    def __init__(
            self, ticker: str,
            price: float,
            units: int,
            transaction_fee: float
    ):
        super().__init__(
            ticker=ticker,
            price=price,
            units=units,
            transaction_fee=transaction_fee
        )
        self.locked_cash, self.posted_margin = self.adjust_capital()
        self.adjust_pnl_for_entry_transaction_costs()

    def adjust_capital(self) -> Tuple:
        notional = self.price * self.units
        locked_cash, posted_margin = 1.5 * notional, 0.5 * notional
        return locked_cash, posted_margin

    def update_leg_price_and_pnl(
            self,
            new_price: float,
            is_pos_closing: bool
    ) -> None:
        self.prev_holding_pnl = self.current_holding_pnl
        self.current_holding_pnl -= self.units * (new_price - self.price)
        if is_pos_closing:
            self.current_holding_pnl -= self.units * new_price * self.transaction_fee
        self.price = new_price

    def __repr__(self):
        return f"Short({self.ticker})"


class TradedPair:
    def __init__(self, long_leg: LongLeg, short_leg: ShortLeg):
        self.long_leg: LongLeg = long_leg
        self.short_leg: ShortLeg = short_leg
        self.pair_prev_holding_pnl: float = 0
        self.pair_current_holding_pnl: float = 0
        self.invested_capital = long_leg.invested_capital
        self.locked_cash = short_leg.locked_cash
        self.posted_margin = short_leg.posted_margin
        self.tot_committed_capital = self.invested_capital + self.posted_margin

    def update_legs(
            self,
            long_price: float,
            short_price: float,
            is_pos_closing: bool
    ) -> None:
        self.long_leg.update_leg_price_and_pnl(new_price=long_price, is_pos_closing=is_pos_closing)
        self.short_leg.update_leg_price_and_pnl(new_price=short_price, is_pos_closing=is_pos_closing)

    def update_traded_pair_pnl(self) -> None:
        self.pair_prev_holding_pnl = self.pair_current_holding_pnl
        self.pair_current_holding_pnl += (self.long_leg.current_holding_pnl - self.long_leg.prev_holding_pnl) + \
                                         (self.short_leg.current_holding_pnl - self.short_leg.prev_holding_pnl)

    def __repr__(self):
        return f"TradedPair[{self.long_leg}, {self.short_leg}]"
