import unittest
from src.PairTrader import PairTrader
from datetime import date


class MyTestCase(unittest.TestCase):
    def test_first_few_trades(self):
        pairtrader = PairTrader(coint_window=240, roll_stats_window=240, trade_window=180,
                                backtest_start_date=date(2008, 1, 2), max_active_pairs=20, num_std_away=3)
        pairtrader.init()
        i = 0
        dummy_list = [
            "2008-12-12: Portfolio(FreeCash: 919426, CommCapital: 80574, Good/Bad Trades:0/0, UnrealisedPnL: 0, RealisedPnL: 0, TotPnL: 0)",
            "2008-12-15: Portfolio(FreeCash: 845502, CommCapital: 154498, Good/Bad Trades:0/0, UnrealisedPnL: -274, RealisedPnL: 0, TotPnL: -274)",
            "2008-12-16: Portfolio(FreeCash: 791235, CommCapital: 208765, Good/Bad Trades:0/0, UnrealisedPnL: -1777, RealisedPnL: 0, TotPnL: -1777)",
            "2008-12-17: Portfolio(FreeCash: 868381, CommCapital: 134841, Good/Bad Trades:1/0, UnrealisedPnL: -4282, RealisedPnL: 3222, TotPnL: -1060)",
            "2008-12-18: Portfolio(FreeCash: 806123, CommCapital: 197099, Good/Bad Trades:1/0, UnrealisedPnL: -3878, RealisedPnL: 3222, TotPnL: -657)",
            "2008-12-19: Portfolio(FreeCash: 776709, CommCapital: 229290, Good/Bad Trades:2/0, UnrealisedPnL: -2953, RealisedPnL: 5999, TotPnL: 3046)"
        ]

        while pairtrader.today < date(2008, 12, 19):
            pairtrader.update_pair_signals(pairtrader.today)
            pairtrader.portfolio.rebalance(pairtrader.today, pairtrader.coint_pairs)
            self.assertEqual(f"{pairtrader.today}: {pairtrader.portfolio}", dummy_list[i], "Unexpected mismatch.")
            pairtrader.total_pnl_dict[pairtrader.today] = pairtrader.portfolio.total_pnl
            pairtrader.n_bad_trades_dict[pairtrader.today] = pairtrader.portfolio.n_bad_trades
            pairtrader.n_good_trades_dict[pairtrader.today] = pairtrader.portfolio.n_good_trades
            pairtrader.today = pairtrader.date_manager.go_to_next_day(pairtrader.today)
            i += 1


if __name__ == '__main__':
    unittest.main()
