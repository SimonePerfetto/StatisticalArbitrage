import unittest
from src.PairTrader import PairTrader
from datetime import date


class MyTestCase(unittest.TestCase):
    def test_first_few_trades(self):
        pairtrader = PairTrader(
            coint_window=240,
            roll_stats_window=240,
            trade_window=180,
            backtest_start_date=date(2008, 1, 2),
            max_active_pairs=20,
            num_std_away=3
        )
        pairtrader.init()
        i = 0
        dummy_list = [
            "2008-12-12: Portfolio(FreeCash: 935116, CommCapital: 64884, Good/Bad Trades:0/0, UnrealisedPnL: 0, RealisedPnL: 0, TotPnL: 0)",
            "2008-12-15: Portfolio(FreeCash: 861191, CommCapital: 138809, Good/Bad Trades:0/0, UnrealisedPnL: -46, RealisedPnL: 0, TotPnL: -46)",
            "2008-12-16: Portfolio(FreeCash: 806924, CommCapital: 193076, Good/Bad Trades:0/0, UnrealisedPnL: -1092, RealisedPnL: 0, TotPnL: -1092)",
            "2008-12-17: Portfolio(FreeCash: 951262, CommCapital: 54267, Good/Bad Trades:2/0, UnrealisedPnL: -1113, RealisedPnL: 5530, TotPnL: 4417)",
            "2008-12-18: Portfolio(FreeCash: 889004, CommCapital: 116525, Good/Bad Trades:2/0, UnrealisedPnL: 335, RealisedPnL: 5530, TotPnL: 5864)",
            "2008-12-19: Portfolio(FreeCash: 946049, CommCapital: 62258, Good/Bad Trades:3/0, UnrealisedPnL: 9, RealisedPnL: 8307, TotPnL: 8316)"
        ]

        while pairtrader.today < date(2008, 12, 19):
            pairtrader.update_pair_signals(today=pairtrader.today)
            pairtrader.portfolio.rebalance(pairtrader.today, pairtrader.coint_pairs)
            self.assertEqual(f"{pairtrader.today}: {pairtrader.portfolio}", dummy_list[i], "Unexpected mismatch.")
            pairtrader.total_pnl_dict[pairtrader.today] = pairtrader.portfolio.total_pnl
            pairtrader.n_bad_trades_dict[pairtrader.today] = pairtrader.portfolio.n_bad_trades
            pairtrader.n_good_trades_dict[pairtrader.today] = pairtrader.portfolio.n_good_trades
            pairtrader.today = pairtrader.date_manager.go_to_next_day(pairtrader.today)
            i += 1


if __name__ == '__main__':
    unittest.main()
