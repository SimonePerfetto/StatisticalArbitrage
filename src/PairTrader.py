import time
import logging
from datetime import date, timedelta, datetime
from typing import Optional

from src.Clusterer import Clusterer
from src.Cointegrator import Cointegrator, AdfPrecisions
from src.DataRepository import DataRepository
from src.Filters import Filters
from src.Portfolio import Portfolio
from src.SignalGenerator import SignalGenerator
from src.Window import Window

class PairTrader:

    def __init__(self,
                 logger: logging.Logger,
                 backtest_start: date = date(2008, 1, 2),
                 backtest_end: Optional[date] = None,
                 target_number_of_coint_pairs: int = 100,
                 max_active_pairs: int = 10,
                 clust_and_coint_frequency_per_window = 8,
                 window_length: timedelta = timedelta(days=120),
                 trading_freq: timedelta = timedelta(days=1),
                 adf_confidence_level: AdfPrecisions = AdfPrecisions.ONE_PCT,
                 max_mean_rev_time: int = 15,
                 hurst_exp_threshold: float = 0.15,
                 entry_z: float = 1.5,
                 emergency_delta_z: float = 3,
                 exit_delta_z: float = 0.5):
        # If end_date is None, run for the entirety of the dataset
        # Window is the lookback period (from t=window_length to t=0 (today) over which we analyse data
        # to inform us on trades to make on t=0 (today).
        self.logger: logging.Logger = logger
        self.backtest_start: date = backtest_start
        self.max_active_pairs: int = max_active_pairs
        self.clust_and_coint_frequency_per_window: int = clust_and_coint_frequency_per_window
        self.window_length: timedelta = window_length
        self.trading_freq: timedelta = trading_freq
        self.adf_confidence_level: AdfPrecisions = adf_confidence_level  # e.g. "5%" or "1%"
        self.max_mean_rev_time: int = max_mean_rev_time
        self.hurst_exp_threshold: float = hurst_exp_threshold
        self.entry_z: float = entry_z
        self.emergency_delta_z: float = emergency_delta_z
        self.exit_delta_z: float = exit_delta_z

        # Last SNP date, hard coded for now...
        self.backtest_end = date(year=2019, month=12, day=31) if backtest_end is None else backtest_end
        self.repository = DataRepository(self.window_length)
        self.current_window: Window = Window(window_start=backtest_start,
                                             win_len=window_length,
                                             repository=self.repository)

        self.today = self.current_window.window_business_days[-1]
        self.day_count: int = 0
        self.last_traded_date: Optional[date] = None

        self.clusterer = Clusterer()
        self.cointegrator = Cointegrator(self.repository,
                                         target_number_of_coint_pairs,
                                         self.adf_confidence_level,
                                         self.max_mean_rev_time,
                                         self.entry_z,
                                         self.exit_delta_z,
                                         previous_cointegrated_pairs=[])

        self.filters = Filters()
        self.portfolio: Portfolio = Portfolio(100_000, self.current_window)
        self.signalgenerator = SignalGenerator(self.portfolio, entry_z, exit_delta_z,
                                             emergency_delta_z, max_active_pairs)

    ######################################
    ######## define main method ##########
    ######################################
    def trade(self):
        while self.today < self.backtest_end:
            print(f"Today: {self.today.strftime('%Y-%m-%d')}\t"
                  f"Win Start: {self.current_window.window_start.strftime('%Y-%m-%d')}\t"
                  f"Win End: {self.current_window.window_end.strftime('%Y-%m-%d')}\n")

            if self.last_traded_date is None or ((self.today - self.last_traded_date).days % self.trading_freq.days == 0):

                # set a bool that is true every n days I want to cluster/cointegrate again
                is_new_clustering_and_cointegration_time = self.day_count%(self.window_length.days/
                                                                           self.clust_and_coint_frequency_per_window) == 0
                # if such bool is true, then we cluster & cointegrate using current window
                if is_new_clustering_and_cointegration_time:
                    print("Clustering...")
                    clusters = self.clusterer.dbscan(self.today, eps=0.02,
                                                     min_samples=4, window=self.current_window)
                    print("Cointegrating...")
                    cointegrated_pairs = self.cointegrator.generate_pairs(clusters,
                                                                          self.hurst_exp_threshold,
                                                                          self.current_window)

                # use already-cointegrated pairs if the bool value is False
                else:
                    cointegrated_pairs = self.cointegrator.get_previous_cointegrated_pairs(self.current_window)

                trades_to_execute_list = self.signalgenerator.make_decision(cointegrated_pairs)
                self.portfolio.execute_trades(trades_to_execute_list)
                self.last_traded_date = self.today
                print("total open count: ", self.signalgenerator.open_count_tot)
                print("day open count: ", self.signalgenerator.open_count_today)
                print("current open count: ", self.signalgenerator.open_count_current)
                print("total natural close count: ", self.signalgenerator.natural_close_count)
                print("total emergency-threshold close count: ", self.signalgenerator.emergency_close_count)
                print("total time stop-loss close count: ", self.signalgenerator.time_stop_loss_count)

            self.__evolve()

        self.portfolio.get_port_hist().to_csv('backtest_results' + self.portfolio.timestamp)
        self.portfolio.summary()
        return

    def __evolve(self):
        # Do all the things to push the window forward to next working day
        self.day_count += 1
        self.current_window.roll_forward_one_day()
        self.today = self.current_window.window_business_days[-1]
        # portfolio status (i.e., changes in price for portf companies,  subsequent unrealized
        # pnl, etc.) is updated AFTER the window is rolled one day ahead
        self.portfolio.update_portfolio(self.today)




if __name__ == '__main__':

    start_time = time.time()
    logging.basicConfig(filename='log' + datetime.now().strftime("%Y%M%d%H%M%S"),
                        filemode='a',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    global_logger = logging.getLogger(__name__)

    PairTrader(
        logger=global_logger,
        backtest_start=date(2008, 1, 2),  # must be a trading day
        window_length=timedelta(days=120),
        trading_freq=timedelta(days=1),
        target_number_of_coint_pairs=200,
        clust_and_coint_frequency_per_window=8,
        max_active_pairs=8,  # how many pairs (positions) we allow ourselves to have open at any one time
        hurst_exp_threshold=0.2,
        backtest_end=date(2019, 12, 31),
        adf_confidence_level=AdfPrecisions.ONE_PCT,
        max_mean_rev_time=15,  # we don't want any pairs that mean-revert slower than this (number larger)
        entry_z=2.0,  # how many stds away from mean the residual is, our entry signal
        exit_delta_z=1.0,  # when to close, in units of std
        emergency_delta_z=1.5  # where |emergency_z| = |entry_z + emergency_delta_z|
        # when to exit in an emergency, as each stock in the pair is deviating further from the other
    ).trade()

    print(f"Backtest took {time.time() - start_time:.4f}s to run.")


# ALMOST DONE  1) PORTFOLIO VARIANCE
# ALMOST DONE  2) VaR - t dist/normal

# 4) LOOK AHEAD VARIANCE

#fig = df.iplot(asFigure=True)
#fig.show()