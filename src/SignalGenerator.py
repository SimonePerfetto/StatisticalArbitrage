from datetime import date, timedelta
from typing import List, Tuple

from src.Cointegrator import CointegratedPair
from src.Filters import Filters
from src.Portfolio import Portfolio, Position
from src.util.Features import PositionType
from src.util.Tickers import Tickers


class SignalGenerator:

    def __init__(self,
                 port: Portfolio,
                 entry_z_lower_bound: float,
                 entry_z_upper_bound: float,
                 exit_z: float,
                 emergency_delta_z: float,
                 max_active_pairs: int):
        self.port: Portfolio = port
        self.entry_z_lower_bound: float = entry_z_lower_bound
        self.entry_z_upper_bound: float = entry_z_upper_bound
        self.exit_z: float = exit_z
        self.emergency_delta_z: float = emergency_delta_z
        self.max_active_pairs = max_active_pairs
        self.time_stop_loss = 15
        self.open_count_today = 0
        self.open_count_current = 0
        self.open_count_tot = 0
        self.natural_close_count = 0
        self.emergency_close_count = 0
        self.time_stop_loss_count = 0
        self.filter = Filters()
        self.volume_shock_filter_counter = 0

    def __exit_requirement_check(self, pos_type: PositionType, recent_dev_scaled, position_init_z):
        """
        :return: bool values for natural close and emergency close
        """
        if pos_type == PositionType.LONG:
            is_natural_close_required = recent_dev_scaled < self.exit_z
            is_emergency_close_required = recent_dev_scaled > (self.emergency_delta_z + position_init_z)
        elif pos_type == PositionType.SHORT:
            is_natural_close_required = recent_dev_scaled > -self.exit_z
            is_emergency_close_required = recent_dev_scaled < \
                                          (position_init_z - self.emergency_delta_z)

        else:
            is_natural_close_required, is_emergency_close_required = False, False
        return is_natural_close_required, is_emergency_close_required


    def entering_decision(self, coint_pair: CointegratedPair,
                          trades_to_execute_list: List[Position],
                          decision_day):
        """
        :param coint_pair: CointegratedPair object
        :param trades_to_execute_list: list of today's trades to be executed (both close and open)
        and that should therefore be ignored during this cointegration period
        :param decision_day: day in which trade is being entered
        :return: None
        check entry requirements and decide if going long, short or doing nothing;
        if any position is entered, the new position is appended in the list of
        currently-opened positions
        """
        # go long pair (i.e., long p1 short p2)
        if self.entry_z_lower_bound <coint_pair.recent_dev_scaled < self.entry_z_upper_bound:
            newpos = PositionType.LONG
            w1 = coint_pair.scaled_beta
        # go short pair (i.e., short p1 long p2)
        else:
            newpos = PositionType.SHORT
            w1 = - coint_pair.scaled_beta

        shock_bool = self.filter.run_volume_shock_filter_single_pair(coint_pair.pair,
                                                                     self.port.current_window)
        if not shock_bool:
            coint_pair.position.change_position_type(new_pos=newpos)
            coint_pair.position.weight1, coint_pair.position.weight2 = w1, 1-w1
            coint_pair.position.init_date = decision_day
            coint_pair.position.init_z = coint_pair.recent_dev_scaled
            trades_to_execute_list.append(coint_pair.position)
            self.open_count_tot += 1
            self.open_count_current += 1
            self.open_count_today += 1
        else:
            self.volume_shock_filter_counter += 1
        return

    def exiting_decision(self, position: Position, coint_pairs_tickers_list: List[Tuple[Tuple[Tickers]]],
                         coint_pairs_list: List[CointegratedPair], trades_to_execute_list: List[Position],
                         coint_wdw_already_closed_positions_list: List[CointegratedPair], decision_day: date) -> None:
        # position_pair is not an object from the CointegratedPair class; it is a tuple of the two
        # attributes asset1, asset2 of a "Position" object; thus, to inquire the statistical
        # properties of such pair, need to check the corresponding CointegratedPair object
        position_pair: Tuple[Tickers, Tickers] = (position.asset1, position.asset2)
        # 1) if pair not cointegrated, exit position
        if position_pair not in coint_pairs_tickers_list:
            position.change_position_type(PositionType.NOT_INVESTED)
            position.closingtype = "emergency"
            trades_to_execute_list.append(position)
            self.emergency_close_count += 1
            self.open_count_current -= 1
        else:
            # retrieve position_pair idx in the coint_pairs_list; the idx will be the same
            # of position_pair's corresponding CointegratedPair in the other list
            # CointegratedPair in coint_pairs_tickers_list (may want to re-think this logic)
            idx = coint_pairs_tickers_list.index(position_pair)
            coint_pair: CointegratedPair = coint_pairs_list[idx]
            # 2) if position passed time limit, exit position
            is_passed_time_limit = decision_day > (position.init_date + timedelta(self.time_stop_loss))
            if is_passed_time_limit:
                position.change_position_type(PositionType.NOT_INVESTED)
                position.closingtype = "emergency"
                trades_to_execute_list.append(position)
                coint_wdw_already_closed_positions_list.append(coint_pair)
                self.time_stop_loss_count += 1
                self.open_count_current -= 1

            # else, check if need to exit
            else:
                is_natural_close_required, is_emergency_close_required = \
                    self.__exit_requirement_check(position.new_pos,
                                                  coint_pair.recent_dev_scaled,
                                                  position.init_z)
                if is_natural_close_required or is_emergency_close_required:
                    position.change_position_type(PositionType.NOT_INVESTED)
                    if is_natural_close_required:
                        self.natural_close_count += 1
                        position.closingtype = "natural"
                    else:
                        self.emergency_close_count += 1
                        position.closingtype = "emergency"
                    trades_to_execute_list.append(position)
                    coint_wdw_already_closed_positions_list.append(position)
                    self.open_count_current -= 1
                else:
                    # no need to close, so keep the same position as before (being it SHORT or LONG)
                    pass

    def make_decision(self, coint_pairs_list: List[CointegratedPair]) -> List[Position]:
        trades_to_execute_list: List[Position] = []
        # need currently_opened_positions_list to check if we want to exit any specific position
        currently_opened_positions_list: List[Position] = self.port.cur_positions
        coint_wdw_already_closed_positions_list: List[CointegratedPair] = self.port.coint_wdw_already_closed_positions
        # need current_position-pairs to exclude them from a potential duplicate of opened position
        current_posn_pairs = [(pos.asset1, pos.asset2) for pos in currently_opened_positions_list
                              if pos.new_pos is not PositionType.NOT_INVESTED]
        today = self.port.current_window.window_end
        self.open_count_today = 0

        ####### opening new positions ##########
        # loop through cointegrated pairs
        for coint_pair in coint_pairs_list:
            # stop trading if max_active_pair reached
            if self.open_count_current >= self.max_active_pairs: break
            # if coint_pair not invested, check if we need to open position by looking at zscores
            pair_is_invested: bool = coint_pair.pair in current_posn_pairs
            pair_was_already_exploited: bool = coint_pair.pair in coint_wdw_already_closed_positions_list
            if not pair_is_invested and not pair_was_already_exploited and \
                    abs(self.entry_z_lower_bound) < abs(coint_pair.recent_dev_scaled) < abs(self.entry_z_upper_bound):
                self.entering_decision(coint_pair, trades_to_execute_list, today)
            ##########**************########## printprintprint
            if pair_is_invested: print(coint_pair.pair, coint_pair.recent_dev_scaled_hist)
            ##########**************########## printprintprint

        ####### closing current positions ##########
        # loop through all already-invested positions to check if we want to close them
        coint_pairs_tickers_list = [pa.pair for pa in coint_pairs_list]
        for position in currently_opened_positions_list:
            self.exiting_decision(position, coint_pairs_tickers_list, coint_pairs_list,
                                  trades_to_execute_list, coint_wdw_already_closed_positions_list, today)
        return trades_to_execute_list


# controlla se current_positions is updated somewhere
