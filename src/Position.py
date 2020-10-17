from pandas import DataFrame, to_datetime
from datetime import datetime
from src.util.Features import PositionType
from src.util.Tickers import Tickers


class Position:
    def set_position_value(self, value):
        self.init_value = value
        self.current_value = value

    def __init__(self,
                 ticker1,
                 ticker2,
                 new_pos: PositionType = PositionType.NOT_INVESTED,
                 old_pos: PositionType = PositionType.NOT_INVESTED,
                 weight2=0,
                 init_date=None,
                 init_z=None,
                 weight1=0,
                 ):
        self.asset1: Tickers = ticker1
        self.asset2: Tickers = ticker2
        self.old_pos: PositionType = old_pos
        self.new_pos: PositionType = new_pos
        self.weight1: float = weight1
        self.weight2: float = weight2
        self.init_date: datetime = init_date
        self.init_z: float = init_z
        self.init_value: float
        self.quantity1: float = 0
        self.quantity2: float = 0
        self.simple_return: float = 0
        self.commission: float = 0
        self.pnl: float = 0
        self.current_value: float = 0
        self.pos_hist = list()
        self.closed = False

    def update_weight(self, asset1_val, asset2_val):
        self.weight1 = asset1_val / (asset1_val + asset2_val)
        self.weight2 = asset2_val / (asset1_val + asset2_val)

    def change_position_type(self, new_pos):
        self.old_pos = self.new_pos
        self.new_pos = new_pos

    def update_position_pnl(self, value, window):
        if not self.closed:
            self.pnl += value - self.current_value
            self.simple_return = value / self.current_value - 1
            self.current_value = value
        self.pos_hist.append([window.window_end, self.current_value, self.pnl, self.simple_return])

    def rebalance_pos(self, new_weights, rebalance_value):
        self.weight1 = new_weights[0]
        self.weight2 = new_weights[1]
        self.pnl -= self.commission
        self.current_value += rebalance_value

    def close_trade(self, value, window):
        self.pnl += value - self.current_value - self.commission
        self.current_value = value
        self.closed = True
        self.pos_hist.append([window.window_end, self.current_value, self.pnl])

    def get_pos_hist(self):
        # returns a time series of position value and pnl
        df = DataFrame(self.pos_hist, columns=['date', 'pos_value', 'pnl', 'return'])
        df['date'] = to_datetime(df['date'])
        df = df.set_index('date')
        return df.round(2)
