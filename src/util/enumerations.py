from enum import Enum

class TradingAction(Enum):
    OpenLong = 0
    OpenShort = 1
    CloseLong = 2
    CloseShort = 3
    HoldLong = 4
    HoldShort = 5
    Pass = 6