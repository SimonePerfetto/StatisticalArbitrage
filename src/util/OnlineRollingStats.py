class OnlineRollingStats:

    def __init__(self, roll_window_size, mean, stdv):
        self._roll_window_size = roll_window_size
        self._mean = mean
        self._stdv = stdv
        self._variance = stdv ** 2

    @property
    def roll_window_size(self): return self._roll_window_size

    @property
    def mean(self): return self._mean

    @mean.setter
    def mean(self, value): self._mean = value

    @property
    def stdv(self): return self._stdv

    @stdv.setter
    def stdv(self, value): self._stdv = value

    @property
    def variance(self): return self._variance

    @variance.setter
    def variance(self, value): self._variance = value

    def update(self, new, old):
        oldavg = self.mean
        newavg = oldavg + (new - old) / self.roll_window_size
        self.mean = newavg
        self.variance += (new - old) * (new - newavg + old - oldavg) / (self.roll_window_size - 1)
        self.stdv = self.variance ** 0.5
        return self.mean, self.stdv
