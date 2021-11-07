class OnlineRollingStats:

    def __init__(self, window_size, mean, stdv):
        self.window_size = window_size
        self.mean = mean
        self.stdv = stdv
        self.variance = stdv ** 2

    def update(self, new, old):
        oldavg = self.mean
        newavg = oldavg + (new - old) / self.window_size
        self.mean = newavg
        self.variance += (new - old) * (new - newavg + old - oldavg) / (self.window_size - 1)
        self.stdv = self.variance ** 0.5
        return self.mean, self.stdv
