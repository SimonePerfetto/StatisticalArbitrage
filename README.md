# Statistical Arbitrage - US Equity Pair Trading - Backtest Engine

The Project aims at simulating a long-short, dollar-neutral pair trading strategy on a US equity portfolio.

## Rationale

The idea behind trading a pair of stocks relies on the fact that prices of certain pairs "move together" in the long run, as they are exposed to common risk factors. A classical example is public companies operating in the same sector and sharing a similar business model. In the short term, prices of similar stocks can temporarily deviate one from the other, before eventually converging again. A simple trading strategy would therefore exploit this temporary divergence by going long on the relatively underpriced stock and short on the relatively overpriced one, closing both positions when this gap shrinks.

There are different ways of evaluating the "similarity" of two stocks, with the important caveat that a high degree of similarity should imply the desirable property of long-term "co-movement" of the two corresponding price time series. In turn, the "co-movement" property can be exploited in the trading strategy.

## Cointegration

In this project, I am using the statistical concept of cointegration to evaluate the co-movement property. Loosely speaking, two stochastic, non-stationary time series are cointegrated if there exists a linear combination of such time series that is stationary. Stationarity is a desirable property for a stochastic process, as it implies that the mean and standard deviation (and higher moments) of the underlying probability distribution are constant over time. 

The Engle-Granger approach to test the cointegration of two time series relies on two steps. 
1. Estimate the cointegrating factor β via Ordinary Least Square of one time series against the other (arbitrarily choosing which one is acting as "y" vs which one is acting as "x")
2. Testing stationarity of the OLS residuals via Augmented Dickey-Fueller test

If the ADF test rejects the null hypothesis of non-stationarity of OLS residuals with some degree of confidence, then we can treat the two price time series as cointegrated and build a trading strategy based on this.
To corroborate the cointegration property of two stocks with an economic rational, I am only testing pair of stocks from the same sector.

Below an example of two (normalized) price time series that pass the cointegration test, ADBE (Adobe Inc.) and ANSS (Ansys). Both companies develop software for product design.


![](https://github.com/SimonePerfetto/StatisticalArbitrage/blob/master/src/images/cointpair.svg)

## Trading Strategy

Once a cointegrated pair is found at a given point in time t, I compute the new residual from the cointegration relationship at each time s > t. Moreover, I build Bollinger Bands for the time series of residuals. Upper and Lower Bollinger bands will serve as thresholds to initiate trading action. Specifically:

1. if  new residual is above the upper band, go short the pair (i.e. short 1 unit of y, buy β units of x)
2. if  new residual is below the lower band, go long the pair (i.e. buy 1 unit of y, sell β units of x)

The exit from an already active long (short) pair, i.e., the liquidation of the positions in both long and short leg, is triggered when the newest residual overcome the moving average of residuals from below (above).

The actual code uses a more sophisticated logic exploiting some smoothing properties of a Kalman filter, but the core idea is well summarized above.

Below a plot with time series of residuals, moving average and Bollinger bands.

![](https://github.com/SimonePerfetto/StatisticalArbitrage/blob/master/src/images/res.svg)


## Backtest
The backtest below is performed on a portfolio of (max) 20 pairs dynamically entered / exited over time. Entry/Exit transaction costs assumed at 5 bps per trade, initial notional of USD 1 MM. Cointegration test is performed at fixed intervals over time, and there is a fixed trading window period to trade based on the previous cointegration information. More details in the code.

![](https://github.com/SimonePerfetto/StatisticalArbitrage/blob/master/src/images/backtest.svg)

## Final remarks
The Project is ongoing and I might implement new features in the future and / or refactor some of the classes used. I hope you enjoyed the read!
