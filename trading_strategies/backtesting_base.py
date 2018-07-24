import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from data_fetcher import get_data


class BackTestingBase(object):
    DEFAULT_INITIAL_CAPITAL = float(100000.0)

    DEFAULT_QTY_TRADES = 100

    def __init__(self, ticker, look_back_days=None):
        self.ticker = ticker
        self.asset_prices = pd.DataFrame()
        self.signals = None
        self.portfolio = None
        self.positions = None
        self.get_underlying_data(look_back_days)

    def get_underlying_data(self, look_back_days=None):
        if look_back_days:
            start_date = datetime.datetime.today() - BDay(look_back_days)
            self.asset_prices = get_data(self.ticker, start=start_date, useQuandl=True)
        else:
            self.asset_prices = get_data(self.ticker, useQuandl=True)

    def generate_signals(self):
        raise NotImplementedError("Child class needs to implement this method.")

    def plot_signals(self):
        raise NotImplementedError("Child class needs to implement this method.")

    def _generate_positions(self):
        self.positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        self.positions[self.ticker] = self.DEFAULT_QTY_TRADES * self.signals['signal']

    def backtest_portfolio(self):
        self._generate_positions()
        print(self.positions)
        # Initialize the portfolio with value owned
        self.portfolio = self.positions.multiply(self.asset_prices['Adj. Close'], axis=0)
        # Store the difference in shares owned
        pos_diff = self.positions.diff()
        # Add `holdings` to portfolio
        self.portfolio['holdings'] = (self.positions.multiply(self.asset_prices['Adj. Close'], axis=0)).sum(axis=1)
        # Add `cash` to portfolio
        self.portfolio['cash'] = self.DEFAULT_INITIAL_CAPITAL - \
                                 (pos_diff.multiply(self.asset_prices['Adj. Close'], axis=0)).sum(axis=1).cumsum()
        self.portfolio['total'] = self.portfolio['cash'] + self.portfolio['holdings']
        self.portfolio['returns'] = self.portfolio['total'].pct_change()

    def plot_portfolio(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
        ax1.plot(self.signals.index.map(mdates.date2num), self.portfolio['total'])
        ax1.plot(self.portfolio.loc[self.signals.positions == 1.0].index,
                 self.portfolio.total[self.signals.positions == 1.0],
                 '^', markersize=10, color='m')
        ax1.plot(self.portfolio.loc[self.signals.positions == -1.0].index,
                 self.portfolio.total[self.signals.positions == -1.0],
                 'v', markersize=10, color='k')
        plt.show()

    def sharpe_ratio(self):
        # Isolate the returns of your strategy
        returns = self.portfolio['returns']
        # annualized Sharpe ratio
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        # Print the Sharpe ratio
        print("Sharpe Ratio", sharpe_ratio)
        return sharpe_ratio

    def cagr(self):
        # Compound Annual Growth Rate (CAGR)
        # Get the number of days in `aapl`
        days = (self.asset_prices.index[-1] - self.asset_prices.index[0]).days
        # Calculate the CAGR
        cagr = ((self.asset_prices['Adj. Close'][-1] / self.asset_prices['Adj. Close'][1])
                ** (365.0 / days)) - 1
        # Print the CAGR
        print("CAGR ", cagr)
        return cagr
