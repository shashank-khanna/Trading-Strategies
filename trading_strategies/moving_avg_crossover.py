import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_strategies.backtesting_base import BackTestingBase


class MovingAverageCrossOver(BackTestingBase):
    SHORT_WINDOW = 55
    LONG_WINDOW = 233

    def __init__(self, ticker, look_back_days=None):
        super(MovingAverageCrossOver, self).__init__(ticker, look_back_days)

    def generate_signals(self):
        self.signals = pd.DataFrame(index=self.asset_prices.index)
        self.signals['signal'] = 0.0
        self.signals['short_mavg'] = self.asset_prices['Adj. Close'].rolling(window=self.SHORT_WINDOW, min_periods=1,
                                                                             center=False).mean()
        self.signals['long_mavg'] = self.asset_prices['Adj. Close'].rolling(window=self.LONG_WINDOW, min_periods=1,
                                                                            center=False).mean()
        self.signals['signal'][self.SHORT_WINDOW:] = np.where(self.signals['short_mavg'][self.SHORT_WINDOW:]
                                                              > self.signals['long_mavg'][self.SHORT_WINDOW:], 1.0, 0.0)
        self.signals['positions'] = self.signals['signal'].diff()

        self.signals.reset_index(inplace=True)
        self.signals.set_index(self.asset_prices.index, inplace=True)
        self.signals['Date'] = self.signals['Date'].map(mdates.date2num)

    def plot_signals(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Price in $')
        ax1.plot(self.signals.index.map(mdates.date2num), self.asset_prices['Adj. Close'])
        self.signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
        ax1.plot(self.signals.ix[self.signals.positions == 1.0].index,
                 self.signals.short_mavg[self.signals.positions == 1], "^", markersize=10, color="m")
        ax1.plot(self.signals.ix[self.signals.positions == -1.0].index,
                 self.signals.short_mavg[self.signals.positions == -1], "v", markersize=10, color="k")
        plt.show()


if __name__ == '__main__':
    strategy = MovingAverageCrossOver('AAPL')
    strategy.generate_signals()
    strategy.plot_signals()
    strategy.backtest_portfolio()
    strategy.plot_portfolio()
    strategy.sharpe_ratio()
    strategy.cagr()
