import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_strategies.backtesting_base import BackTestingBase


class RandomStrategy(BackTestingBase):

    def __init__(self, ticker, look_back_days=None):
        super(RandomStrategy, self).__init__(ticker, look_back_days)

    def generate_signals(self):
        self.signals = pd.DataFrame(index=self.asset_prices.index)
        self.signals['signal'] = 0.0
        self.signals['price'] = self.asset_prices['Adj. Close']
        self.signals['signal'] = np.sign(np.random.randn(len(self.signals)))
        # The first five elements are set to zero in order to minimise
        # upstream NaN errors in the forecaster.
        self.signals['signal'][0:5] = 0.0
        self.signals['positions'] = self.signals['signal']
        self.signals.reset_index(inplace=True)
        self.signals.set_index(self.asset_prices.index, inplace=True)
        self.signals['Date'] = self.signals['Date'].map(mdates.date2num)
        print(self.signals)

    def plot_signals(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Price in $')
        ax1.plot(self.signals.index.map(mdates.date2num), self.asset_prices['Adj. Close'])
        ax1.plot(self.signals.ix[self.signals.positions == 1.0].index,
                 self.signals['price'][self.signals.positions == 1], "^", markersize=2, color="m")
        ax1.plot(self.signals.ix[self.signals.positions == -1.0].index,
                 self.signals['price'][self.signals.positions == -1], "v", markersize=2, color="k")
        plt.show()


if __name__ == '__main__':
    strategy = RandomStrategy('TSLA')
    strategy.generate_signals()
    strategy.plot_signals()
    strategy.backtest_portfolio()
    strategy.plot_portfolio()
    strategy.sharpe_ratio()
    strategy.cagr()
