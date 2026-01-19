import backtrader as bt

from bt.ema_indicator import EmaIndicator
from bt.turn_indicator import TurnIndicator


class TurnStrategy(bt.Strategy):
    params = dict(
        fast_period=10,
        symbol = '159857.SZ'
    )

    def __init__(self):
        self.mySma = EmaIndicator(self.data, period=self.p.fast_period)
        self.myTurning = TurnIndicator(symbol=self.p.symbol)
        self.ma = bt.indicators.SimpleMovingAverage(self.data, period=self.p.fast_period)

        self.market = None
        # self.trader = StockTrader(self.cfg, self.market)
        self.pos = None
        self.trade_sum = 0
        pass

    def next(self):
        return

    """
        timestamp = self.data.datetime.datetime(0).strftime('%Y-%m-%d %H:%M:%S')
        bar = BarData('', self.data.datetime.datetime(0), self.data.open[0], self.data.close[0], self.data.volume[0])
        self.update_pos(timestamp)
        if timestamp.endswith('15:00:00'):
            self.market.pre(bar)
            self.trader.prepare(self.pos)
            return
        self.market.next(bar)

        # 更新仓位

        if timestamp == '2025-12-05 14:00:00':
            print()

        # 补仓
        if timestamp.endswith('14:55:00'):
            supp_amount = self.trader.supp_amount(self.pos)
            if supp_amount > 0:
                print()
            self.__trade(timestamp, supp_amount)
            return

        # 加仓减仓
        amount = self.trader.amount(self.pos)
        self.__trade(timestamp, amount)

    def __trade(self, timestamp, amount):
        if amount == 0:
            return
        self.trade_sum += 1

        # 买入
        if amount > 0:
            self.buy(size=amount)
            if self.pos is None:
                self.pos = Position(self.symbol, 0, amount, round(self.data.close[0], 3))
            else:
                self.pos.amount += amount
            self.print_pos(timestamp, f'买入={amount}')

        # 卖出
        if amount < 0:
            self.sell(size=-amount)
            self.pos.amount += amount
            self.pos.enable_amount += amount
            self.print_pos(timestamp, f'卖出={amount}')

    def update_pos(self, timestamp):
        if self.pos is None:
            return
        self.pos.last_sale_price = round(self.data.close[0], 3)
        if timestamp.endswith('09:30:00'):
            print('=====================================================================')
            self.pos.enable_amount = self.pos.amount
            self.print_pos(timestamp, '开盘后')

    def stop(self):
        print(f'交易次数={self.trade_sum},市值={self.pos.amount * self.pos.last_sale_price}')
        pass

    def print_pos(self, timestamp, msg):
        print(f'{timestamp}: {msg}， 总量={self.pos.amount}, 可用={self.pos.enable_amount}, '
              f'仓位={round(self.pos.last_sale_price * self.pos.amount / self.cfg.pos_capital, 2)}')
"""
