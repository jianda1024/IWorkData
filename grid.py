from __future__ import annotations

from typing import NamedTuple

import pandas as pd


# region ====== 数据模型层 (Mod) ======
class ModBar:
    __slots__ = ('datetime', 'bar',
                 'sma', 'macd', 'atr',
                 'avl', 'obv', 'rsi',
                 )

    def __init__(self, bar):
        self.datetime = bar.datetime
        self.bar = ModKLine(bar)
        self.sma: dict[int, float] = {}
        self.macd: ModMACD | None = None

        self.avl: ModAVL | None = None
        self.atr: ModATR | None = None
        self.obv: ModOBV | None = None
        self.rsi: ModRSI | None = None


class ModKLine:
    __slots__ = ('volume', 'money', 'price', 'open', 'high', 'low', 'close')

    def __init__(self, bar):
        self.volume = round(float(bar.volume), 3)  # 交易量
        self.money = round(float(bar._base.money), 3)  # 交易金额
        self.price = round(float(bar.price), 3)  # 最新价
        self.close = round(float(bar.close), 3)  # 收盘价
        self.open = round(float(bar.open), 3)  # 开盘价
        self.high = round(float(bar.high), 3)  # 最高价
        self.low = round(float(bar.low), 3)  # 最低价

    def agg(self, bar):
        self.volume = round(self.volume + float(bar.volume), 3)
        self.money = round(self.money + float(bar._base.money), 3)
        self.close = round(float(bar.close), 3)
        self.high = round(max(self.high, float(bar.high)), 3)
        self.low = round(min(self.low, float(bar.low), 3))


class ModPos:
    __slots__ = ('symbol', 'total_amount', 'avail_amount', 'curr_price', 'cost_price')

    def __init__(self, pos):
        self.symbol = pos.sid  # 股票代码
        self.total_amount = pos.amount  # 总持仓数量
        self.avail_amount = pos.enable_amount  # 可用持仓数量
        self.curr_price = pos.last_sale_price  # 最新价格
        self.cost_price = pos.cost_basis  # 成本价格

    def update(self, pos):
        self.total_amount = pos.amount
        self.avail_amount = pos.enable_amount
        self.curr_price = pos.last_sale_price
        self.cost_price = pos.cost_basis

    @property
    def total_value(self):
        return round(self.total_amount * self.curr_price, 3)  # 总市值

    @property
    def avail_value(self):
        return round(self.avail_amount * self.curr_price, 3)  # 可用市值

    @property
    def principal(self):
        return round(self.total_amount * self.cost_price, 3)  # 本金


class ModMACD(NamedTuple):
    fast: float = 0.0
    slow: float = 0.0
    dif: float = 0.0
    dea: float = 0.0
    val: float = 0.0


class ModATR(NamedTuple):
    tr_sum: float = 0.0  # 平滑后的 TR 和
    val: float = 0.0  # 平均真实波幅


class ModAVL(NamedTuple):
    money: float = 0.0  # 当日累计成交额
    volume: float = 0.0  # 当日累计成交量
    price: float = 0.0  # 成交量加权平均价（AVL）


class ModOBV(NamedTuple):
    val: float = 0.0  # OBV 累积值
    ma: float = 0.0  # OBV 的均线


class ModRSI(NamedTuple):
    gain: float = 0.0
    loss: float = 0.0
    val: float | None = None  # None 表示数据不足、RSI 无效


# endregion
# region ====== 技术指标层 (Ind) ======
class IndSMA:
    @classmethod
    def calc(cls, bars: list, periods: list):
        if not bars: return
        length = len(bars)
        cur_sma = bars[-1].sma
        tmp_avg = None

        for p in periods:
            if length > p:
                cur_sma[p] = sum([b.bar.close for b in bars[-p:]]) / p
            else:
                if tmp_avg is None:
                    tmp_avg = sum([b.bar.close for b in bars]) / length
                cur_sma[p] = tmp_avg


class IndMACD:
    @classmethod
    def calc(cls, bars: list, fast_period=12, slow_period=26, sign_period=9):
        if not bars: return
        cur = bars[-1]
        price = cur.bar.close

        if len(bars) == 1:
            cur.macd = ModMACD(fast=price, slow=price)
            return

        pre_macd = bars[-2].macd
        fast = pre_macd.fast + (2.0 / (fast_period + 1) * (price - pre_macd.fast))
        slow = pre_macd.slow + (2.0 / (slow_period + 1) * (price - pre_macd.slow))
        dif = fast - slow
        dea = pre_macd.dea + (2.0 / (sign_period + 1) * (dif - pre_macd.dea))
        val = round((dif - dea) * 2.0, 4)
        cur.macd = ModMACD(fast=fast, slow=slow, dif=dif, dea=dea, val=val)


class IndAVl:
    @classmethod
    def calc(cls, bars: list):
        if not bars: return
        cur = bars[-1]

        if len(bars) == 1:
            money = cur.bar.money
            volume = cur.bar.volume
        else:
            pre_avl = bars[-2].avl
            money = pre_avl.money + cur.bar.money
            volume = pre_avl.volume + cur.bar.volume
        price = round(money / volume, 3) if volume else cur.bar.close
        cur.avl = ModAVL(money=money, volume=volume, price=price)


class IndATR:
    @classmethod
    def calc(cls, bars: list, period: int = 14) -> None:
        if not bars: return
        cur = bars[-1]
        cur_bar = cur.bar

        if len(bars) == 1:
            tr_sum = cur_bar.high - cur_bar.low
            val = round(tr_sum, 4)
            cur.atr = ModATR(tr_sum=tr_sum, val=val)
            return

        pre = bars[-2]
        pre_atr = pre.atr
        tr = max(cur_bar.high - cur_bar.low, abs(cur_bar.high - pre.bar.close), abs(cur_bar.low - pre.bar.close))

        if len(bars) <= period + 1:
            tr_sum = pre_atr.tr_sum + tr
            val = round(tr_sum / len(bars), 4)
        else:
            tr_sum = pre_atr.tr_sum - pre_atr.tr_sum / period + tr
            val = round(tr_sum / period, 4)
        cur.atr = ModATR(tr_sum=tr_sum, val=val)


class IndOBV:
    @classmethod
    def calc(cls, bars: list, period: int = 20) -> None:
        if not bars: return
        cur = bars[-1]
        cur_bar = cur.bar
        if len(bars) == 1:
            cur.obv = ModOBV()
            return

        pre = bars[-2]
        pre_obv = pre.obv
        if cur_bar.close > pre.bar.close:
            val = pre_obv.val + cur_bar.volume
        elif cur_bar.close < pre.bar.close:
            val = pre_obv.val - cur_bar.volume
        else:
            val = pre_obv.val

        if len(bars) == 2:
            ma = val
        else:
            alpha = 2.0 / (period + 1)
            ma = alpha * val + (1 - alpha) * pre_obv.ma
        cur.obv = ModOBV(val=val, ma=ma)


class IndRSI:
    @classmethod
    def calc(cls, bars: list, period=14):
        if not bars: return

        cur = bars[-1]
        close = cur.bar.close
        if len(bars) == 1:
            cur.rsi = ModRSI()
            return

        pre = bars[-2]
        pre_rsi = pre.rsi
        change = close - pre.bar.close
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0

        if len(bars) <= period + 1:
            n = len(bars) - 1
            cur_gain   = (pre_rsi.gain * (n - 1) + gain) / n
            cur_loss  = (pre_rsi.loss * (n - 1) + loss) / n
        else:
            cur_gain  = (pre_rsi.gain * (period - 1) + gain) / period
            cur_loss  = (pre_rsi.loss * (period - 1) + loss) / period

        val = None
        if len(bars) > period:
            if cur_loss == 0:
                val = 100.0
            else:
                rs = cur_gain / cur_loss
                val = round(100.0 - 100.0 / (1.0 + rs), 4)
        cur.rsi = ModRSI(gain=cur_gain, loss=cur_loss, val=val)

## 卡尔曼滤波趋势、R平方与斜率双特征控制
# endregion
# region ====== 策略逻辑层 (Stg) ======
class Stg:
    def next_state(self):
        pass

    def execute(self):
        pass


class StgBullTrend(Stg):
    def next_state(self):
        pass

    def execute(self):
        pass


class StgBullDecay(Stg):
    def next_state(self):
        pass

    def execute(self):
        pass


class StgBearTrend(Stg):
    def next_state(self):
        pass

    def execute(self):
        pass


class StgBearDecay(Stg):
    def next_state(self):
        pass

    def execute(self):
        pass


class StgRangeWave(Stg):
    def next_state(self):
        pass


# endregion
# region ========  环境和配置  ========
class Var:
    base_fund = 3000  # 交易基础金额（元）


class Env:
    markets: dict[str, Market] = {}


# endregion
# region ========  行情和交易  ========
class Market:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.dayBus: list[ModBar] = []  # 日线数据
        self.minBus: list[ModBar] = []  # 分钟数据
        self.dayBar: ModBar | None = None  # 当前日K
        self.curPos: ModPos | None = None  # 当前持仓

    def prepare(self, day_df: pd.DataFrame):
        for _, row in day_df.iterrows():
            self.dayBus.append(ModBar(row))
            self.day_calc(self.dayBus)
        return self

    def running(self, pos, bar):
        self.minBus.append(ModBar(bar))
        self.min_calc(self.minBus)
        if self.dayBar is None:
            self.dayBar = ModBar(bar)
            self.curPos = ModPos(pos)
        else:
            self.dayBar.bar.agg(bar)
            self.curPos.update(pos)
        if bar.datetime.minute % 10 == 0:
            bars = [*self.dayBus, self.dayBar]
            self.day_calc(bars)

    @staticmethod
    def day_calc(bars: list):
        IndEMA.calc(bars, [10, 20, 30, 60, 120])
        IndMACD.calc(bars, 12, 26, 9)
        IndATR.calc(bars, 14)
        IndOBV.calc(bars, 20)
        IndRSI.calc(bars, 14)

    @staticmethod
    def min_calc(bars: list):
        IndEMA.calc(bars, [10, 20, 30, 60])
        IndMACD.calc(bars, 12, 26, 9)
        IndAVl.calc(bars)
        IndATR.calc(bars, 14)


class Trader:
    # 执行买、卖
    pass


# endregion
# region ======== PTrade 框架 ========
def initialize(context):
    """启动时执行一次"""
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    positions = context.portfolio.positions
    all_codes = list(positions.keys())
    set_universe(pos_codes)
    day_history = get_history(250, frequency='1d', security_list=all_codes, fq='dypre')
    for code in all_codes:
        day_df = day_history.query(f'code in ["{code}"]')
        Env.markets[code] = Market(code).prepare(day_df)


def handle_data(context, data):
    """每个单位周期执行一次"""
    if not Env.markets:
        before_trading_start(context, data)
    positions = context.portfolio.positions
    for symbol, market in Env.markets.items():
        bar = data[symbol]
        pos = positions.get(symbol)
        market.running(pos, bar)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass

# endregion
