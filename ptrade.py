#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Phelix Ruan. All rights reserved.
#
# Description: Personal Stock Trading Strategy - Market Analysis
# Author: Phelix Ruan
# Created: 2025-11-20
# Email: jianda1024@163.com
#

"""
COPYRIGHT:
    This code is proprietary to Phelix Ruan.
    No commercial use, distribution, or modification without permission.

DISCLAIMER:
    For educational purposes only. Not investment advice.
    Use at your own risk. Author not liable for any losses.
"""
from types import SimpleNamespace
from typing import Self, Callable
from collections import deque


############################################################
class Vary:
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class PosConfig(Vary):
    def __init__(self, **kwargs):
        self.base_principal = 8000  # 基础资金
        self.cost_limit = 1.50  # 成本上限（比例）
        self.loss_limit = 0.15  # 亏损上限（比例）
        self.gain_limit = 0.05  # 盈利上限（比例）
        self.update(**kwargs)


class SmaConfig(Vary):
    def __init__(self, **kwargs):
        self.fast = 10  # 快线周期
        self.slow = 30  # 慢线周期
        self.update(**kwargs)


class MACDConfig(Vary):
    def __init__(self, **kwargs):
        self.fast = 12  # 快线周期
        self.slow = 26  # 慢线周期
        self.sign = 9  # 信号线周期
        self.update(**kwargs)


class TradeConfig(Vary):
    def __init__(self, **kwargs):
        self.amp_lower_limit = 0.004  # 振幅下限（比例）
        self.rise_add_quotas = [0.300]  # 加仓额度（比例）
        self.rise_thresholds = [0.004]  # 加仓阈值（比例）
        self.fall_sub_quotas = [0.350, 0.350, 0.300]  # 减仓额度（比例）
        self.fall_thresholds = [0.004, 0.006, 0.009]  # 减仓阈值（比例）
        self.update(**kwargs)


class StockConfig(Vary):
    def __init__(self, **kwargs):
        self.pos = PosConfig()
        self.sma = SmaConfig()
        self.macd = MACDConfig()
        self.trade = TradeConfig()
        self.update(**kwargs)

    def shift(self, level: str = 'default'):
        match level:
            case 'default':
                self.sma = SmaConfig()
            case 'day_bar':
                self.sma = SmaConfig(fast=5, slow=10)


############################################################
class NodeMark:
    def __init__(self):
        # 涨跌等级，标记在节点上，用于避免重复计算
        self.rise_level = -2
        self.fall_level = -2


class TurnMark:
    def __init__(self):
        # 涨跌批次，标记在拐点上，用于避免重复交易
        self.rise_lots = CFG.trade.rise_add_quotas.copy()
        self.fall_lots = CFG.trade.fall_sub_quotas.copy()


############################################################
class Pos:
    def __init__(self, pos, price_last: float = 0.0):
        self.avail_amount = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
        self.total_amount = getattr(pos, 'amount', 0.0)  # 总持仓数量
        self.cost_price = getattr(pos, 'cost_basis', 0.0)  # 成本价格
        self.last_price = getattr(pos, 'last_sale_price', price_last)  # 最新价格
        self.valuation = round(self.total_amount * self.last_price, 2)  # 市值
        self.principal = round(self.total_amount * self.cost_price, 2)  # 本金

    def gt_cost_limit(self):
        """超过成本上限"""
        return self.principal > CFG.pos.base_principal * CFG.pos.cost_limit

    def gt_loss_limit(self):
        """超过亏损上限"""
        return self.principal - self.valuation >= CFG.pos.base_principal * CFG.pos.loss_limit

    def remain_amount(self):
        """获取保留的股票数量"""
        if self.total_amount > self.avail_amount:
            return 0
        if self.valuation - self.principal > CFG.pos.base_principal * CFG.pos.gain_limit:
            return 0
        return 100


class Bar:
    def __init__(self, bar):
        self.datetime = bar.datetime  # 时间
        self.price = round(bar.price, 4)  # 最新价
        self.close = round(bar.close, 4)  # 收盘价
        self.open = round(bar.open, 4)  # 开盘价
        self.high = round(bar.high, 4)  # 最高价
        self.low = round(bar.low, 4)  # 最低价
        self.money = round(bar.money, 2)  # 交易金额
        self.volume = round(bar.volume)  # 交易量


class SMABar:
    def __init__(self):
        self.fast = 0.0  # SMA快线
        self.slow = 0.0  # SMA慢线

    def first(self, bar: Bar):
        self.fast = round(bar.close, 4)
        self.slow = round(bar.close, 4)
        return self

    def next(self, bar: Bar, pre_sma: Self):
        fast = (pre_sma.fast * (CFG.sma.fast - 1) + bar.close) / CFG.sma.fast
        slow = (pre_sma.slow * (CFG.sma.slow - 1) + bar.close) / CFG.sma.slow
        self.fast = round(fast, 4)
        self.slow = round(slow, 4)
        return self


class MACDBar:
    def __init__(self):
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.dif = 0.0
        self.dea = 0.0
        self.macd = 0.0

    def first(self, bar: Bar):
        self.ema_fast = round(bar.close, 4)
        self.ema_slow = round(bar.close, 4)
        return self

    def next(self, bar: Bar, pre_macd: Self):
        self.ema_fast = self._ema(bar.close, CFG.macd.fast, pre_macd.ema_fast)
        self.ema_slow = self._ema(bar.close, CFG.macd.slow, pre_macd.ema_slow)
        self.dif = round(self.ema_fast - self.ema_slow, 4)
        self.dea = self._ema(self.dif, CFG.macd.sign, pre_macd.dea)
        self.macd = round((self.dif - self.dea) * 2, 4)
        return self

    @staticmethod
    def _ema(price, period, pre_ema):
        alpha = 2 / (period + 1)
        ema = alpha * price + (1 - alpha) * pre_ema
        return round(ema, 4)


class NodeBar:
    def __init__(self, bar: Bar):
        self.datetime = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.price = round(bar.price, 4)
        self.value = 0.0
        self.turning = 0

        self._bar = bar
        self._sma = None
        self._macd = None
        self._mark = NodeMark()

    def first(self) -> Self:
        """初始节点，NodeBar"""
        self._sma = SMABar().first(self._bar)
        # self._macd = MACDBar().first(self._bar)
        self.value = self._sma.fast
        self.turning = -2
        return self

    def next(self, pre_node: Self) -> Self:
        """下一个节点，NodeBar"""
        self._sma = SMABar().next(self._bar, pre_node.sma())
        # self._macd = MACDBar().next(self._bar, pre_node.macd())
        self.value = self._sma.fast
        return self

    def turn(self):
        """转换为拐点，TurnBar"""
        return TurnBar(self)

    def bar(self) -> Bar:
        return self._bar

    def sma(self) -> SMABar:
        return self._sma

    def macd(self) -> MACDBar:
        return self._macd

    def mark(self) -> NodeMark:
        return self._mark


class TurnBar:
    def __init__(self, node: NodeBar):
        self.node = node
        self.mark = TurnMark()
        self._max_node = None
        self._min_node = None

    def max_node(self, node: NodeBar, threshold: float):
        """到下一拐点前：最大的凸点"""
        if node.value - self.node.value > threshold:
            if self._max_node is None or node.value > self._max_node.value:
                self._max_node = node

    def min_node(self, node: NodeBar, threshold: float):
        """到下一拐点前：最小的凹点"""
        if self.node.value - node.value > threshold:
            if self._min_node is None or node.value < self._min_node.value:
                self._min_node = node

    def next_turn(self, node: NodeBar, threshold: float):
        """下一拐点"""
        if self.node.turning == -2:
            diff = round(self.node.value - node.value, 4)
            if abs(diff) > threshold:
                self.node.turning = 1 if diff > 0 else -1
            return None
        if self._max_node and self._max_node.value - node.value > threshold:
            return self._max_node.turn()
        if self._min_node and node.value - self._min_node.value > threshold:
            return self._min_node.turn()
        return None


class DataBus:
    def __init__(self, stock_code):
        self.stock_code = stock_code  # 股票代码
        self.base_price = 0.0  # 基准价格
        self.base_amount = 0.0  # 基准持仓
        self.hists = []  # 历史节点
        self.nodes = []  # 当前节点
        self.turns = []  # 当前拐点
        self.pos = None  # 当前仓位

    def prepare(self, bars):
        self.base_price = round(bars[-1].close, 4)
        self.hists.clear()
        self.nodes.clear()
        self.turns.clear()

        CFG.shift("day_bar")
        self.hists.append(NodeBar(bars[0]).first())
        for bar in bars[1:]:
            node = NodeBar(bar).next(self.hists[-1])
            self.hists.append(node)
        CFG.shift()
        return self

    def running(self, bar, pos):
        self.pos = Pos(pos)
        if not self.nodes:
            self.__first(bar)
        else:
            self.__next(bar)
        return self

    def __first(self, bar):
        self.base_amount = self.pos.avail_amount
        node = NodeBar(bar).first()
        turn = node.turn()
        self.nodes.append(node)
        self.turns.append(turn)

    def __next(self, bar):
        node = NodeBar(bar).next(self.nodes[-1])
        if node.value == self.nodes[-1].value:
            return
        self.nodes.append(node)
        if len(self.nodes) < 3:
            return

        # 最小振幅价格差
        threshold = round(self.base_price * CFG.trade.amp_lower_limit, 4)

        # 计算凹凸点
        prev = self.nodes[-3]
        node = self.nodes[-2]
        post = self.nodes[-1]
        if prev.value < node.value > post.value:
            node.turning = 1
            self.turns[-1].max_node(node, threshold)
        elif prev.value > node.value < post.value:
            node.turning = -1
            self.turns[-1].min_node(node, threshold)

        # 计算拐点
        turn = self.turns[-1].next_turn(post, threshold)
        self.turns.append(turn) if turn else None

        if self.nodes[-1].datetime > '2025-10-25 13:09:00':
            pass


############################################################
class State:
    def __init__(self, bus: DataBus):
        self.stock_code = bus.stock_code
        self.base_price = bus.base_price
        self.base_amount = bus.base_amount
        self.yest_node = bus.hists[-1]
        self.last_node = bus.nodes[-1]
        self.last_turn = bus.turns[-1]
        self.pos = bus.pos

    def is_buy(self):
        """是否买入"""
        # 如果亏损超过上限
        if self.pos.gt_loss_limit():
            return False
        # 如果仓位超过上限
        if self.pos.gt_cost_limit():
            return False
        # 如果日线下跌
        if self.yest_node.sma().fast <= self.yest_node.sma().slow:
            return False
        # 如果分钟线下跌
        if self.last_node.sma().fast <= self.last_node.sma().slow:
            return False
        if self.last_node.sma().fast <= self.last_turn.node.sma().fast:
            return False
        # 如果涨幅未达到阈值
        if self.rise_level() == -1:
            return False
        # 如果是重复操作
        if self.last_turn.mark.rise_lots[self.rise_level()] == 0:
            return False
        # 决定买入
        return True

    def is_sell(self):
        """是否卖出"""
        # 如果可用持仓不大于底仓
        if self.pos.avail_amount <= self.pos.remain_amount():
            return False
        # 如果分钟线上涨
        if self.last_node.sma().fast >= self.last_node.sma().slow:
            return False
        if self.last_node.sma().fast >= self.last_turn.node.sma().fast:
            return False
        # 如果跌幅未达到阈值
        if self.fall_level() == -1:
            return False
        # 如果是重复操作
        if self.last_turn.mark.fall_lots[self.fall_level()] == 0:
            return False
        # 决定卖出
        return True

    def rise_level(self):
        """当前上涨等级"""
        mark = self.last_node.mark()
        if mark.rise_level == -2:
            mark.rise_level = self.__level(CFG.trade.rise_thresholds)
        return mark.rise_level

    def fall_level(self):
        """当前下跌等级"""
        mark = self.last_node.mark()
        if mark.fall_level == -2:
            mark.fall_level = self.__level(CFG.trade.fall_thresholds)
        return mark.fall_level

    def __level(self, thresholds):
        diff_value = abs(self.last_node.sma().fast - self.last_turn.node.sma().fast)
        diff_ratio = round(diff_value / self.base_price, 4)
        for threshold in reversed(thresholds):
            if diff_ratio > threshold:
                return thresholds.index(threshold)
        return -1


class StockLog:
    def __init__(self, symbol, amount):
        self.symbol = symbol  # 股票代码
        if amount > 0:
            self.trade_type = 'buy'
            self.trade_amount = amount
        else:
            self.trade_type = 'sell'
            self.trade_amount = -amount

        # 节点信息
        self.bar_datetime = ''
        self.bar_price = 0.0
        self.bar_value = 0.0

        # 映射信息
        self.map_datetime = ''
        self.map_price = 0.0
        self.map_value = 0.0

    def node(self, node: NodeBar):
        self.bar_datetime = node.datetime
        self.bar_price = node.price
        self.bar_value = node.value
        return self

    def turn(self, turn: TurnBar):
        self.map_datetime = turn.node.datetime
        self.map_price = turn.node.price
        self.map_value = turn.node.value
        return self


class StockLogger:
    def __init__(self):
        self.logs = deque(maxlen=1000)

    def info(self, state: State, amount):
        logger = StockLog(state.stock_code, amount)
        logger.node(state.last_node)
        logger.turn(state.last_turn)
        self.logs.append(logger)


class StockBroker:
    @staticmethod
    def trade(bus: DataBus, func: Callable):
        state = State(bus)
        if state.is_buy():
            StockBroker.buy(state, func)
            return

        if state.is_sell():
            StockBroker.sell(state, func)
            return

    @staticmethod
    def buy(state: State, func: Callable):
        lvl = state.rise_level()
        lots = state.last_turn.mark.rise_lots
        buy_amount = CFG.pos.base_principal / state.base_price * lots[lvl]
        amount = round(buy_amount / 100) * 100

        # 执行买入
        func(state.stock_code, amount)
        lots[lvl] = 0.0

    @staticmethod
    def sell(state: State, func: Callable):
        lvl = state.fall_level()
        lots = state.last_turn.mark.fall_lots

        # 最小数量：根据基准本金
        min_qty = CFG.pos.base_principal / state.base_price * lots[lvl]
        # 减仓数量：根据当日初始可用持仓
        cur_qty = state.base_amount * lots[lvl]
        # 避免低仓位时，还分多次减仓
        sell_qty = max(min_qty, cur_qty)
        # 不得超过当前可用持仓
        sell_amount = min(sell_qty, state.pos.avail_amount)
        # 调整到100的倍数，并留下底仓
        amount = round(sell_amount / 100) * 100 - state.pos.remain_amount()

        # 执行卖出
        func(state.stock_code, -amount)
        lots[lvl] = 0.0


class StockMarket:
    def __init__(self):
        self.buses: dict[str, DataBus] = {}

    def prepare(self, symbol, bars):
        self.buses[symbol] = DataBus(symbol).prepare(bars)

    def running(self, symbol, bar, pos):
        bus = self.buses.get(symbol).running(bar, pos)
        StockBroker.trade(bus, order)

    def delete(self, symbol):
        del self.buses[symbol]

    def clear(self):
        self.buses.clear()


"""
########################################################################################################################
import logging
from ptrade import GlobalVars

log = logging.getLogger()
g = GlobalVars()


def set_universe(symbol):
    g.symbol = symbol.replace('SS', 'SH')
    pass


def order(symbol, amount):
    pass


def get_history(**kwargs):
    pass
def get_positions():
    pass


########################################################################################################################
"""

# 全局配置
CFG = StockConfig()
LOG = StockLogger()


def initialize(context):
    """启动时执行一次"""
    g.symbols = ['562600.SS','159857.SZ']
    set_universe(g.symbols)
    g.market = StockMarket()
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    his_data = get_history(60, frequency='1d')
    for symbol in g.symbols:
        df = his_data.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in df.iterrows()]
        g.market.prepare(symbol, bars)


def handle_data(context, data):
    """每个单位周期执行一次"""
    positions = context.portfolio.positions
    for symbol in g.symbols:
        pos = positions.get(symbol)
        bar = data[symbol]
        g.market.running(symbol, bar, pos)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
