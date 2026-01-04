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
from collections import deque
from typing import Self, Callable


class Vary:
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class PosConfig(Vary):
    def __init__(self, **kwargs):
        self.target_profit = 0.03  # 盈利目标（比例）
        self.upper_limit = 1.5  # 仓位上限（比例）
        self.base_quota = 10000  # 基础额度
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
        self.min_offset = 0.004  # 最小振幅差值（比例）
        self.add_quotas = [0.250, 0.250]  # 加仓额度（比例）
        self.add_limits = [0.004, 0.006]  # 加仓阈值（比例）
        self.sub_quotas = [0.400, 0.300, 0.300]  # 减仓额度（比例）
        self.sub_limits = [0.004, 0.007, 0.010]  # 减仓阈值（比例）
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
                self.sma = SmaConfig(fast=10, slow=60)


############################################################

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

    def init(self, bar: Bar):
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

    def init(self, bar: Bar):
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
        self.turning = 0  # 起点-2，凹点-1，凸点1，其他0
        self.price = round(bar.price, 4)
        self.value = 0.0

        self._bar = bar
        self._sma = None
        self._macd = None

    def init(self) -> Self:
        """初始节点，NodeBar"""
        self._sma = SMABar().init(self._bar)
        self._macd = MACDBar().init(self._bar)
        self.value = self._sma.fast
        self.turning = -2
        return self

    def next(self, pre_node: Self) -> Self:
        """下一个节点，NodeBar"""
        self._sma = SMABar().next(self._bar, pre_node.sma())
        self._macd = MACDBar().next(self._bar, pre_node.macd())
        self.value = self._sma.fast
        return self

    def turn(self, base_price: float) -> Self:
        """转换为拐点，TurnBar"""
        return TurnBar(self, base_price)

    def bar(self) -> Bar:
        return self._bar

    def sma(self) -> SMABar:
        return self._sma

    def macd(self) -> MACDBar:
        return self._macd


class TurnBar:
    def __init__(self, node: NodeBar, base_price: float):
        self.datetime = node.datetime
        self.turning = node.turning
        self.price = node.price
        self.value = node.value

        self.lots_add = CFG.trade.add_quotas.copy()
        self.lots_sub = CFG.trade.sub_quotas.copy()

        self._threshold = round(base_price * CFG.trade.min_offset, 4)
        self._node_max = None
        self._node_min = None

    def max_node(self, node: NodeBar):
        """到下一拐点前：最大的凸点"""
        if node.value - self.value > self._threshold:
            if self._node_max is None or node.value > self._node_max.value:
                self._node_max = node

    def min_node(self, node: NodeBar):
        """到下一拐点前：最小的凹点"""
        if self.value - node.value > self._threshold:
            if self._node_min is None or node.value < self._node_min.value:
                self._node_min = node

    def next_turn(self, node: NodeBar, base_price: float):
        """下一拐点"""
        if self.turning == -2:
            diff = round(self.value - node.value, 4)
            if abs(diff) > self._threshold:
                self.turning = 1 if diff > 0 else -1
            return None
        if self._node_max and self._node_max.value - node.value > self._threshold:
            return self._node_max.turn(base_price)
        if self._node_min and node.value - self._node_min.value > self._threshold:
            return self._node_min.turn(base_price)
        return None


class HistMarket:
    def __init__(self):
        self.nodes = []

    def calc(self, bars: list[Bar]):
        if not bars:
            return
        CFG.shift("day_bar")
        fist_node = NodeBar(bars[0]).init()
        self.nodes.append(fist_node)
        for bar in bars[1:]:
            node = NodeBar(bar).next(self.nodes[-1])
            self.nodes.append(node)
        CFG.shift()


class PresMarket:
    def __init__(self):
        self.base_price = 0.0  # 基准价格
        self.nodes = []  # 节点
        self.turns = []  # 拐点

    def prep(self, bar):
        self.base_price = round(bar.close, 4)
        self.nodes.clear()
        self.turns.clear()

    def next(self, bar):
        if len(self.nodes) == 0:
            self.__init(bar)
        else:
            self.__next(bar)

    def __init(self, bar):
        node = NodeBar(bar).init()
        turn = node.turn(self.base_price)
        self.nodes.append(node)
        self.turns.append(turn)

    def __next(self, bar):
        node = NodeBar(bar).next(self.nodes[-1])
        if node.value == self.nodes[-1].value:
            return
        self.nodes.append(node)
        if len(self.nodes) < 3:
            return

        # 计算凹凸点
        prev = self.nodes[-3]
        node = self.nodes[-2]
        post = self.nodes[-1]
        if prev.value < node.value > post.value:
            node.turning = 1
            self.turns[-1].max_node(node)
        elif prev.value > node.value < post.value:
            node.turning = -1
            self.turns[-1].min_node(node)

        # 计算拐点
        turn = self.turns[-1].next_turn(post, self.base_price)
        self.turns.append(turn) if turn else None

        # if self.nodes[-1].datetime > '2025-10-27 13:09:00':
        #     pass


############################################################

class Pos:
    def __init__(self, pos, price_last: float = 0.0):
        self.amount_avail = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
        self.amount_total = getattr(pos, 'amount', 0.0)  # 总持仓数量
        self.price_cost = getattr(pos, 'cost_basis', 0.0)  # 成本价格
        self.price_last = getattr(pos, 'last_sale_price', price_last)  # 最新价格
        self.valuation = self.amount_total * self.price_last  # 持仓市值


class State:
    def __init__(self, his_mkt: HistMarket, cur_mkt: PresMarket, pos: Pos):
        self.base_price = cur_mkt.base_price  # 基准价格
        self._cur_turn = cur_mkt.turns[-1]  # 当前拐点(分钟)
        self._cur_node = cur_mkt.nodes[-1]  # 当前节点(分钟)
        self._his_node = his_mkt.nodes[-1]  # 历史节点(昨天)
        self._lvl_rise = -2  # 当前上涨等级
        self._lvl_fall = -2  # 当前下跌等级
        self._pos = pos  # 当前持仓

    def cur_turn(self):
        return self._cur_turn

    def is_buy(self):
        """是否买入"""
        # 如果仓位超过上限
        if self._pos.valuation >= CFG.pos.base_quota * CFG.pos.upper_limit:
            return False
        # 如果日线下跌
        if self._his_node.sma().fast <= self._his_node.sma().low:
            return False
        # 如果分钟线下跌
        if self._cur_node.sma().fast <= self._cur_node.sma().low:
            return False
        if self._cur_node.sma().fast <= self._cur_turn.sma().fast:
            return False
        # 如果涨幅未达到阈值
        if self.lvl_rise() == -1:
            return False
        # 如果是重复操作
        if self._cur_turn.lots_add[self.lvl_rise()] == 0:
            return False
        # 决定买入
        return True

    def is_sell(self):
        """是否卖出"""
        # 如果没有可用持仓
        if self._pos.amount_avail <= 0:
            return False
        # 如果日线上涨
        if self._his_node.sma().fast >= self._his_node.sma().low:
            return False
        # 如果分钟线上涨
        if self._cur_node.sma().fast >= self._cur_node.sma().low:
            return False
        if self._cur_node.sma().fast >= self._cur_turn.sma().fast:
            return False
        # 如果跌幅未达到阈值
        if self.lvl_fall() == -1:
            return False
        # 如果是重复操作
        if self._cur_turn.lots_sub[self.lvl_rise()] == 0:
            return False
        # 决定卖出
        return True

    def lvl_rise(self):
        """当前上涨等级"""
        if self._lvl_rise == -2:
            self._lvl_rise = self._level(CFG.trade.add_limits)
        return self._lvl_rise

    def lvl_fall(self):
        """当前下跌等级"""
        if self._lvl_fall == -2:
            self._lvl_fall = self._level(CFG.trade.sub_limits)
        return self._lvl_fall

    def _level(self, limits):
        diff_value = abs(self._cur_node.sma().fast - self._cur_turn.sma().fast)
        diff_ratio = round(diff_value / self.base_price, 4)
        for threshold in reversed(limits):
            if diff_ratio > threshold:
                return limits.index(threshold)
        return -1


class TradeLog:
    def __init__(self, node: NodeBar, amount):
        if amount > 0:
            self.trade_type = 'buy'
            self.trade_amount = amount
        else:
            self.trade_type = 'sell'
            self.trade_amount = -amount

        # 节点信息
        self.bar_datetime = node.datetime
        self.bar_price = node.price
        self.bar_value = node.value

        # 映射信息
        self.map_datetime = ''
        self.map_price = 0.0
        self.map_value = 0.0

    def map(self, turn: TurnBar):
        self.map_datetime = turn.datetime
        self.map_price = turn.price
        self.map_value = turn.value
        return self


class StockBroker:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.base_amount = 0.0  # 基准持仓
        self.trade_logs = deque(maxlen=1000)  # 交易日志

    def init(self, pos):
        self.base_amount = Pos(pos).amount_avail

    def trade(self, state: State, order_func: Callable):
        if state.is_buy():
            lvl = state.lvl_rise()
            lots = state.cur_turn().lots_add
            amount = CFG.pos.base_quota * lots[lvl] / state.base_price
            amount = round(amount / 100) * 100
            order_func(self.symbol, amount)
            lots[lvl] = 0.0
            return

        if state.is_sell():
            lvl = state.lvl_fall()
            lots = state.cur_turn().lots_sub
            amount = self.base_amount * lots[lvl]
            amount = round(amount / 100) * 100
            order_func(self.symbol, -amount)
            lots[lvl] = 0.0
            return


class StockManager:
    def __init__(self, symbol):
        self.symbol = symbol
        self.base_pos = None  # 基准仓位
        self.curr_pos = None  # 当前仓位
        self.hist_mkt = State()  # 历史行情
        self.pres_mkt = PresMarket()  # 当前行情

    def prep(self, bar, pos, his_data):
        self.base_pos = Pos(pos)
        self.pres_mkt.prep(bar)

    def next(self, bar, pos):
        self.curr_pos = Pos(pos)
        self.pres_mkt.next(bar)


############################################################
# 全局配置
CFG = StockConfig()

############################################################

# """
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


########################################################################################################################
# """

def initialize(context):
    """启动时执行一次"""
    g.symbol = '512480.SS'
    set_universe(g.symbol)

    config = StockConfig(symbol=g.symbol)
    # g.market = StockMarket(config)
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    his = get_history(20, frequency='1m', field=['datetime', 'open', 'close', 'volume'], security_list=None, fq='pre')
    first_bar = his.query(f'code in ["{g.symbol}"]').iloc[-1]
    first_pos = context.portfolio.positions.get(g.symbol)
    g.market.pre(first_bar)
    g.trader.pre(first_bar, first_pos)
    pass


def handle_data(context, data):
    """每个单位周期执行一次"""
    g.market.next(data[g.symbol])

    cur_pos = context.portfolio.positions.get(g.symbol)
    cur_trading = g.market.turnings[-1]  # 拐点：-1凹点，1凸点，0其他
    amount = 0
    if cur_trading.point.turn_val == 1:
        amount = -g.trader.__get_sell_amount(cur_pos, cur_trading)
    if cur_trading.point.turn_val == -1:
        amount = g.trader.__get_buy_amount(cur_pos, cur_trading)
    if data[g.symbol].datetime.strftime('%H:%M:%S').endswith('14:58:00'):
        amount = g.trader.get_supp_amount(cur_pos)
    if amount != 0:
        if cur_pos is not None:
            log.info(f'sell={amount}, pre_price={cur_trading.point.price}, cur_price={cur_pos.last_sale_price}')
        order(g.symbol, amount)
    pass


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
