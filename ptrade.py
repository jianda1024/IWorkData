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
        self.basic_quota = 10000  # 基础额度
        self.upper_limit = 1.5  # 仓位上限(%)
        self.lower_limit = 0.9  # 仓位下限(%)
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
        self.min_offset = 0.004  # 最小振幅差值(%)
        self.add_quotas = [0.50, 0.50, 0.00]  # 加仓额度(%)
        self.sub_quotas = [0.50, 0.30, 0.20]  # 减仓额度(%)
        self.thresholds = [0.004, 0.006, 0.008]  # 涨跌阈值(%)
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

class StockContext:
    def __init__(self):
        self.symbol = ''  # 股票代码
        self.start_price = 0.0  # 初始价格
        self.start_amount = 0.0  # 初始持仓


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

    def turn(self) -> Self:
        """转换为拐点，TurnBar"""
        return TurnBar(self)

    def bar(self) -> Bar:
        return self._bar

    def sma(self) -> SMABar:
        return self._sma

    def macd(self) -> MACDBar:
        return self._macd


class TurnBar:
    def __init__(self, node: NodeBar):
        self.datetime = node.datetime
        self.turning = node.turning
        self.price = node.price
        self.value = node.value

        self._threshold = round(CTX.start_price * CFG.trade.min_offset, 4)
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

    def next_turn(self, node: NodeBar):
        """下一拐点"""
        if self.turning == -2:
            diff = round(self.value - node.value, 4)
            if abs(diff) > self._threshold:
                self.turning = 1 if diff > 0 else -1
            return None
        if self._node_max and self._node_max.value - node.value > self._threshold:
            return self._node_max.turn()
        if self._node_min and node.value - self._node_min.value > self._threshold:
            return self._node_min.turn()
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
        self.nodes = []  # 节点
        self.turns = []  # 拐点

    def prep(self, bar):
        CTX.start_price = round(bar.close, 4)
        self.nodes.clear()
        self.turns.clear()

    def next(self, bar):
        if len(self.nodes) == 0:
            self.__init(bar)
        else:
            self.__next(bar)

    def __init(self, bar):
        node = NodeBar(bar).init()
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
        turn = self.turns[-1].next_turn(post)
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
        self._his_node = his_mkt.nodes[-1]  # 历史节点(昨天)
        self._cur_node = cur_mkt.nodes[-1]  # 当前节点(分钟)
        self._cur_turn = cur_mkt.turns[-1]  # 当前拐点(分钟)
        self._level = -2  # 当前涨跌等级
        self._pos = pos  # 当前持仓

    def gt_upper_limit(self) -> bool:
        """仓位：是否超过上限"""
        return self._pos.valuation >= CFG.pos.basic_quota * CFG.pos.upper_limit

    def gt_lower_limit(self) -> bool:
        """仓位：是否超过下限"""
        return self._pos.valuation >= CFG.pos.basic_quota * CFG.pos.lower_limit

    def is_min_rise(self) -> bool:
        """分钟线：是否上涨"""
        return self._cur_node.sma().fast > self._cur_node.sma().low

    def is_min_fall(self) -> bool:
        """分钟线：是否下跌"""
        return self._cur_node.sma().fast < self._cur_node.sma().low

    def is_day_rise(self) -> bool:
        """日线：是否上涨"""
        return self._his_node.sma().fast > self._his_node.sma().low

    def is_day_fall(self) -> bool:
        """日线：是否下跌"""
        return self._his_node.sma().fast < self._his_node.sma().low

    def is_rise_level(self) -> bool:
        """是否达到上涨等级"""
        return self.level() >= 0 and self._cur_turn.sma().fast < self._cur_node.sma().fast

    def is_fall_level(self) -> bool:
        """是否达到下跌等级"""
        return self.level() >= 0 and self._cur_turn.sma().fast > self._cur_node.sma().fast

    def level(self):
        """当前涨跌等级"""
        if self._level == -2:
            self._level = -1
            diff_value = abs(self._cur_node.sma().fast - self._cur_turn.sma().fast)
            diff_ratio = round(diff_value / CTX.start_price, 4)
            for threshold in reversed(CFG.trade.thresholds):
                if diff_ratio > threshold:
                    self._level = CFG.trade.thresholds.index(threshold)
                    break
        return self._level


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
        self.prices_add = []  # 已加仓价格
        self.prices_sub = []  # 已减仓价格

        self.trade_logs = deque(maxlen=1000)  # 交易日志

    def init(self, pos):
        self.base_amount = Pos(pos).amount_avail

    def prep(self, pos, mkt: PresMarket):
        if mkt and mkt.nodes and mkt.turns:
            self.base_price = mkt.base_price
            self.node = mkt.nodes[-1]
            self.turn = mkt.turns[-1]
            self.pos = Pos(pos)
            self.lvl = -2
            self.ready = True

    def supp_buy(self, order_func: Callable):
        """补仓"""
        if not self.ready: return
        lower_limit = CFG.pos.capital * CFG.pos.lower
        if self.pos.valuation >= lower_limit: return
        amount = max(lower_limit - self.pos.valuation, CFG.pos.capital * 0.25) / self.base_price
        amount = round(amount / 100) * 100
        order_func(self.symbol, amount)

    def do_trade(self, order_func: Callable):
        """交易"""
        if not self.ready: return

        # 减仓
        if self.__is_sell():
            self.__do_sell(order_func)
            return

        # 不买
        if self.__no_buy():
            return

        # 加仓
        lvl = self.__get_lvl()
        if self.__is_buy():
            self.__do_buy(lvl, order_func)
            return

        # 回购
        if self.__is_back():
            lvl = lvl if lvl != -1 else 0
            self.__do_buy(lvl, order_func)

    def __do_buy(self, lvl: int, order_func: Callable):
        amount = CFG.pos.capital * self.turn.lots_add[lvl] / self.base_price
        amount = round(amount / 100) * 100
        order_func(self.symbol, amount)
        self.prices_add.append(self.pos.price_last)
        self.turn.lots_add[lvl] = 0.0

    def __do_sell(self, order_func: Callable):
        count = 0
        lvl = self.__get_lvl()
        lots = self.turn.lots_sub
        for i in range(lvl + 1):
            if lots[i] != 0:
                self.prices_sub.append(self.pos.price_last)
                count += lots[i]
                lots[i] = 0.0
        amount = round(self.base_amount * count / 100) * 100
        amount = -min(self.pos.amount_avail - 100, amount)
        order_func(self.symbol, amount)

    def __is_buy(self):
        """是否买入"""
        # 判断涨幅超过阈值
        level = self.__get_lvl()
        if level == -1:
            return False

        # 判断是否重复操作
        if self.turn.lots_add[level] == 0:
            return False

        # 买入
        return True

    def __is_back(self):
        """是否回购"""
        # 判断买卖次数是否平衡
        length = len(self.prices_add)
        if length >= len(self.prices_sub):
            return False

        # 判断当前价格是否大于卖价
        self.prices_sub.sort()
        if self.pos.price_last <= self.prices_sub[length]:
            return False

        # 回购
        return True

    def __is_sell(self):
        """是否卖出"""
        # 判断可用持仓
        if self.pos.base_amount <= 100:
            return False

        # 判断涨跌：节点SMA快线，是否超过节点SMA慢线
        if self.node.sma().fast >= self.node.sma().low:
            return False

        # 判断涨跌：节点SMA快线，是否超过拐点SMA快线
        if self.node.sma().fast >= self.turn.sma().fast:
            return False

        # 判断跌幅：是否超过阈值
        level = self.__get_lvl()
        if level == -1:
            return False

        # 判断是否重复操作
        if self.turn.lots_sub[level] == 0:
            return False

        # 决定卖出
        return True


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
# 上下文
CTX = StockContext()
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
