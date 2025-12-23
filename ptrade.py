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


class Config:
    def __init__(self, **kwargs):
        # MACD
        self._macd_fast = 12  # 快线周期
        self._macd_slow = 26  # 慢线周期
        self._macd_sign = 9  # 信号线周期

        # SMA
        self._sma_fast = 10  # 快线周期
        self._sma_slow = 30  # 慢线周期

        # 仓位管理
        self._pos_capital = 10000  # 仓位资金
        self._pos_upper = 1.5  # 仓位上限比例
        self._pos_lower = 0.9  # 仓位下限比例
        self._pos_lot_adds = [0.50, 0.50, 0.00]  # 批次加仓比例
        self._pos_lot_subs = [0.50, 0.30, 0.20]  # 批次减仓比例

        self.min_amp = 0.004  # 最小振幅
        self.thresholds = [0.004, 0.006, 0.008]  # 涨跌阈值

        # 更新配置
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 配置分组
        self.macd = SimpleNamespace(fast=self._macd_fast, slow=self._macd_slow, sign=self._macd_sign)
        self.sma = SimpleNamespace(fast=self._sma_fast, slow=self._sma_slow)
        self.pos = SimpleNamespace(capital=self._pos_capital, upper=self._pos_upper, lower=self._pos_lower,
                                   lot_adds=self._pos_lot_adds, lot_subs=self._pos_lot_subs)


# 全局配置
CFG = Config()


class Pos:
    def __init__(self, pos, price_last: float = 0.0):
        self.amount_avail = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
        self.amount_total = getattr(pos, 'amount', 0.0)  # 总持仓数量
        self.price_cost = getattr(pos, 'cost_basis', 0.0)  # 成本价格
        self.price_last = getattr(pos, 'last_sale_price', price_last)  # 最新价格
        self.valuation = self.amount_total * self.price_last  # 持仓市值


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
        self.lots_add = CFG.pos.lot_adds.copy()  # 加仓批次
        self.lots_sub = CFG.pos.lot_subs.copy()  # 减仓批次
        self.datetime = node.datetime
        self.turning = node.turning
        self.value = node.value

        self._threshold = round(base_price * CFG.min_amp, 4)
        self._base_price = base_price
        self._head_max = None
        self._foot_min = None

    def head_max(self, node: NodeBar):
        """到下一拐点前：最大的凸点"""
        if node.value - self.value > self._threshold:
            if self._head_max is None or node.value > self._head_max.value:
                self._head_max = node

    def foot_min(self, node: NodeBar):
        """到下一拐点前：最小的凹点"""
        if self.value - node.value > self._threshold:
            if self._foot_min is None or node.value < self._foot_min.value:
                self._foot_min = node

    def next_turn(self, node: NodeBar):
        """下一拐点"""
        if self.turning == -2:
            diff = round(self.value - node.value, 4)
            if abs(diff) > self._threshold:
                self.turning = 1 if diff > 0 else -1
            return None
        if self._head_max and self._head_max.value - node.value > self._threshold:
            return self._head_max.turn(self._base_price)
        if self._foot_min and node.value - self._foot_min.value > self._threshold:
            return self._foot_min.turn(self._base_price)
        return None


class TradeLog:
    def __init__(self):
        self.map_datetime = ''
        self.map_price = ''
        self.map_value = ''

        self.datetime = ''
        self.amount = ''
        self.price = ''
        self.value = ''
        self.type = ''


class LiveState:
    def __init__(self, ):
        self.ready = False  # 是否就绪
        self.node = None  # 当前节点
        self.turn = None  # 当前拐点
        self.pos = None  # 当前持仓
        self.lvl = -2  # 当前涨跌等级

    def is_over_upper_limit(self) -> bool:
        """仓位是否超过上限"""
        return self.pos.valuation >= CFG.pos.capital * CFG.pos.upper

    def is_over_lower_limit(self) -> bool:
        """仓位是否超过下限"""
        return self.pos.valuation >= CFG.pos.capital * CFG.pos.lower

    def is_sma(self) -> bool:
        return self.node.sma().fast > self.node.sma().low


    def node(self, node: NodeBar):
        self.node = node
        return self

    def turn(self, turn: TurnBar):
        self.turn = turn
        return self

    def pos(self, pos: Pos):
        self.pos = pos
        return self

    def lv(self, lvl: int):
        self.lvl = lvl
        return self


class StockMarket:
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
            self.turns[-1].head_max(node)
        elif prev.value > node.value < post.value:
            node.turning = -1
            self.turns[-1].foot_min(node)

        # 计算拐点
        turn = self.turns[-1].next_turn(post)
        self.turns.append(turn) if turn else None

        # if self.nodes[-1].datetime > '2025-10-27 13:09:00':
        #     pass


class StockBroker:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.base_amount = 0.0  # 基准持仓
        self.base_price = 0.0  # 基准价格
        self.prices_add = []  # 已加仓价格
        self.prices_sub = []  # 已减仓价格

        # 当前信息
        self.ready = False  # 是否就绪
        self.node = None  # 当前节点
        self.turn = None  # 当前拐点
        self.pos = None  # 当前持仓
        self.lvl = -2  # 当前涨跌等级

    def init(self, pos):
        self.base_amount = Pos(pos).amount_avail
        self.prices_add.clear()
        self.prices_sub.clear()
        self.ready = False

    def prep(self, pos, mkt: StockMarket):
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

    def __no_buy(self):
        """是否不买"""
        # 判断仓位：是否超过上限
        if self.pos.valuation >= CFG.pos.capital * CFG.pos.upper:
            return True

        # 判断涨跌：节点SMA快线，是否超过节点SMA慢线
        if self.node.sma().fast <= self.node.sma().low:
            return True

        # 判断涨跌：节点SMA快线，是否超过拐点SMA快线
        return self.node.sma().fast <= self.turn.sma().fast

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

    def __get_lvl(self):
        if self.lvl == -2:
            self.lvl = -1
            diff_value = abs(self.node.sma().fast - self.turn.sma().fast)
            diff_ratio = round(diff_value / self.base_price, 3)
            for level in range(len(CFG.thresholds) - 1, -1, -1):
                if diff_ratio > CFG.thresholds[level]:
                    self.lvl = level
                    return self.lvl
        return self.lvl


class StockManager:
    def __init__(self, symbol):
        self.symbol = symbol
        self.base_pos = None  # 基准仓位
        self.curr_pos = None  # 当前仓位
        self.hist_mkt = LiveState()  # 历史行情
        self.pres_mkt = StockMarket()  # 当前行情

    def prep(self, bar, pos, his_data):
        self.base_pos = Pos(pos)
        self.pres_mkt.prep(bar)

    def next(self, bar, pos):
        self.curr_pos = Pos(pos)
        self.pres_mkt.next(bar)


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

    config = Config(symbol=g.symbol)
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
