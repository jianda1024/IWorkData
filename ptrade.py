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
from typing import Self
from collections import deque


class StockConfig:
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

        self.turn_min_diff = 0.004  # 拐点最小差值百分比
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
cfg = StockConfig()


class Bar:
    def __init__(self, bar):
        self.datetime = bar.datetime  # 时间
        self.price = round(bar.price, 3)  # 最新价
        self.close = round(bar.close, 3)  # 收盘价
        self.open = round(bar.open, 3)  # 开盘价
        self.high = round(bar.high, 3)  # 最高价
        self.low = round(bar.low, 3)  # 最低价
        self.volume = round(bar.volume)  # 交易量
        self.money = round(bar.money, 2)  # 交易金额


class SMABar:
    def __init__(self):
        self.fast = 0.0  # SMA快线
        self.slow = 0.0  # SMA慢线

    def prep(self, bar: Bar):
        self.fast = round(bar.close, 3)
        self.slow = round(bar.close, 3)
        return self

    def next(self, bar: Bar, pre_sma: Self):
        fast = (pre_sma.fast * (cfg.sma.fast - 1) + bar.close) / cfg.sma.fast
        slow = (pre_sma.slow * (cfg.sma.slow - 1) + bar.close) / cfg.sma.slow
        self.fast = round(fast, 3)
        self.slow = round(slow, 3)
        return self


class MACDBar:
    def __init__(self):
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.dif = 0.0
        self.dea = 0.0
        self.macd = 0.0

    def prep(self, bar: Bar):
        self.ema_fast = round(bar.close, 3)
        self.ema_slow = round(bar.close, 3)
        return self

    def next(self, bar: Bar, pre_macd: Self):
        self.ema_fast = self._ema(bar.close, cfg.macd.fast, pre_macd.ema_fast)
        self.ema_slow = self._ema(bar.close, cfg.macd.slow, pre_macd.ema_slow)
        self.dif = round(self.ema_fast - self.ema_slow, 3)
        self.dea = self._ema(self.dif, cfg.macd.sign, pre_macd.dea)
        self.macd = round((self.dif - self.dea) * 2, 3)
        return self

    @staticmethod
    def _ema(price, period, pre_ema):
        alpha = 2 / (period + 1)
        ema = alpha * price + (1 - alpha) * pre_ema
        return round(ema, 3)


class NodeBar:
    def __init__(self, bar: Bar):
        self.index = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.value = 0.0
        self.turn = 0  # 起点-2，凹点-1，凸点1，其他0

        self._bar = bar
        self._sma = None
        self._macd = None

    def prep(self):
        self._sma = SMABar().prep(self._bar)
        self._macd = MACDBar().prep(self._bar)
        self.value = self._sma.fast
        return self

    def next(self, pre_node: Self):
        self._sma = SMABar().next(self._bar, pre_node.sma())
        self._macd = MACDBar().next(self._bar, pre_node.macd())
        self.value = self._sma.fast
        return self

    def bar(self):
        return self._bar

    def sma(self):
        return self._sma

    def macd(self):
        return self._macd


class TurnBar:
    def __init__(self, base_price: float, node: NodeBar):
        self.threshold = round(base_price * cfg.turn_min_diff, 3)
        self.turn_val = 0  # 起点-2，凹点-1，凸点1，其他0
        self.lots_add = cfg.pos.lot_adds.copy()  # 加仓批次
        self.lots_sub = cfg.pos.lot_subs.copy()  # 减仓批次

        self._node = node
        self._node_max = None
        self._node_min = None

    def most(self, node: NodeBar):
        if self._node.value - node.value > self.threshold:
            if self._node_min is None or node.value < self._node_min.value:
                self._node_min = node
            return
        if node.value - self._node.value > self.threshold:
            if self._node_max is None or node.value > self._node_max.value:
                self._node_max = node



class Pos:
    def __init__(self, pos):
        self.is_empty = pos is None
        self.amount_avail = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
        self.amount_total = getattr(pos, 'amount', 0.0)  # 总持仓数量
        self.price_cost = getattr(pos, 'cost_basis', 0.0)  # 成本价格
        self.price_last = getattr(pos, 'last_sale_price', 0.0)  # 最新价格
        self.valuation = self.amount_total * self.price_last  # 持仓市值


class NodePos:
    def __init__(self, bar: NodeBar, pos: Pos):
        self._bar = bar
        self._pos = pos

    def bar(self):
        return self._bar

    def pos(self):
        return self._pos


class HistMarket:
    def __init__(self):
        self.basis_price = 0.0  # 基准价格


class PresMarket:
    def __init__(self):
        self.curr_pos = None  # 当前仓位
        self.base_pos = None  # 基准仓位
        self.base_price = 0.0  # 基准价格

        self.state = -1  # 状态：-1未启动、0已重置、1已就绪
        self.nodes = []  # 节点
        self.turns = []  # 拐点
        self.marks = []  # 标记点

    def init(self, bar_data):
        self.base_price = round(bar_data.close, 3)
        self.nodes.clear()
        self.turns.clear()
        self.marks.clear()
        self.state = 0

    def next(self, bar, pos):
        match self.state:
            case -1:
                self.init(bar)
            case 0:
                self.__prep(bar, pos)
            case 1:
                self.__next(bar, pos)

    def __prep(self, bar_data, pos_data):
        bar = Bar(bar_data)
        node = NodeBar(bar).prep()
        node.turn().turn_val = -2
        self.nodes.append(node)
        self.turns.append(node)
        self.marks.append(node)
        self.base_pos = Pos(pos_data)
        self.state = 1

    def __next(self, bar_data, pos_data):
        bar = Bar(bar_data)
        pos = Pos(pos_data)
        pre_node = self.nodes[-1]
        node = NodeBar(bar).next(pre_node)
        if node.sma().fast != pre_node.sma().fast:
            self.nodes.append(node)
            self.__cals_turns()
            self.curr_pos = NodePos(node, pos)

    def __turn(self):
        if len(self.nodes) < 3:
            return
        prev = self.nodes[-1]
        node = self.nodes[-2]
        post = self.nodes[-3]
        if prev.value < node.value > post.value:
            node.turn().turn_val = 1
        elif prev.valuel > node.value < post.value:
            node.turn().turn_val = -1

    def __prep_mark(self):
        if len(self.marks) == 1 and self.marks[-1].turn().turn_val == -2:
            curr_val = self.nodes[-1].value
            mark_val = self.marks[-1].value
            if abs(curr_val - mark_val) / self.base_price > cfg.turn_min_diff:
                self.marks[-1].turn().turn_val = 1 if mark_val > curr_val else -1

    def __next_mark(self):
        # 最低拐点、最高拐点
        turns = []
        start_time = self.marks[-1].datetime
        for node in reversed(self.turns):
            if node.datetime <= start_time:
                break
            turns.append(node)
        if not turns:
            return
        turn_min = min(turns, key=lambda x: x.value)
        turn_max = max(turns, key=lambda x: x.value)
        val_min = turn_min.value
        val_max = turn_max.value

        # 左右边界值、阈值
        val_left = self.marks[-1].value
        val_right = self.nodes[-1].value
        threshold = round(self.base_price * cfg.turn_min_diff, 3)

        # 差值大于阈值的拐点
        if val_left - val_min > threshold and val_right - val_min > threshold:
            self.marks.append(turn_min.marking())
        if val_max - val_left > threshold and val_max - val_right > threshold:
            self.marks.append(turn_max.marking())


class StockManager:
    def __init__(self):
        self.hist_pos = None
        self.hist_mkt = HistMarket()
        self.pres_mkt = PresMarket()

    def pre_next(self, bar, pos):
        self.hist_pos = Pos(pos)
        self.pres_mkt.pre_next(Bar(bar))

    def next(self, bar, pos):
        self.pres_mkt.next(Bar(bar))
        cur_pos = NodePos(Pos(pos))


class State:
    def __init__(self, cfg: StockConfig, mkt: StockMarket, pos):
        # 价格、持仓数量、估值
        self.price = getattr(pos, 'last_sale_price', mkt.nodes[-1].price)
        self.amount = getattr(pos, 'amount', 0)
        self.valuation = self.amount * self.price
        self.avail_amount = getattr(pos, 'enable_amount', 0)

        # 是否上涨、是否下跌、是否均线向下趋势
        self.is_rise = mkt.nodes[-1].price > mkt.turns[-1].price
        self.is_fall = mkt.nodes[-1].price < mkt.turns[-1].price
        self.is_sma_rise = mkt.nodes[-1].sma_fast >= mkt.nodes[-1].sma_slow
        self.is_sma_fall = mkt.nodes[-1].sma_fast <= mkt.nodes[-1].sma_slow

        # 是否超过上限、是否低于下限
        self.is_above_upper = self.valuation >= cfg.pos_capital * cfg.pos_upper_limit
        self.is_under_lower = self.valuation <= cfg.pos_capital * cfg.pos_lower_limit

        # 当前涨跌阶段
        self.lv = -1
        dff_val = abs(mkt.turns[-1].sma_fast - mkt.nodes[-1].sma_fast)
        dff_ratio = round(dff_val / mkt.close, 3)
        for lv in range(len(cfg.thresholds) - 1, -1, -1):
            if dff_ratio > cfg.thresholds[lv]:
                self.lv = lv
                break


class StockTrader:
    def __init__(self, cfg: StockConfig, mkt: StockMarket):
        self.cfg = cfg  # 配置
        self.mkt = mkt  # 行情数据
        self.prices_add = []  # 加仓价格
        self.prices_sub = []  # 减仓价格
        self.avail_amount = 0  # 起始可用持仓

    def prepare(self, pos):
        self.prices_add = []
        self.prices_sub = []
        self.avail_amount = getattr(pos, 'enable_amount', 0)

    def supp_amount(self, pos):
        """补仓"""
        state = State(self.cfg, self.mkt, pos)
        if state.is_under_lower:
            lower_limit = self.cfg.pos_capital * self.cfg.pos_lower_limit
            amount = max(lower_limit - state.valuation, self.cfg.pos_capital * 0.25) / self.mkt.close
            return round(amount / 100) * 100
        return 0

    def amount(self, pos):
        state = State(self.cfg, self.mkt, pos)

        # 减仓
        is_sell, amount = self.__sell(state)
        if is_sell:
            return amount

        # 加仓
        is_buy, amount = self.__is_buy(state)
        if is_buy:
            return amount

        # 回购
        is_back, amount = self.__is_back(state)
        if is_back:
            return amount
        return 0

    def __sell(self, state: State):
        # 是否减仓
        if state.avail_amount <= 100:
            return False, 0
        if state.is_rise or state.is_sma_rise:
            return False, 0
        levels = self.mkt.turns[-1].lvs_sub
        if state.lv == -1 or levels[state.lv] == 0:
            return False, 0

        # 减仓
        count = 0
        for i in range(state.lv + 1):
            if levels[i] != 0:
                self.prices_sub.append(state.price)
                count += levels[i]
                levels[i] = 0.0
        amount = self.avail_amount * count / self.mkt.close
        amount = round(amount / 100) * 100
        return True, -min(state.avail_amount - 100, amount)

    def __is_buy(self, state: State):
        # 是否加仓
        if state.is_above_upper:
            return False, 0
        if state.is_fall or state.is_sma_fall:
            return False, 0
        levels = self.mkt.turns[-1].lvs_add
        if state.lv == -1 or levels[state.lv] == 0:
            return False, 0

        # 加仓
        amount = self.cfg.pos_capital * levels[state.lv] / self.mkt.close
        amount = round(amount / 100) * 100
        self.prices_add.append(state.price)
        levels[state.lv] = 0.0
        return True, amount

    def __is_back(self, state: State):
        # 是否回购
        if state.is_above_upper:
            return False, 0
        if state.is_fall or state.is_sma_fall:
            return False, 0
        length = len(self.prices_add)
        if length >= len(self.prices_sub):
            return False, 0
        self.prices_sub.sort()
        if state.price <= self.prices_sub[length]:
            return False, 0

        # 回购
        lv = state.lv if state.lv != -1 else 0
        levels = self.mkt.turns[-1].lvs_add
        amount = self.cfg.pos_capital * levels[lv] / self.mkt.close
        amount = round(amount / 100) * 100
        self.prices_add.append(state.price)
        levels[lv] = 0.0
        return True, amount


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
    g.market = StockMarket(config)
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
