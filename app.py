#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Phelix Ruan. All rights reserved.
#
# Description: Personal Stock Trading Strategy - Market Analysis
# Author: Phelix Ruan
# Created: 2026-04-12
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
from __future__ import annotations

from collections import deque


class Config:
    class Macd:
        def __init__(self):
            self.fast = 12  # 快线周期
            self.slow = 26  # 慢线周期
            self.sign = 9  # 信号线周期


############################################################
class Bin:
    class Bar:
        def __init__(self, bar):
            self.datetime = bar.datetime
            self.volume: float = round(bar.volume, 2)  # 交易量
            self.money: float = round(bar.money, 2)  # 交易金额
            self.price: float = round(bar.price, 5)  # 最新价
            self.close: float = round(bar.close, 5)  # 收盘价
            self.open: float = round(bar.open, 5)  # 开盘价
            self.high: float = round(bar.high, 5)  # 最高价
            self.low: float = round(bar.low, 5)  # 最低价

    class Pos:
        def __init__(self, pos):
            self.symbol: str = getattr(pos, 'sid', '')  # 股票代码
            self.total_amount: float = getattr(pos, 'amount', 0.0)  # 总持仓数量
            self.avail_amount: float = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
            self.last_price: float = getattr(pos, 'last_sale_price', 0.0)  # 最新价格
            self.cost_price: float = getattr(pos, 'cost_basis', 0.0)  # 成本价格
            self.valuation: float = round(self.total_amount * self.last_price, 2)  # 市值
            self.principal: float = round(self.total_amount * self.cost_price, 2)  # 本金

    class Sma:
        def __init__(self):
            self.sma05: float = 0.0
            self.sma10: float = 0.0
            self.sma20: float = 0.0
            self.sma30: float = 0.0
            self.sma60: float = 0.0

    class Macd:
        def __init__(self):
            self.fast: float = 0.0
            self.slow: float = 0.0
            self.sign: float = 0.0
            self.dif_: float = 0.0
            self.dea_: float = 0.0
            self.macd: float = 0.0

    class Node:
        def __init__(self, bar):
            self.idx: str = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
            self.bar: Bin.Bar = Bin.Bar(bar)
            self.sma: Bin.Sma = Bin.Sma()
            self.macd: Bin.Macd = Bin.Macd()


class Bus:
    def __init__(self, maxlen=None):
        self.data: deque[Bin.Node] = deque(maxlen=maxlen)
        self.temp = {}

    def __len__(self):
        return len(self.data)

    def get(self, idx: int) -> Bin.Node:
        return self.data[idx]


class Line:
    class Sma:
        _keys = {5: 'sma05', 10: 'sma10', 20: 'sma20', 30: 'sma30', 60: 'sma60'}

        @staticmethod
        def calc(bus: Bus):
            node = bus.get(-1)
            if len(bus) == 1:
                Line.Sma._prep(bus, node, 5)
                Line.Sma._prep(bus, node, 10)
                Line.Sma._prep(bus, node, 20)
                Line.Sma._prep(bus, node, 30)
                Line.Sma._prep(bus, node, 60)
            else:
                Line.Sma._next(bus, node, 5)
                Line.Sma._next(bus, node, 10)
                Line.Sma._next(bus, node, 20)
                Line.Sma._next(bus, node, 30)
                Line.Sma._next(bus, node, 60)

        @staticmethod
        def _prep(bus: Bus, node: Bin.Node, period: int):
            key = Line.Sma._keys.get(period)
            dqe = deque(maxlen=period)
            price = node.bar.price
            dqe.append(price)
            bus.temp[key] = dqe
            setattr(node.sma, key, price)

        @staticmethod
        def _next(bus: Bus, node: Bin.Node, period: int):
            key = Line.Sma._keys.get(period)
            dqe = bus.temp.get(key)
            dqe.append(node.bar.price)
            value = round(sum(dqe) / len(dqe), 5)
            setattr(node.sma, key, value)

    class Macd:
        @staticmethod
        def calc(bus: Bus):
            if len(bus) == 1:
                Line.Macd._prep(bus)
            else:
                Line.Macd._next(bus)

        @staticmethod
        def _prep(bus: Bus):
            node = bus.get(-1)
            price = node.bar.price
            node.macd.fast = price
            node.macd.slow = price
            node.macd.dif_ = 0.0
            node.macd.dea_ = 0.0
            node.macd.macd = 0.0

        @staticmethod
        def _next(bus: Bus):
            node = bus.get(-1)
            price = node.bar.price
            pre_macd = bus.get(-2).macd
            fast = Line.Macd._ema(price, Var.macd.fast, pre_macd.fast)
            slow = Line.Macd._ema(price, Var.macd.slow, pre_macd.slow)
            dif_ = round(fast - slow, 5)
            dea_ = Line.Macd._ema(dif_, Var.macd.sign, pre_macd.dea_)
            macd = round((dif_ - dea_) * 2, 5)
            node.macd.fast = fast
            node.macd.slow = slow
            node.macd.dif_ = dif_
            node.macd.dea_ = dea_
            node.macd.macd = macd

        @staticmethod
        def _ema(price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 5)


############################################################
class Market:
    def __init__(self, sid: str):
        self.dayBus: Bus = Bus(maxlen=180)  # 日线数据
        self.minBus: Bus = Bus(maxlen=240)  # 分钟数据
        self.sid = sid
        self.ctx = {}

    def next(self, tick):
        node = Node(tick)
        self.bars.append(node)
        Line.Ema.calc(self)
        Line.Macd.calc(self)


############################################################
class Var:
    macd = Config.Macd()


class Env:
    markets: dict[str, Market] = {}

    @staticmethod
    def refresh(context):
        positions = context.portfolio.positions
        now_keys = set(positions.keys())
        env_keys = set(Env.markets.keys())
        for key in env_keys - now_keys:
            del Env.markets[key]
        for key in now_keys - env_keys:
            Env.markets[key] = Market(key)


############################################################
def initialize(context):
    """启动时执行一次"""
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    Env.markets.clear()
    Env.refresh(context)
    pass


def handle_data(context, data):
    """每个单位周期执行一次"""
    cur_time = context.blotter.current_dt
    if cur_time.minute % 5 == 0:
        Env.refresh(context)
    pass


def tick_data(context, data):
    """每个tick执行一次"""
    pass


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
