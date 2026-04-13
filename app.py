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
from types import SimpleNamespace
from typing import Callable

import pandas as pd


class Var:
    base_fund: int = 3200  # 交易基础金额（元）
    back_time: str = '14:55:00'  # 补仓时间（HH:MM:SS）

    class Macd:
        fast: int = 12  # 快线周期
        slow: int = 26  # 慢线周期
        sign: int = 9  # 信号线周期


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

    class Tick:
        def __init__(self, tick: pd.Series):
            self.trade_status = 0.0  # 交易状态TRADE交易中
            self.hsTimeStamp = 0.0  # 时间戳，格式为YYYYMMDDHHMISS
            self.trade_mins = 0.0  # 交易时间，距离开盘已过多少分钟

            self.amount = 0.0  # 持仓量
            self.business_amount = 0.0  # 成交数量
            self.business_amount_in = 0.0  # 内盘成交量
            self.business_amount_out = 0.0  # 外盘成交量
            self.business_balance = 0.0  # 成交金额
            self.business_count = 0.0  # 成交笔数
            self.current_amount = 0.0  # 最近成交量(现手)

            self.up_px = 0.0  # 涨停价格
            self.down_px = 0.0  # 跌停价格
            self.preclose_px = 0.0  # 昨收价
            self.open_px = 0.0  # 今开盘价
            self.last_px = 0.0  # 最新成交价
            self.high_px = 0.0  # 最高价
            self.low_px = 0.0  # 最低价
            self.avg_px = 0.0  # 均价

            self.turnover_ratio = 0.0  # 换手率
            self.entrust_diff = 0.0  # 委差
            self.entrust_rate = 0.0  # 委比
            self.vol_ratio = 0.0  # 量比

            self.offer_grp = {}  # 卖档位
            self.bid_grp = {}  # 买档位

            data = tick.to_dict()
            for key, value in data.items():
                setattr(self, str(key), value)

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

    def add(self, node: Bin.Node):
        self.data.append(node)

    def get(self, idx: int) -> Bin.Node:
        return self.data[idx]

    def clear(self):
        self.data.clear()
        self.temp.clear()


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
            fast = Line.Macd._ema(price, Var.Macd.fast, pre_macd.fast)
            slow = Line.Macd._ema(price, Var.Macd.slow, pre_macd.slow)
            dif_ = round(fast - slow, 5)
            dea_ = Line.Macd._ema(dif_, Var.Macd.sign, pre_macd.dea_)
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
    def __init__(self, symbol: str):
        self.dayBus = Bus(maxlen=120)  # 日线数据
        self.fenBus = Bus(maxlen=240)  # 分钟数据
        self.tikBus = Bus(maxlen=480)  # 秒钟数据
        self.symbol = symbol  # 股票代码
        self.nowPos = None  # 当前持仓
        self.ctxMap = {}  # 上下文数据

    def prepare(self, bars: list):
        if not bars:
            return self
        for bar in bars:
            self.dayBus.add(Bin.Node(bar))
            Line.Sma.calc(self.dayBus)
            Line.Macd.calc(self.dayBus)
        return self

    def running(self, pos, bar):
        self.nowPos = Bin.Pos(pos)
        self.fenBus.add(Bin.Node(bar))
        Line.Sma.calc(self.fenBus)
        Line.Macd.calc(self.fenBus)

    def trading(self, buy: Callable, sell: Callable):
        pass


class Trader:
    def __init__(self, symbol: str):
        self.market = Market(symbol)

    @staticmethod
    def buy_amount(market: Market) -> float:
        """买入数量"""
        amount = Var.base_fund / market.nowPos.last_price
        return round(amount / 100) * 100

    @staticmethod
    def sell_amount(market: Market) -> float:
        """卖出数量"""
        pos = market.nowPos
        least_amount = round(1000 / pos.last_price / 100) * 100
        if pos.avail_amount < least_amount:
            return 0
        sell_amount = round(Var.base_fund / pos.last_price / 100) * 100
        if pos.avail_amount > sell_amount:
            return sell_amount
        if pos.avail_amount < pos.total_amount:
            return pos.avail_amount
        return pos.avail_amount - 100


############################################################
class Env:
    markets: dict[str, Market] = {}

    @staticmethod
    def reload(context):
        positions = context.portfolio.positions
        now_keys = set(positions.keys())
        env_keys = set(Env.markets.keys())
        for key in env_keys - now_keys:
            del Env.markets[key]

        symbols = list(now_keys - env_keys)
        if not symbols:
            return
        history = get_history(120, frequency='1d', security_list=symbols)
        for symbol in symbols:
            data = history.query(f'code in ["{symbol}"]')
            bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in data.iterrows()]
            Env.markets[symbol] = Market(symbol).prepare(bars)


############################################################
def initialize(context):
    """启动时执行一次"""
    set_commission(commission_ratio=0.00005, min_commission=0.5, type="ETF")
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    Env.markets.clear()
    Env.reload(context)
    pass


def handle_data(context, data):
    """每个单位周期执行一次"""
    cur_time = context.blotter.current_dt
    if cur_time.minute % 5 == 0:
        Env.reload(context)

    positions = context.portfolio.positions
    for symbol, market in Env.markets.items():
        pos = positions.get(symbol)
        bar = data.get(symbol)
        market.running(pos, bar)
        market.trading(order, order)


def tick_data(context, data):
    """每个tick执行一次"""
    pass


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    Env.markets.clear()
    pass
