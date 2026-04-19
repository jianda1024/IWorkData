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


class Var:
    base_fund = 3200  # 交易基础金额（元）
    open_time = '09:45:00'  # 开启交易时间（HH:MM:SS）
    back_time = '14:55:00'  # 补仓买回时间（HH:MM:SS）

    class Macd:
        fast = 12  # 快线周期
        slow = 26  # 慢线周期
        sign = 9  # 信号线周期
        base = 5  # 基准周期


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

    class Ema:
        def __init__(self):
            self.ema05: float = 0.0
            self.ema10: float = 0.0
            self.ema20: float = 0.0
            self.ema30: float = 0.0
            self.dif10: float = 0.0
            self.dif20: float = 0.0
            self.dif30: float = 0.0

    class Macd:
        def __init__(self):
            self.fast: float = 0.0
            self.slow: float = 0.0
            self.ema_: float = 0.0
            self.dif_: float = 0.0
            self.dea_: float = 0.0
            self.macd: float = 0.0

    class Node:
        def __init__(self, bar, flag=0):
            self.idx: str = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
            self.bar: Bin.Bar = Bin.Bar(bar)
            self.ema: Bin.Ema = Bin.Ema()
            self.macd: Bin.Macd = Bin.Macd()
            self.flag: int = flag


class Bus:
    def __init__(self, maxlen=None):
        self.data: deque[Bin.Node] = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.data)

    def add(self, node: Bin.Node):
        self.data.append(node)

    def get(self, idx: int) -> Bin.Node:
        return self.data[idx]

    def last(self) -> Bin.Node:
        return self.data[-1]

    def rollback(self):
        node = self.data.pop()
        if node.flag >= 0:
            self.data.append(node)


class Line:
    class Ema:
        @staticmethod
        def calc(bus: Bus):
            node = bus.last()
            price = node.bar.price
            curr_ema = node.ema
            if len(bus) == 1:
                Line.Ema.first(curr_ema, price)
            else:
                prev_ema = bus.get(-2).ema
                Line.Ema.next(prev_ema, curr_ema, price)

        @staticmethod
        def first(curr_ema: Bin.Ema, price: float):
            curr_ema.ema05 = price
            curr_ema.ema10 = price
            curr_ema.ema20 = price
            curr_ema.ema30 = price
            curr_ema.ema60 = price
            curr_ema.dif10 = 0.0
            curr_ema.dif20 = 0.0
            curr_ema.dif30 = 0.0

        @staticmethod
        def next(prev_ema: Bin.Ema, curr_ema: Bin.Ema, price: float):
            curr_ema.ema05 = Line.Ema.ema(prev_ema.ema05, price, 5)
            curr_ema.ema10 = Line.Ema.ema(prev_ema.ema10, price, 10)
            curr_ema.ema20 = Line.Ema.ema(prev_ema.ema20, price, 20)
            curr_ema.ema30 = Line.Ema.ema(prev_ema.ema30, price, 30)
            curr_ema.dif10 = round((curr_ema.ema05 - curr_ema.ema10) / curr_ema.ema05 * 100, 3)
            curr_ema.dif20 = round((curr_ema.ema05 - curr_ema.ema20) / curr_ema.ema05 * 100, 3)
            curr_ema.dif30 = round((curr_ema.ema05 - curr_ema.ema30) / curr_ema.ema05 * 100, 3)

        @staticmethod
        def ema(prev_val: float, price: float, period: int):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

    class Macd:
        @staticmethod
        def calc(bus: Bus):
            node = bus.last()
            price = node.bar.price
            curr_macd = node.macd
            if len(bus) == 1:
                curr_macd.fast = price
                curr_macd.slow = price
                curr_macd.ema_ = price
                curr_macd.dif_ = 0.0
                curr_macd.dea_ = 0.0
                curr_macd.macd = 0.0
            else:
                prev_macd = bus.get(-2).macd
                Line.Macd.next(prev_macd, curr_macd, price)

        @staticmethod
        def next(prev_macd: Bin.Macd, curr_macd: Bin.Macd, price: float):
            fast = Line.Ema.ema(prev_macd.fast, price, Var.Macd.fast)
            slow = Line.Ema.ema(prev_macd.slow, price, Var.Macd.slow)
            ema_ = Line.Ema.ema(prev_macd.ema_, price, Var.Macd.base)
            dif_ = round((fast - slow) / ema_ * 100, 4)
            dea_ = Line.Ema.ema(prev_macd.dea_, dif_, Var.Macd.sign)
            macd = round((dif_ - dea_) * 2, 4)
            curr_macd.fast = fast
            curr_macd.slow = slow
            curr_macd.ema_ = ema_
            curr_macd.dif_ = dif_
            curr_macd.dea_ = dea_
            curr_macd.macd = macd


############################################################
class Broker:
    @staticmethod
    def buy_amount(market: Market) -> float:
        """买入数量"""
        amount = Var.base_fund / market.nowPos.last_price
        return round(amount / 100) * 100

    @staticmethod
    def sell_amount(market: Market) -> float:
        """卖出数量"""
        pos = market.nowPos
        base_amount = round(Var.base_fund / pos.last_price / 100) * 100
        sell_amount = min(pos.avail_amount, base_amount)
        if sell_amount < pos.total_amount:
            return sell_amount
        return sell_amount - 100

    @staticmethod
    def is_today_buy(market: Market) -> bool:
        """当天是否执行买入"""
        macd = market.dayBus.last().macd
        if macd.dif_ < 0 and macd.dea_ < 0 and macd.macd < 1:
            return False
        if macd.dif_ > 2 and macd.dea_ > 2 and macd.macd > -2:
            return True
        if macd.macd < -0.5:
            return False
        return True

    @staticmethod
    def is_sell_day(market: Market) -> bool:
        """当天是否执行卖出"""
        pos = market.nowPos
        return pos.avail_amount * pos.last_price >= 1000

    



class Market:
    def __init__(self, symbol: str):
        self.dayBus = Bus(maxlen=120)  # 日线数据
        self.fenBus = Bus(maxlen=240)  # 分钟数据
        self.symbol = symbol  # 股票代码
        self.nowPos = None  # 当前持仓
        self.ctxMap = {}  # 上下文数据

    def prepare(self, bars: list):
        if not bars:
            return self
        for bar in bars:
            self.dayBus.add(Bin.Node(bar))
            Line.Ema.calc(self.dayBus)
            Line.Macd.calc(self.dayBus)
        return self

    def running(self, pos, bar):
        self.nowPos = Bin.Pos(pos)
        self.fenBus.add(Bin.Node(bar))
        Line.Ema.calc(self.fenBus)
        Line.Macd.calc(self.fenBus)
        self.dayBus.rollback()
        self.dayBus.add(Bin.Node(bar, flag=-1))
        Line.Ema.calc(self.dayBus)
        Line.Macd.calc(self.dayBus)

    def trading(self, buy: Callable, sell: Callable):
        pass


############################################################
class Env:
    symbols: list[str] = ['000001.SS', '000852.SS']
    indexes: dict[str, Market] = {}
    markets: dict[str, Market] = {}

    @staticmethod
    def launch(context):
        Env.indexes.clear()
        Env.markets.clear()
        positions = context.portfolio.positions
        pos_codes = list(positions.keys())
        all_codes = pos_codes + Env.symbols
        history = get_history(120, frequency='1d', security_list=all_codes)
        for code in Env.symbols:
            Env.indexes[code] = Env._new_market(history, code)
        for code in pos_codes:
            Env.markets[code] = Env._new_market(history, code)

    @staticmethod
    def reload(context):
        positions = context.portfolio.positions
        pos_codes = set(positions.keys())
        mkt_codes = set(Env.markets.keys())
        for code in mkt_codes - pos_codes:
            del Env.markets[code]
        new_codes = list(pos_codes - mkt_codes)
        if not new_codes:
            return
        history = get_history(120, frequency='1d', security_list=new_codes)
        for code in new_codes:
            Env.markets[code] = Env._new_market(history, code)

    @staticmethod
    def _new_market(history, symbol):
        data = history.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in data.iterrows()]
        return Market(symbol).prepare(bars)


############################################################
def initialize(context):
    """启动时执行一次"""
    set_commission(commission_ratio=0.00005, min_commission=0.5, type="ETF")
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    Env.launch(context)
    pass


def handle_data(context, data):
    """每个单位周期执行一次"""
    cur_time = context.blotter.current_dt
    if cur_time.minute % 5 == 0:
        Env.reload(context)

    history = get_history(1, frequency='1m', security_list=Env.symbols)
    for symbol, market in Env.indexes.items():
        data = history.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in data.iterrows()]
        market.running(None, bars[-1])

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
