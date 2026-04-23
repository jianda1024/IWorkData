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
from datetime import time
from types import SimpleNamespace
from typing import Callable

import pandas as pd


class Var:
    base_fund = 3200  # 交易基础金额（元）
    tick_time = time(9, 36, 0)  # tick时间
    back_time = time(14, 55, 0)  # 补仓买回时间

    class Macd:
        fast = 12  # 快线周期
        slow = 26  # 慢线周期
        sign = 9  # 信号线周期


class Bus:
    def __init__(self, maxlen=None):
        self.data: deque[Node] = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.data)

    def add(self, node: Node):
        self.data.append(node)

    def get(self, idx: int) -> Node:
        return self.data[idx]

    def last(self) -> Node:
        return self.data[-1]

    def rollback(self):
        node = self.data.pop()
        if node.state >= 0:
            self.data.append(node)


class Bin:
    class Ctx:
        def __init__(self):
            self.has_buy = False
            self.has_sell = False
            self.base_price = 0.0

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

    class Avl:
        def __init__(self):
            self.volume: float = 0.0
            self.money: float = 0.0
            self.value: float = 0.0

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

    class Tick:
        def __init__(self, tick: pd.Series):
            self.trade_status = ''  # 交易状态TRADE交易中
            self.hsTimeStamp = 0.0  # 时间戳，格式为YYYYMMDDHHMISS

            self.amount = 0.0  # 持仓量
            self.business_amount = 0.0  # 成交数量
            self.business_amount_in = 0.0  # 内盘成交量
            self.business_amount_out = 0.0  # 外盘成交量
            self.business_balance = 0.0  # 成交金额

            self.preclose_px = 0.0  # 昨收盘价
            self.wavg_px = 0.0  # 加权平均价
            self.open_px = 0.0  # 今开盘价
            self.last_px = 0.0  # 最新价
            self.high_px = 0.0  # 最高价
            self.low_px = 0.0  # 最低价

            self.offer_grp = {}  # 卖档位
            self.bid_grp = {}  # 买档位

            data = tick.to_dict()
            for key, value in data.items():
                setattr(self, str(key), value)


class Node:
    def __init__(self):
        self.state: int = 0
        self.index: str = ''
        self.price: float = 0.0
        self.avl: Bin.Avl = Bin.Avl()
        self.ema: Bin.Ema = Bin.Ema()
        self.macd: Bin.Macd = Bin.Macd()
        self.bar: Bin.Bar | None = None
        self.tick: Bin.Tick | None = None

    def new(self, bar, state=0) -> Node:
        self.bar = Bin.Bar(bar)
        self.index = self.bar.datetime.strftime('%H:%M:%S')
        self.price = self.bar.close
        self.state = state
        return self

    def add(self, tick) -> Node:
        self.tick = Bin.Tick(tick)
        self.price = self.tick.last_px
        self.index = pd.to_datetime(self.tick.hsTimeStamp, format="%Y%m%d%H%M%S").strftime("%H:%M:%S")
        return self


class Line:
    class Avl:
        @staticmethod
        def calc(bus: Bus):
            node = bus.last()
            avl = node.avl
            if len(bus) == 1:
                avl.volume = node.bar.volume
                avl.money = node.bar.money
            else:
                prev_avl = bus.get(-2).avl
                avl.volume = prev_avl.volume + node.bar.volume
                avl.money = prev_avl.money + node.bar.money
            avl.value = avl.volume / avl.money

    class Ema:
        @staticmethod
        def calc(bus: Bus):
            node = bus.last()
            price = node.price
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
            price = node.price
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
            ema_ = Line.Ema.ema(prev_macd.ema_, price, 5)
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
    def is_buy_day(market: Market) -> bool:
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
    def unable_buy(market: Market) -> bool:
        """是否不能买入"""
        if market.ctxMap.has_buy:
            return True
        node = market.fenBus.last()
        if node.avl.value < node.ema.ema05:
            return True
        return False

    @staticmethod
    def unable_sell(market: Market) -> bool:
        """是否不能卖出"""
        if market.ctxMap.has_sell:
            return True
        pos = market.nowPos
        return pos.avail_amount * pos.last_price < 1000

    @staticmethod
    def buy(market: Market, buy: Callable):
        if not Broker.is_buy_day(market):
            return

    @staticmethod
    def sell(market: Market, sell: Callable):
        if Broker.unable_sell(market):
            return

    @staticmethod
    def tick_trading(market: Market, sell: Callable):
        if Broker.unable_sell(market): return
        if len(market.tikBus) == 0: return
        node = market.tikBus.last()
        if node.macd.dif_ >= 0: return
        if node.macd.dea_ >= 0: return
        if node.macd.macd >= 0: return
        if node.ema.ema05 >= node.ema.ema10: return
        if node.ema.ema10 >= node.ema.ema20: return
        if node.ema.ema20 >= node.ema.ema30: return
        # 开盘急跌
        tick = node.tick
        fall = (node.ema.ema05 - tick.open_px) / tick.preclose_px
        if fall > -0.005: return
        # 卖出
        amount = Broker.sell_amount(market)
        sell(market.symbol, -amount, limit_price=tick.last_px - 0.005)
        market.ctxMap.has_sell = True

    @staticmethod
    def back_trading(market: Market, buy: Callable):
        pass

    @staticmethod
    def trading(market: Market, buy: Callable, sell: Callable):
        node = market.fenBus.last()


class Market:
    def __init__(self, symbol: str):
        self.dayBus = Bus(maxlen=120)  # 日线数据
        self.fenBus = Bus(maxlen=240)  # 分钟数据
        self.tikBus = Bus(maxlen=360)  # tick数据
        self.symbol = symbol  # 股票代码
        self.nowPos = None  # 当前持仓
        self.ctxMap = Bin.Ctx()  # 上下文数据

    def prepare(self, bars: list):
        if not bars:
            return self
        for bar in bars:
            self.dayBus.add(Node().new(bar))
            Line.Ema.calc(self.dayBus)
            Line.Macd.calc(self.dayBus)
        self.ctxMap.base_price = self.dayBus.last().bar.close
        return self

    def running(self, pos, bar):
        self.nowPos = Bin.Pos(pos)
        self.fenBus.add(Node().new(bar))
        Line.Ema.calc(self.fenBus)
        Line.Macd.calc(self.fenBus)
        self.dayBus.rollback()
        self.dayBus.add(Node().new(bar, state=-1))
        Line.Ema.calc(self.dayBus)
        Line.Macd.calc(self.dayBus)

    def tick_running(self, tik):
        self.tikBus.add(Node().add(tik))
        Line.Ema.calc(self.tikBus)
        Line.Macd.calc(self.tikBus)

    def trading(self, buy: Callable, sell: Callable):
        Broker.trading(self, buy, sell)

    def tick_trading(self, sell: Callable):
        Broker.tick_trading(self, sell)

    def back_trading(self, buy: Callable):
        Broker.back_trading(self, buy)


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
    cur_time = context.blotter.current_dt.time()
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
        if Var.tick_time < cur_time < Var.back_time:
            market.trading(order, order)
        if cur_time >= Var.back_time:
            market.back_trading(order)


def tick_data(context, data):
    """每个tick执行一次"""
    if not is_trade():
        return
    cur_time = context.blotter.current_dt.time
    if cur_time > Var.tick_time:
        return
    for symbol, market in Env.markets.items():
        tick = data[symbol]['tick'].iloc[0]
        market.tick_running(tick)
        market.tick_trading(order)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    Env.markets.clear()
    pass
