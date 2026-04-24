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
    tick_time = time(9, 33, 0)  # tick时间
    back_time = time(14, 55, 0)  # 补仓买回时间

    class Day:
        macd_fast = 12
        macd_slow = 26
        macd_sign = 9

    class Fen:
        macd_fast = 13
        macd_slow = 60
        macd_sign = 5


class Bin:
    class Pos:
        def __init__(self, pos):
            self.symbol: str = getattr(pos, 'sid', '')  # 股票代码
            self.total_amount: float = getattr(pos, 'amount', 0.0)  # 总持仓数量
            self.avail_amount: float = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
            self.last_price: float = getattr(pos, 'last_sale_price', 0.0)  # 最新价格
            self.cost_price: float = getattr(pos, 'cost_basis', 0.0)  # 成本价格
            self.valuation: float = round(self.total_amount * self.last_price, 2)  # 市值
            self.principal: float = round(self.total_amount * self.cost_price, 2)  # 本金

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

    class Tik:
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


class Box:
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

    class Mava:
        def __init__(self):
            self.volume: float = 0.0
            self.money: float = 0.0
            self.value: float = 0.0

    class Macd:
        def __init__(self):
            self.fast: float = 0.0
            self.slow: float = 0.0
            self.ema_: float = 0.0
            self.dif_: float = 0.0
            self.dea_: float = 0.0
            self.macd: float = 0.0


class Bus:
    def __init__(self, maxlen=None, conf=None):
        self.data: deque[Node] = deque(maxlen=maxlen)
        self.conf = conf

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


class Node:
    def __init__(self, bar=None):
        self.avl = Box.Avl()
        self.ema = Box.Ema()
        self.macd = Box.Macd()

        self.state: int = 0
        self.tik: Bin.Tik | None = None
        self.bar: Bin.Bar | None = None if bar is None else Bin.Bar(bar)

    def tick(self, tick) -> Node:
        self.tik = Bin.Tik(tick)
        return self

    def mark(self, state):
        self.state = state
        return self

    def price(self):
        if self.bar is not None:
            return self.bar.price
        if self.tik is not None:
            return self.tik.last_px
        return 0.0


############################################################
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
            price = node.price()
            if len(bus) == 1:
                Line.Ema.first(node, price)
            else:
                Line.Ema.next(bus, node, price)

        @staticmethod
        def first(node: Node, price: float):
            curr_ema = node.ema
            curr_ema.ema05 = price
            curr_ema.ema10 = price
            curr_ema.ema20 = price
            curr_ema.ema30 = price
            curr_ema.ema60 = price
            curr_ema.dif10 = 0.0
            curr_ema.dif20 = 0.0
            curr_ema.dif30 = 0.0

        @staticmethod
        def next(bus: Bus, node: Node, price: float):
            curr_ema = node.ema
            prev_ema = bus.get(-2).ema
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
            price = node.price()
            if len(bus) == 1:
                Line.Macd._first(node, price)
            else:
                Line.Macd._next(bus, node, price)

        @staticmethod
        def _first(node: Node, price: float):
            curr_macd = node.macd
            curr_macd.fast = price
            curr_macd.slow = price
            curr_macd.ema_ = price
            curr_macd.dif_ = 0.0
            curr_macd.dea_ = 0.0
            curr_macd.macd = 0.0

        @staticmethod
        def _next(bus: Bus, node: Node, price: float):
            curr_macd = node.macd
            prev_macd = bus.get(-2).macd
            curr_macd.fast = Line.Ema.ema(prev_macd.fast, price, bus.conf.macd_fast)
            curr_macd.slow = Line.Ema.ema(prev_macd.slow, price, bus.conf.macd_slow)
            curr_macd.ema_ = Line.Ema.ema(prev_macd.ema_, price, 5)
            curr_macd.dif_ = round((curr_macd.fast - curr_macd.slow) / curr_macd.ema_ * 100, 4)
            curr_macd.dea_ = Line.Ema.ema(prev_macd.dea_, curr_macd.dif_, bus.conf.macd_sign)
            curr_macd.macd = round((curr_macd.dif_ - curr_macd.dea_) * 2, 4)


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
    def unable_buy(market: Market) -> bool:
        """不能买"""
        return market.status.has_buy

    @staticmethod
    def unable_sell(market: Market) -> bool:
        """不能卖"""
        if market.status.has_sell:
            return True
        pos = market.nowPos
        return pos.avail_amount * pos.last_price < 500

    @staticmethod
    def no_buy_day(market: Market) -> bool:
        """今天不买"""
        macd = market.dayBus.last().macd
        if macd.dif_ < 0 and macd.dea_ < 0 and macd.macd < 1:
            return True
        if macd.dif_ > 2 and macd.dea_ > 2 and macd.macd > -2:
            return False
        if macd.macd < -0.5:
            return True
        return False

    @staticmethod
    def no_fast_fall(node: Node) -> bool:
        if node.macd.dif_ >= 0: return True
        if node.macd.dea_ >= 0: return True
        if node.macd.macd >= 0: return True
        if node.ema.ema05 >= node.ema.ema10: return True
        if node.ema.ema10 >= node.ema.ema20: return True
        if node.ema.ema20 >= node.ema.ema30: return True
        return False

    @staticmethod
    def no_fast_rise(node: Node) -> bool:
        if node.macd.dif_ <= 0: return True
        if node.macd.dea_ <= 0: return True
        if node.macd.macd <= 0: return True
        if node.ema.ema05 <= node.ema.ema10: return True
        if node.ema.ema10 <= node.ema.ema20: return True
        if node.ema.ema20 <= node.ema.ema30: return True
        return False


class Trader:
    @staticmethod
    def tick_trading(market: Market, buy: Callable, sell: Callable):
        if len(market.tikBus) == 0: return
        node = market.tikBus.last()
        open_pct = (node.tik.open_px - node.tik.preclose_px) / node.tik.preclose_px * 100
        wave_pct = (node.ema.ema05 - node.tik.open_px) / node.tik.preclose_px * 100
        if open_pct > -2.5:
            # 开盘大于-2.5%，急跌 --> 卖出
            if wave_pct > -0.5: return
            if market.status.has_buy: return
            if Broker.unable_sell(market): return
            if Broker.no_fast_fall(node): return
            amount = Broker.sell_amount(market)
            sell(market.symbol, -amount, limit_price=node.tik.last_px - 0.003)
            market.mark_sell()
        else:
            # 开盘小于-2.5%，急拉 --> 买入
            if wave_pct < 0.5: return
            if market.status.has_sell: return
            if Broker.no_buy_day(market): return
            if Broker.no_fast_rise(node): return
            amount = Broker.buy_amount(market)
            buy(market.symbol, amount, limit_price=node.tik.last_px + 0.003)
            market.mark_buy()

    @staticmethod
    def back_trading(market: Market, buy: Callable):
        if market.status.has_buy: return
        pass

    @staticmethod
    def trading(market: Market, buy: Callable, sell: Callable):
        node = market.fenBus.last()


############################################################
class Status:
    def __init__(self):
        self.has_buy = False
        self.has_sell = False
        self.base_price = 0.0
        self.buy__price = 0.0
        self.sell_price = 0.0


class Market:
    def __init__(self, symbol: str):
        self.dayBus = Bus(maxlen=120, conf=Var.Day)  # 日线数据
        self.fenBus = Bus(maxlen=240, conf=Var.Fen)  # 分钟数据
        self.tikBus = Bus(maxlen=360, conf=Var.Fen)  # tick数据
        self.symbol = symbol  # 股票代码
        self.nowPos = None  # 当前持仓
        self.status = Status()  # 上下文数据

    def prepare(self, bars: list):
        if not bars:
            return self
        for bar in bars:
            self.dayBus.add(Node(bar))
            Line.Ema.calc(self.dayBus)
            Line.Macd.calc(self.dayBus)
        self.status.base_price = self.dayBus.last().bar.close
        return self

    def running(self, pos, bar):
        self.nowPos = Bin.Pos(pos)
        self.fenBus.add(Node(bar))
        Line.Ema.calc(self.fenBus)
        Line.Macd.calc(self.fenBus)
        self.dayBus.rollback()
        self.dayBus.add(Node(bar).mark(-1))
        Line.Ema.calc(self.dayBus)
        Line.Macd.calc(self.dayBus)

    def trading(self, buy: Callable, sell: Callable):
        Trader.trading(self, buy, sell)

    def tick_running(self, tik):
        self.tikBus.add(Node().tick(tik))
        Line.Ema.calc(self.tikBus)
        Line.Macd.calc(self.tikBus)

    def tick_trading(self, sell: Callable):
        Trader.tick_trading(self, sell)

    def back_trading(self, buy: Callable):
        Trader.back_trading(self, buy)

    def mark_buy(self):
        self.status.has_buy = True
        self.status.buy__price = self.nowPos.last_price

    def mark_sell(self):
        self.status.has_sell = True
        self.status.sell_price = self.nowPos.last_price


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
