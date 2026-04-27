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
import yaml

# 你的 YAML 配置（多行字符串形式）
tree_conf = """
version: "1.0"
name: "分时行情决策树"

root:
  name: "趋势判断"
  field: "trend"
  rules:
    - operator: ">="
      value: 0
      next_node: "volume_check_up"
    - operator: "default"
      next_node: "volume_check_down"

nodes:
  volume_check_up:
    name: "上涨趋势-量能检查"
    field: "volume_ratio"
    rules:
      - operator: ">="
        value: 1.5
        action: "BUY"
        action_params:
          percent: 0.3
          reason: "放量上涨"
      - operator: "default"
        action: "HOLD"
        action_params:
          reason: "量能不足"

  volume_check_down:
    name: "下跌趋势-量能检查"
    field: "volume_ratio"
    rules:
      - operator: ">="
        value: 1.5
        action: "SELL"
        action_params:
          percent: 1.0
          reason: "放量下跌"
      - operator: "default"
        action: "HOLD"
        action_params:
          reason: "缩量下跌"
"""


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
            self.valuation: float = round(self.total_amount * self.last_price, 3)  # 市值
            self.principal: float = round(self.total_amount * self.cost_price, 3)  # 本金

    class Bar:
        def __init__(self, bar):
            self.datetime = bar.datetime
            self.volume: float = round(bar.volume, 3)  # 交易量
            self.money: float = round(bar.money, 3)  # 交易金额
            self.price: float = round(bar.price, 3)  # 最新价
            self.close: float = round(bar.close, 3)  # 收盘价
            self.open: float = round(bar.open, 3)  # 开盘价
            self.high: float = round(bar.high, 3)  # 最高价
            self.low: float = round(bar.low, 3)  # 最低价

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
    class Ema:
        def __init__(self):
            self.ema05: float = 0.0
            self.ema10: float = 0.0
            self.ema20: float = 0.0
            self.ema30: float = 0.0

    class Macd:
        def __init__(self):
            self.fast: float = 0.0
            self.slow: float = 0.0
            self.ema_: float = 0.0
            self.dif_: float = 0.0
            self.dea_: float = 0.0
            self.macd: float = 0.0

    class Vwap:
        def __init__(self):
            self.volume: float = 0.0
            self.money: float = 0.0
            self.value: float = 0.0


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
        self.ema = Box.Ema()
        self.macd = Box.Macd()
        self.vwap = Box.Vwap()
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
class Util:
    @staticmethod
    def get_field_value(obj, path):
        val = obj
        for key in path.split('.'):
            if key.startswith('[') and key.endswith(']'):
                val = val[int(key[1:-1])]
            elif isinstance(val, dict):
                val = val[key]
            else:
                val = getattr(val, key)
        return val

    @staticmethod
    def do_compare(value, operator, target) -> bool:
        if operator == '<': return value < target
        if operator == '>': return value > target
        if operator == '<=': return value <= target
        if operator == '>=': return value >= target
        if operator == '!=': return value != target
        if operator == '==': return value == target
        if operator == 'in': return target[0] <= value <= target[1]
        return False


class Line:
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

        @staticmethod
        def next(bus: Bus, node: Node, price: float):
            curr_ema = node.ema
            prev_ema = bus.get(-2).ema
            curr_ema.ema05 = Line.Ema.ema(prev_ema.ema05, price, 5)
            curr_ema.ema10 = Line.Ema.ema(prev_ema.ema10, price, 10)
            curr_ema.ema20 = Line.Ema.ema(prev_ema.ema20, price, 20)
            curr_ema.ema30 = Line.Ema.ema(prev_ema.ema30, price, 30)

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

    class Vwap:
        @staticmethod
        def calc(bus: Bus):
            node = bus.last()
            vwap = node.vwap
            if len(bus) == 1:
                vwap.volume = node.bar.volume
                vwap.money = node.bar.money
            else:
                prev_vwap = bus.get(-2).vwap
                vwap.volume = prev_vwap.volume + node.bar.volume
                vwap.money = prev_vwap.money + node.bar.money
            vwap.value = round(vwap.volume / vwap.money, 4)


class Tree:
    def __init__(self):
        """决策树引擎"""
        self.tree_conf = yaml.safe_load(tree_conf)
        self.node_list = self.tree_conf.get('nodes', {})
        self.root_node = self.tree_conf.get('root')
        self.curr_node = self.root_node
        self.path = [self.curr_node.get('name')]

    def evaluate(self, market: Market) -> str | None:
        """决策判断"""
        branches = self.curr_node.get('branches', [])
        if not branches: return None
        for branch in branches:
            conditions = self.curr_node.get('conditions', [])
            is_success = Tree._check_conditions(market, conditions)
            if not is_success: continue
            next_node = branch.get('next_node')
            if next_node is not None:
                self._to_next(next_node)
            return branch.get('action')
        return None

    def _to_next(self, node_name: str):
        next_node = self.node_list.get(node_name)
        if not next_node: return
        self.curr_node = next_node
        self.path.append(next_node.get('name'))

    @staticmethod
    def _check_conditions(market: Market, conditions: list[dict]) -> bool:
        if not conditions: return True
        for condition in conditions:
            field = condition.get('field')
            actor = condition.get('actor')
            target = condition.get('value')
            origin = Util.get_field_value(market, field)
            if origin is None: return False
            result = Util.do_compare(origin, actor, target)
            if not result: return False
        return True


############################################################
class Broker:
    @staticmethod
    def can_not_buy(bus: Bus) -> bool:
        """不能买"""
        macd = bus.last().macd
        if macd.dif_ < 0 and macd.dea_ < 0 and macd.macd < 1:
            return True
        if macd.dif_ > 2 and macd.dea_ > 2 and macd.macd > -2:
            return False
        if macd.macd < -0.5:
            return True
        return False

    @staticmethod
    def ema_trend(ema: Box.Ema) -> str:
        """EMA趋势"""
        if ema.ema05 < ema.ema10 < ema.ema20 < ema.ema30: return 'fall'
        if ema.ema05 > ema.ema10 > ema.ema20 > ema.ema30: return 'rise'
        return 'flat'

    @staticmethod
    def macd_trend(macd: Box.Macd):
        """MACD趋势"""
        if macd.dif_ < 0 and macd.dea_ < 0 and macd.macd < 0: return 'fall'
        if macd.dif_ > 0 and macd.dea_ > 0 and macd.macd > 0: return 'rise'
        return 'flat'


class Trader:
    def __init__(self, market: Market, buy: Callable, sell: Callable):
        self.market = market
        self.tree = Tree()
        self.sell = sell
        self.buy = buy

    def tick_trading(self, market: Market):
        if len(market.tikBus) == 0: return
        node = market.tikBus.last()
        open_pct = (node.tik.open_px - node.tik.preclose_px) / node.tik.preclose_px * 100
        wave_pct = (node.ema.ema05 - node.tik.open_px) / node.tik.preclose_px * 100
        if open_pct > -2.5:
            # 开盘大于-2.5%，急跌 --> 卖出
            if wave_pct > -0.5: return
            if Broker.ema_trend(node.ema) != 'fall': return
            if Broker.macd_trend(node.macd) != 'fall': return
            self._do_sell(market)
        else:
            # 开盘小于-2.5%，急拉 --> 买入
            if wave_pct < 0.5: return
            if Broker.ema_trend(node.ema) != 'rise': return
            if Broker.macd_trend(node.macd) != 'rise': return
            self._do_buy(market)

    def tail_trading(self, market: Market):
        if market.status.has_buy:
            self._do_sell(market)
            return
        if market.status.has_sell:
            self._do_buy(market)
            return

    def midd_trading(self, market: Market):
        if market.status.has_buy and market.status.has_sell: return
        action = self.tree.evaluate(market)
        if action == 'buy':
            self._do_buy(market)
            return
        if action == 'sell':
            self._do_sell(market)

    def _do_buy(self, market: Market):
        if market.status.has_buy: return
        if market.status.can_not_buy: return
        pos = market.nowPos
        last_price = pos.last_price
        buy_amount = round(Var.base_fund / last_price / 100) * 100
        self.buy(market.symbol, buy_amount, limit_price=last_price + 0.003)
        market.status.has_buy = True
        market.status.buy_price = last_price

    def _do_sell(self, market: Market):
        if market.status.has_sell: return
        pos = market.nowPos
        last_price = pos.last_price
        if pos.avail_amount * pos.last_price < 500: return
        base_amount = round(Var.base_fund / last_price / 100) * 100
        able_amount = min(pos.avail_amount, base_amount)
        sell_amount = able_amount - (0 if able_amount < pos.total_amount else 100)
        self.sell(market.symbol, -sell_amount, limit_price=last_price - 0.003)
        market.status.has_sell = True
        market.status.sell_price = last_price


############################################################
class Status:
    def __init__(self):
        self.has_buy = False  # 是否已经买入
        self.has_sell = False  # 是否已经卖出
        self.buy_price = 0.0  # 买入时的价格
        self.sell_price = 0.0  # 卖出时的价格

        self.base_price = 0.0  # 基准价格（昨日收盘价）
        self.open_price = 0.0  # 开盘价格
        self.curr_price = 0.0  # 最新价格

        self.open_pct = 0.0  # 开盘价（%）
        self.curr_pct = 0.0  # 最新价（%）
        self.wave_pct = 0.0  # 波动价（%）

        self.vwap_diff = 0.0  # VWAP与MA的差值
        self.ema_trend = 'flat'  # EMA 趋势
        self.macd_trend = 'flat'  # MACD 趋势
        self.can_not_buy = False  # 是否不能买入

    def update(self, market: Market):
        self.curr_pct = round((self.curr_price - self.base_price) / self.base_price * 100, 2)
        self.wave_pct = round((self.curr_price - self.open_price) / self.base_price * 100, 2)
        node = market.fenBus.last()
        self.vwap_diff = node.ema.ema05 - node.vwap.value
        self.ema_trend = Broker.ema_trend(node.ema)
        self.macd_trend = Broker.macd_trend(node.macd)
        self.can_not_buy = Broker.can_not_buy(market.dayBus)


class Market:
    def __init__(self, symbol: str):
        self.dayBus = Bus(maxlen=120, conf=Var.Day)  # 日线数据
        self.fenBus = Bus(maxlen=240, conf=Var.Fen)  # 分钟数据
        self.tikBus = Bus(maxlen=360, conf=Var.Fen)  # tick数据
        self.symbol = symbol  # 股票代码
        self.nowPos = None  # 当前持仓
        self.status = Status()  # 上下文数据

    def prep(self, bars: list):
        if not bars:
            return self
        for bar in bars:
            self.dayBus.add(Node(bar))
            Line.Macd.calc(self.dayBus)
        self.status.base_price = self.dayBus.last().bar.close
        self.status.can_not_buy = Broker.can_not_buy(self.dayBus)
        return self

    def next(self, pos, bar):
        # 分钟数据
        node = Node(bar)
        self.fenBus.add(node)
        Line.Ema.calc(self.fenBus)
        Line.Macd.calc(self.fenBus)
        Line.Vwap.calc(self.fenBus)
        # 日线数据
        self.dayBus.rollback()
        self.dayBus.add(Node(bar).mark(-1))
        Line.Macd.calc(self.dayBus)

        # 状态数据
        if len(self.fenBus) == 1:
            self.status.open_price = node.bar.open
            base_price = self.status.base_price
            self.status.open_pct = round((node.bar.open - base_price) / base_price * 100, 2)
        self.nowPos = Bin.Pos(pos)
        self.status.last_price = self.nowPos.last_price
        self.status.update(self)

    def tick_next(self, pos, tik):
        self.nowPos = Bin.Pos(pos)
        self.tikBus.add(Node().tick(tik))
        Line.Ema.calc(self.tikBus)
        Line.Macd.calc(self.tikBus)


############################################################
class Env:
    symbols: list[str] = ['000001.SS', '000852.SS']
    indexes: dict[str, Market] = {}
    markets: dict[str, Market] = {}
    traders: dict[str, Trader] = {}

    @staticmethod
    def launch(context):
        Env.clear()
        positions = context.portfolio.positions
        pos_codes = list(positions.keys())
        all_codes = pos_codes + Env.symbols
        history = get_history(120, frequency='1d', security_list=all_codes)
        for code in Env.symbols:
            Env.indexes[code] = Env.market(history, code)
        for code in pos_codes:
            Env.markets[code] = Env.market(history, code)

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
            Env.markets[code] = Env.market(history, code)

    @staticmethod
    def market(history, symbol):
        data = history.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in data.iterrows()]
        return Market(symbol).prep(bars)

    @staticmethod
    def trader(market: Market):
        symbol = market.symbol
        trader = Env.traders.get(symbol)
        if trader is None:
            Env.traders[symbol] = Trader(market, order, order)
        return trader

    @staticmethod
    def clear():
        Env.indexes.clear()
        Env.markets.clear()
        Env.traders.clear()


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
    history = get_history(1, frequency='1m', security_list=Env.symbols)
    for symbol, market in Env.indexes.items():
        data = history.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in data.iterrows()]
        market.next(None, bars[-1])

    positions = context.portfolio.positions
    for symbol, market in Env.markets.items():
        pos = positions.get(symbol)
        bar = data.get(symbol)
        market.next(pos, bar)
        trader = Env.trader(market)
        if Var.tick_time < cur_time < Var.back_time:
            trader.midd_trading(market)
        if cur_time >= Var.back_time:
            trader.tail_trading(market)

    # 加载持仓
    Env.reload(context)


def tick_data(context, data):
    """每个tick执行一次"""
    if not is_trade():
        return
    cur_time = context.blotter.current_dt.time
    if cur_time > Var.tick_time:
        return
    positions = context.portfolio.positions
    for symbol, market in Env.markets.items():
        pos = positions.get(symbol)
        tick = data[symbol]['tick'].iloc[0]
        market.tick_next(pos, tick)
        trader = Env.trader(market)
        trader.tick_trading(market)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    Env.clear()
    pass
