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
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from datetime import time
from types import SimpleNamespace
from typing import Callable, Type, Any


class K:
    class Bas:
        base_price = 'base_price'
        base_amount = 'base_amount'

    class Pos:
        avail_amount = 'avail_amount'
        total_amount = 'total_amount'
        cost_price = 'cost_price'
        last_price = 'last_price'
        valuation = 'valuation'
        principal = 'principal'

    class Bar:
        datetime = 'datetime'
        volume = 'volume'
        money = 'money'
        price = 'price'
        close = 'close'
        open = 'open'
        high = 'high'
        low = 'low'

    class Ext:
        live_act = 'live_act'
        turn_nodes = 'turn_nodes'
        turn_turns = 'turn_turns'

    class Ema:
        fast = 'ema_fast'
        slow = 'ema_slow'

    class Smma:
        fast = 'smma_fast'
        slow = 'smma_slow'

    class Sma:
        ma5 = 'MA5'
        ma10 = 'MA10'
        ma20 = 'MA20'
        ma30 = 'MA30'
        ma60 = 'MA60'

    class Macd:
        fast = 'macd_fast'
        slow = 'macd_slow'
        sign = 'macd_sign'
        diff = 'macd_diff'
        dea = 'macd_dea'
        macd = 'macd'

    class Turn:
        node_idx = 'node_idx'
        node_val = 'node_idx'
        turn_lvl = 'turn_lvl'
        turn_idx = 'turn_idx'
        turn_val = 'turn_val'


class Act:
    def __init__(self, symbol: str, bar: dict):
        self.node_val = bar[K.Ema.fast]
        self.node_idx = bar[K.Bar.datetime]
        self.turn_idx = bar[K.Turn.turn_idx]
        self.turn_val = bar[K.Turn.turn_val]
        self.turn_lvl = bar[K.Turn.turn_lvl]
        self.symbol = symbol
        self.amount = 0
        self.status = 0

    def trade(self, amount: float):
        self.status = 1 if amount > 0 else -1 if amount < 0 else 0
        self.amount = amount
        return self


class Pos:
    def __init__(self, pos, bar):
        self.datetime_str = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.total_amount = getattr(pos, 'amount', 0.0)  # 总持仓数量
        self.avail_amount = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
        self.cost_price = getattr(pos, 'cost_basis', 0.0)  # 成本价格
        self.last_price = getattr(pos, 'last_sale_price', 0.0)  # 最新价格
        self.valuation = round(self.total_amount * self.last_price, 2)  # 市值
        self.principal = round(self.total_amount * self.cost_price, 2)  # 本金

    def into(self, bus: Bus):
        pos_dict = {
            K.Pos.avail_amount: self.avail_amount,
            K.Pos.total_amount: self.total_amount,
            K.Pos.cost_price: self.cost_price,
            K.Pos.last_price: self.last_price,
            K.Pos.valuation: self.valuation,
            K.Pos.principal: self.principal,
        }
        bus.put(self.datetime_str, pos_dict)


class Bar:
    class Node:
        def __init__(self):
            self.node_idx = ''  # 节点索引
            self.node_val = 0  # 节点均值
            self.node_lvl = 0  # 节点标记

    class Turn:
        def __init__(self):
            self.turn_idx = ''  # 拐点索引
            self.turn_val = 0  # 拐点均值
            self.turn_lvl = 0  # 拐点标记

            # 加/减仓额度，用于避免重复交易
            self.add_quotas = []
            self.sub_quotas = []

            # 最大/小顶点，用于判断是否有效的拐点
            self.MaxApex: Bar.Node | None = None
            self.MinApex: Bar.Node | None = None

    def __init__(self, bar):
        self.time_str = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.datetime = bar.datetime  # 时间
        self.volume = round(bar.volume, 2)  # 交易量
        self.money = round(bar.money, 2)  # 交易金额
        self.price = round(bar.price, 4)  # 最新价
        self.close = round(bar.close, 4)  # 收盘价
        self.open = round(bar.open, 4)  # 开盘价
        self.high = round(bar.high, 4)  # 最高价
        self.low = round(bar.low, 4)  # 最低价

    def into(self, bus: Bus):
        bar_dict = {
            K.Bar.datetime: self.datetime,
            K.Bar.volume: self.volume,
            K.Bar.money: self.money,
            K.Bar.price: self.price,
            K.Bar.close: self.close,
            K.Bar.open: self.open,
            K.Bar.high: self.high,
            K.Bar.low: self.low,
        }
        bus.put(self.time_str, bar_dict)


class Bus:
    def __init__(self):
        self.keys: list[str] = []
        self.data: list[dict] = []

    def at(self, idx: int) -> dict:
        return self.data[idx]

    def put(self, key: str, value: dict):
        self.keys.append(key)
        self.data.append(value)

    def update(self, idx: int, col: str, val: Any):
        self.data[idx][col] = val

    def modify(self, row: str, col: str, val: Any):
        idx = self.keys.index(row)
        self.data[idx][col] = val

    def key(self, idx: int) -> str:
        return self.keys[idx]

    def clear(self):
        self.keys.clear()
        self.data.clear()


class Env:
    def __init__(self, config: str, trader: str):
        self.config = config
        self.trader = trader


############################################################
class Var:
    class Bas:
        def __init__(self):
            self.base_funds = 8000  # 基准资金
            self.cost_limit = 1.50  # 成本上限（比例）
            self.loss_limit = 0.15  # 亏损上限（比例）
            self.gain_limit = 0.05  # 盈利上限（比例）
            self.start_qty = 3000  # 起步买卖金额
            self.least_qty = 1000  # 最低卖出金额

    class Ema:
        def __init__(self):
            self.fast = 20  # 快线周期
            self.slow = 30  # 慢线周期

    class Sma:
        def __init__(self):
            self.sma5 = 5
            self.sma10 = 10
            self.sma20 = 20
            self.sma30 = 30
            self.sma60 = 60

    class Smma:
        def __init__(self):
            self.fast = 10  # 快线周期
            self.slow = 30  # 慢线周期

    class Macd:
        def __init__(self):
            self.fast = 13  # 快线周期
            self.slow = 60  # 慢线周期
            self.sign = 5  # 信号线周期

    class Turn:
        def __init__(self):
            self.least_wave = 0.004  # 最小摆动（比例）

    class Rise:
        def __init__(self):
            self.add_quotas = [0.300]  # 加仓额度（比例）
            self.thresholds = [0.004]  # 加仓阈值（比例）
            self.upper_macd = 0.0025  # macd限制（比例）

    class Fall:
        def __init__(self):
            self.sub_quotas = [0.300, 0.400, 0.300]  # 减仓额度（比例）
            self.thresholds = [0.004, 0.007, 0.010]  # 减仓阈值（比例）
            self.lower_macd = -0.0025  # macd限制（比例）

    class Config:
        def __init__(self):
            self.bas: Var.Bas = Var.Bas()
            self.ema: Var.Ema = Var.Ema()
            self.smma: Var.Smma = Var.Smma()
            self.macd: Var.Macd = Var.Macd()
            self.turn: Var.Turn = Var.Turn()
            self.rise: Var.Rise = Var.Rise()
            self.fall: Var.Fall = Var.Fall()


class Biz:
    class Base:
        def __init__(self, cfg: Var.Config, tkt: Ticket):
            self.way = 'PTrder'
            self.cfg = cfg
            self.tkt = tkt

        def live_pos(self) -> dict:
            return self.tkt.posBus.at(-1)

        def live_day(self) -> dict:
            return self.tkt.dayBus.at(-1)

        def live_min(self) -> dict:
            return self.tkt.minBus.at(-1)

        def fit_sell_qty(self, plan_qty: int):
            """适配卖出的数量"""
            pos = self.live_pos()
            base_price = self.tkt.extMap.get(K.Bas.base_price)
            start_qty = round(self.cfg.bas.start_qty / base_price / 100) * 100
            least_qty = round(self.cfg.bas.least_qty / base_price / 100) * 100
            avail_qty = pos[K.Pos.avail_amount]

            # 计算需要保留的数量
            profit = pos[K.Pos.valuation] - pos[K.Pos.principal]  # 当前盈利
            unavail_qty = pos[K.Pos.total_amount] - avail_qty  # 不可用持仓数量
            profit_target = self.cfg.bas.base_funds * self.cfg.bas.gain_limit  # 盈利目标
            floor_qty = 0 if unavail_qty == 0 and profit >= profit_target else 100

            # 获取卖出的数量
            if avail_qty <= floor_qty:
                return 0
            if avail_qty <= least_qty:
                return avail_qty if floor_qty == 0 else 0
            sell_qty = max(start_qty, round(plan_qty / 100) * 100)
            if avail_qty <= sell_qty + least_qty:
                return avail_qty - floor_qty
            return sell_qty

        def is_over_budget(self) -> bool:
            """是否超过本金上限/亏损上限"""
            pos = self.live_pos()
            if pos[K.Pos.principal] > self.cfg.bas.base_funds * self.cfg.bas.cost_limit:
                return True
            if pos[K.Pos.principal] - pos[K.Pos.valuation] >= self.cfg.bas.base_funds * self.cfg.bas.loss_limit:
                return True
            return False

        def check_buy_quota(self):
            """检查加仓配额"""
            add_quotas = self.cfg.rise.add_quotas
            thresholds = self.cfg.rise.thresholds
            return self.__check_trade_quota(1, add_quotas, thresholds)

        def check_sell_quota(self):
            """检查减仓配额"""
            sub_quotas = self.cfg.fall.sub_quotas
            thresholds = self.cfg.fall.thresholds
            return self.__check_trade_quota(-1, sub_quotas, thresholds)

        def __check_trade_quota(self, status, quotas, thresholds):
            base_price = self.tkt.extMap.get(K.Bas.base_price)
            acts = [act for act in self.tkt.actSet if act.status == status]

            # 配额已经用完
            length = len(acts)
            if length >= len(quotas):
                return False, 0

            # 首次交易
            if length == 0:
                return True, quotas[length]

            # 距前一次交易，波动未达到阈值
            live_min = self.live_min()
            turn_idx = live_min[K.Turn.turn_idx]
            node_val = live_min[K.Turn.node_val]
            prev_turn_idx = acts[-1].turn_idx
            threshold_val = thresholds[length] * base_price
            if turn_idx == prev_turn_idx and abs(node_val - acts[-1].node_val) < threshold_val:
                return False, 0
            if turn_idx != prev_turn_idx and abs(node_val - live_min[K.Turn.turn_val]) < threshold_val:
                return False, 0

            # 有可用配额
            return True, quotas[length]

    class Trader(ABC, Base):

        @abstractmethod
        def is_buy(self) -> bool:
            pass

        @abstractmethod
        def buy_amount(self) -> float:
            pass

        @abstractmethod
        def is_sell(self) -> bool:
            pass

        @abstractmethod
        def sell_amount(self) -> float:
            pass

        def trading(self, buy: Callable, sell: Callable):
            """执行交易"""
            if self.is_buy():
                amount = self.buy_amount()
                self.__do_buy(buy, self.tkt.symbol, amount)
            if self.is_sell():
                amount = self.sell_amount()
                self.__do_sell(sell, self.tkt.symbol, amount)

        def __do_buy(self, func: Callable, symbol: str, amount: float):
            if amount <= 0:
                return
            if self.way == 'PTrader':
                func(symbol, amount)
            if self.way == 'BTrader':
                func(size=amount)
            self.__add_log(symbol, amount)

        def __do_sell(self, func: Callable, symbol: str, amount: float):
            if amount >= 0:
                return
            if self.way == 'PTrader':
                func(symbol, amount)
            if self.way == 'BTrader':
                func(size=-amount)
            self.__add_log(symbol, amount)

        def __add_log(self, symbol: str, amount: float):
            live_min = self.live_min()
            live_act = Act(symbol, live_min).trade(amount)
            self.tkt.actSet.append(live_act)
            self.tkt.extMap[K.Ext.live_act] = live_act

    class TurnTrader(Trader):
        def is_buy(self) -> bool:
            """判断是否买入"""
            live_day = self.live_day()
            live_min = self.live_min()
            if live_min[K.Bar.datetime].time() < time(9, 40, 0):
                return False  # 前5分钟
            if self.is_over_budget():
                return False  # 超过亏损上限 or 超过本金上限
            if live_day[K.Smma.fast] <= live_day[K.Smma.slow]:
                return False  # 日线下跌
            if live_min[K.Ema.fast] <= live_min[K.Ema.slow]:
                return False  # 分钟线下跌
            #if live_min[K.Macd.diff] <= 0 or live_min[K.Macd.dea] <= 0:
                #return False  # MACD线的diff、dea在零轴下
            if live_min[K.Macd.diff] <= live_min[K.Macd.dea]:
                return False  # MACD线的diff在dea下
            if live_min[K.Macd.macd] >= self.cfg.rise.upper_macd:
                return True  # MACD大于设定的上限
            # 决定买入
            return True

        def buy_amount(self) -> float:
            """买入数量"""
            has_quota, quota = self.check_buy_quota()
            if not has_quota:
                return 0
            base_price = self.tkt.extMap.get(K.Bas.base_price)
            buy_amount = max(self.cfg.bas.base_funds * quota, self.cfg.bas.start_qty) / base_price
            return round(buy_amount / 100) * 100

        def is_sell(self) -> bool:
            """判断是否卖出"""
            live_min = self.live_min()
            if live_min[K.Bar.datetime].time() < time(9, 40, 0):
                return False  # 前5分钟
            if live_min[K.Ema.fast] >= live_min[K.Ema.slow]:
                return False  # 分钟线上涨
            if live_min[K.Macd.macd] > 0.001:
                return False  # MACD上涨
            if live_min[K.Macd.macd] < self.cfg.fall.lower_macd:
                return True  # MACD小于设定的下限，则卖出
            # 决定卖出
            return True

        def sell_amount(self) -> float:
            """卖出数量"""
            has_quota, quota = self.check_sell_quota()
            if not has_quota:
                return 0
            plan_amount = self.cfg.bas.base_funds * quota
            sell_amount = self.fit_sell_qty(plan_amount)
            return -sell_amount


class Line:
    class L(ABC):
        def __init__(self, cfg: Var.Config):
            self.cfg: Var.Config = cfg

        @abstractmethod
        def _calc(self, price: float, period: int, prev_val: float):
            pass

        @staticmethod
        def _first(bus: Bus, key: str):
            price = bus.at(-1).get(K.Bar.close)
            bus.update(-1, key, price)

        def _next(self, bus: Bus, key: str, period: int):
            price = bus.at(-1).get(K.Bar.close)
            prev_val = bus.at(-2).get(key)
            next_val = self._calc(price, period, prev_val)
            bus.update(-1, key, next_val)

    class Ema(L):
        def first(self, bus: Bus):
            self._first(bus, K.Ema.fast)
            self._first(bus, K.Ema.slow)
            return self

        def next(self, bus: Bus):
            self._next(bus, K.Ema.fast, self.cfg.ema.fast)
            self._next(bus, K.Ema.slow, self.cfg.ema.slow)

        def _calc(self, price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

    class Smma(L):
        def first(self, bus: Bus):
            self._first(bus, K.Smma.fast)
            self._first(bus, K.Smma.slow)
            return self

        def next(self, bus: Bus):
            self._next(bus, K.Smma.fast, self.cfg.smma.fast)
            self._next(bus, K.Smma.slow, self.cfg.smma.slow)

        def _calc(self, price: float, period: int, prev_val: float):
            value = (prev_val * (period - 1) + price) / period
            return round(value, 4)

    class Macd(L):
        def first(self, bus: Bus):
            self._first(bus, K.Macd.fast)
            self._first(bus, K.Macd.slow)
            bus.update(-1, K.Macd.diff, 0)
            bus.update(-1, K.Macd.dea, 0)
            bus.update(-1, K.Macd.macd, 0)
            return self

        def next(self, bus: Bus):
            self._next(bus, K.Macd.fast, self.cfg.macd.fast)
            self._next(bus, K.Macd.slow, self.cfg.macd.slow)
            self._next_macd(bus)

        def _next_macd(self, bus: Bus):
            period = self.cfg.macd.sign
            fast_ema = bus.at(-1).get(K.Macd.fast)
            slow_ema = bus.at(-1).get(K.Macd.slow)
            prev_dea = bus.at(-2).get(K.Macd.dea)

            dif = round(fast_ema - slow_ema, 4)
            dea = self._calc(dif, period, prev_dea)
            macd = round((dif - dea) * 2, 4)

            bus.update(-1, K.Macd.diff, dif)
            bus.update(-1, K.Macd.dea, dea)
            bus.update(-1, K.Macd.macd, macd)

        def _calc(self, price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

    class Turn(L):
        def __init__(self, cfg: Var.Config):
            super().__init__(cfg)
            self.threshold = 10000

        @staticmethod
        def new_node(bus: Bus) -> Bar.Node:
            node = Bar.Node()
            node.node_idx = bus.key(-1)
            node.node_val = bus.at(-1).get(K.Ema.fast)
            return node

        def new_turn(self, node: Bar.Node) -> Bar.Turn:
            turn = Bar.Turn()
            turn.turn_idx = node.node_idx
            turn.turn_val = node.node_val
            turn.turn_lvl = node.node_lvl
            turn.add_quotas = copy.copy(self.cfg.rise.add_quotas)
            turn.sub_quotas = copy.copy(self.cfg.fall.sub_quotas)
            return turn

        def first(self, bus: Bus, ext: dict):
            node = self.new_node(bus)
            turn = self.new_turn(node)
            turn.turn_lvl = -2
            ext[K.Ext.turn_nodes] = [node]
            ext[K.Ext.turn_turns] = [turn]

            # 最小振幅价格差
            base_price = ext.get(K.Bas.base_price)
            self.threshold = round(base_price * self.cfg.turn.least_wave, 4)

            bar = bus.at(-1)
            bar[K.Turn.node_idx] = node.node_idx
            bar[K.Turn.node_val] = node.node_val
            bar[K.Turn.turn_idx] = turn.turn_idx
            bar[K.Turn.turn_val] = turn.turn_val
            bar[K.Turn.turn_lvl] = turn.turn_lvl
            return self

        def next(self, bus: Bus, ext: dict):
            # 先预设值
            bar = bus.at(-1)
            bar[K.Turn.node_idx] = bus.key(-1)
            bar[K.Turn.node_val] = bar.get(K.Ema.fast)
            bar[K.Turn.turn_idx] = bus.at(-2).get(K.Turn.turn_idx)
            bar[K.Turn.turn_val] = bus.at(-2).get(K.Turn.turn_val)
            bar[K.Turn.turn_lvl] = 0

            # 跳过连续相等的节点
            nodes = ext.get(K.Ext.turn_nodes)
            node = self.new_node(bus)
            if node.node_val == nodes[-1].node_val:
                return

            # 节点数据不得小于3条
            nodes.append(node)
            if len(nodes) < 3:
                return

            # 计算顶点、拐点
            turns = ext.get(K.Ext.turn_turns)
            self._calc_apex(nodes, turns)
            self._calc_turn(nodes, turns, bus)

        def _calc_apex(self, nodes: list[Bar.Node], turns: list[Bar.Turn]):
            # 计算顶点
            node3 = nodes[-3]
            node2 = nodes[-2]
            node1 = nodes[-1]
            turn1 = turns[-1]
            if node3.node_val < node2.node_val > node1.node_val:
                node2.node_lvl = 1
                self._max_apex(node2, turn1)
            elif node3.node_val > node2.node_val < node1.node_val:
                node2.node_lvl = -1
                self._min_apex(node2, turn1)

        def _calc_turn(self, nodes: list[Bar.Node], turns: list[Bar.Turn], bus: Bus):
            # 起始拐点
            first_turn = turns[0]
            if first_turn.turn_lvl == -2:
                diff = round(first_turn.turn_val - nodes[-1].node_val, 4)
                if abs(diff) > self.threshold:
                    first_turn.turn_lvl = 1 if diff > 0 else -1
                    bus.update(0, K.Turn.turn_lvl, first_turn.turn_lvl)
                return

            # 计算拐点
            turn = self._get_turn(nodes[-1], turns[-1])
            if turn is not None:
                turns.append(turn)
                bus.modify(turn.turn_idx, K.Turn.turn_lvl, turn.turn_lvl)
                bus.update(-1, K.Turn.turn_idx, turn.turn_idx)
                bus.update(-1, K.Turn.turn_val, turn.turn_val)

        def _max_apex(self, apex: Bar.Node, turn: Bar.Turn):
            # 到下一拐点前：最大的凸点
            if apex.node_val - turn.turn_val >= self.threshold:
                if turn.MaxApex is None or apex.node_val > turn.MaxApex.node_val:
                    turn.MaxApex = apex

        def _min_apex(self, apex: Bar.Node, turn: Bar.Turn):
            # 到下一拐点前：最小的凹点
            if turn.turn_val - apex.node_val >= self.threshold:
                if turn.MinApex is None or apex.node_val < turn.MinApex.node_val:
                    turn.MinApex = apex

        def _get_turn(self, node: Bar.Node, turn: Bar.Turn):
            if turn.MaxApex and turn.MaxApex.node_val - node.node_val >= self.threshold:
                return self.new_turn(turn.MaxApex)
            if turn.MinApex and node.node_val - turn.MinApex.node_val >= self.threshold:
                return self.new_turn(turn.MinApex)
            return None

        def _calc(self, price: float, period: int, prev_val: float):
            pass


############################################################
class Ticket:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.posBus = Bus()  # 仓位数据
        self.dayBus = Bus()  # 日线数据
        self.minBus = Bus()  # 分钟数据
        self.actSet = []  # 操作数据
        self.extMap = {}  # 扩展信息

    def clear(self):
        self.posBus.clear()
        self.dayBus.clear()
        self.minBus.clear()
        self.actSet.clear()
        self.extMap.clear()


class Market:
    def __init__(self, config: Type[Var.Config], trader: Type[Biz.Trader], ticket: Ticket):
        self.tkt: Ticket = ticket
        self.cfg: Var.Config = config()
        self.biz: Biz.Trader = trader(self.cfg, self.tkt)
        self.status = 0  # 状态：0-Initial、1-Started、2-Running

        self.line_ema = None
        self.line_smma = None
        self.line_macd = None
        self.line_turn = None

    def prepare(self, bars):
        self.tkt.clear()
        # 日线数据
        Bar(bars[0]).into(self.tkt.dayBus)
        self.line_smma = Line.Smma(self.cfg).first(self.tkt.dayBus)
        for bar in bars[1:]:
            Bar(bar).into(self.tkt.dayBus)
            self.line_smma.next(self.tkt.dayBus)

        # 基准价格
        self.tkt.extMap[K.Bas.base_price] = round(bars[-1].close, 4)
        self.status = 1
        return self

    def running(self, pos, bar):
        if self.status == 2:
            Pos(pos, bar).into(self.tkt.posBus)
            Bar(bar).into(self.tkt.minBus)
            self.line_ema.next(self.tkt.minBus)
            self.line_macd.next(self.tkt.minBus)
            self.line_turn.next(self.tkt.minBus, self.tkt.extMap)
            return

        if self.status == 0:
            self.prepare([bar])
            return

        if self.status == 1:
            # 仓位数据、基准持仓
            position = Pos(pos, bar)
            position.into(self.tkt.posBus)
            self.tkt.extMap[K.Bas.base_amount] = position.avail_amount
            # 分钟数据
            Bar(bar).into(self.tkt.minBus)
            self.line_ema = Line.Ema(self.cfg).first(self.tkt.minBus)
            self.line_macd = Line.Macd(self.cfg).first(self.tkt.minBus)
            self.line_turn = Line.Turn(self.cfg).first(self.tkt.minBus, self.tkt.extMap)
            self.status = 2

    def trading(self, buy: Callable, sell: Callable):
        self.biz.trading(buy, sell)


class Manager:
    tickets: dict[str, Ticket] = {}
    markets: dict[str, Market] = {}
    classes: dict[str, Type[Var.Config | Biz.Trader]] = {
        "config": Var.Config,
        "trader": Biz.TurnTrader,
    }
    # 黑名单、白名单
    blacks: list[str] = ['515450.SS', '515100.SS']
    whites: dict[str, Env] = {
        '159857.SZ': Env('config', 'trader'),
    }

    @staticmethod
    def market(symbol: str) -> Market:
        market = Manager.markets.get(symbol)
        if market is not None:
            return market

        biz = Manager.whites.get(symbol, Env('config', 'trader'))
        config = Manager.classes.get(biz.config)
        trader = Manager.classes.get(biz.trader)
        ticket = Manager.tickets.setdefault(symbol, Ticket(symbol))
        market = Manager.markets.setdefault(symbol, Market(config, trader, ticket))
        return market

    @staticmethod
    def get_symbols(positions: dict) -> list[str]:
        codes = list(Manager.whites)
        if positions:
            sids = [pos.sid for pos in positions.values()]
            codes.extend(sids)
        symbols = list(set(codes))
        for symbol in Manager.blacks:
            Manager.markets.pop(symbol, None)
            if symbol in symbols:
                symbols.remove(symbol)
        return symbols


############################################################
def initialize(context):
    """启动时执行一次"""
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    positions = get_positions()
    symbols = Manager.get_symbols(positions)
    g.symbols = symbols
    set_universe(symbols)

    history = get_history(60, frequency='1d')
    for symbol in symbols:
        df = history.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in df.iterrows()]
        Manager.market(symbol).prepare(bars)


def handle_data(context, data):
    """每个单位周期执行一次"""
    positions = context.portfolio.positions
    for symbol in g.symbols:
        bar = data[symbol]
        pos = positions.get(symbol)
        Manager.market(symbol).running(pos, bar)
        Manager.market(symbol).trading(order, order)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
