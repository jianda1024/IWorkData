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
from collections import deque
from types import SimpleNamespace
from typing import Callable, Type, Any


class K:
    class Bar:
        datetime = 'datetime'
        instant = 'instant'
        volume = 'volume'
        money = 'money'
        price = 'price'
        close = 'close'
        open = 'open'
        high = 'high'
        low = 'low'

    class Pos:
        avail_amount = 'avail_amount'
        total_amount = 'total_amount'
        cost_price = 'cost_price'
        last_price = 'last_price'
        valuation = 'valuation'
        principal = 'principal'

    class Ctx:
        stop_buying = 'stop_buying'
        base_amount = 'base_amount'
        base_price = 'base_price'
        turn_nodes = 'turn_nodes'
        turn_turns = 'turn_turns'
        live_act = 'live_act'

    class Sma:
        sma05 = 'SMA5'
        sma10 = 'SMA10'
        sma20 = 'SMA20'
        sma30 = 'SMA30'
        sma60 = 'SMA60'

    class Macd:
        fast = 'macd_fast'
        slow = 'macd_slow'
        sign = 'macd_sign'
        dif = 'macd_dif'
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
        self.node_val = bar[K.Sma.sma10]
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
        self.datetime = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.total_amount = getattr(pos, 'amount', 0.0)  # 总持仓数量
        self.avail_amount = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
        self.cost_price = getattr(pos, 'cost_basis', 0.0)  # 成本价格
        self.last_price = getattr(pos, 'last_sale_price', 0.0)  # 最新价格
        self.valuation = round(self.total_amount * self.last_price, 2)  # 市值
        self.principal = round(self.total_amount * self.cost_price, 2)  # 本金

    def into(self, lot: Lot):
        pos_dict = {
            K.Pos.avail_amount: self.avail_amount,
            K.Pos.total_amount: self.total_amount,
            K.Pos.cost_price: self.cost_price,
            K.Pos.last_price: self.last_price,
            K.Pos.valuation: self.valuation,
            K.Pos.principal: self.principal,
        }
        lot.add(self.datetime, pos_dict)


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
        self.datetime = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.instant = bar.datetime.strftime('%H:%M:%S')
        self.volume = round(bar.volume, 2)  # 交易量
        self.money = round(bar.money, 2)  # 交易金额
        self.price = round(bar.price, 4)  # 最新价
        self.close = round(bar.close, 4)  # 收盘价
        self.open = round(bar.open, 4)  # 开盘价
        self.high = round(bar.high, 4)  # 最高价
        self.low = round(bar.low, 4)  # 最低价

    def into(self, lot: Lot):
        bar_dict = {
            K.Bar.datetime: self.datetime,
            K.Bar.instant: self.instant,
            K.Bar.volume: self.volume,
            K.Bar.money: self.money,
            K.Bar.price: self.price,
            K.Bar.close: self.close,
            K.Bar.open: self.open,
            K.Bar.high: self.high,
            K.Bar.low: self.low,
        }
        lot.add(self.datetime, bar_dict)


class Lot:
    def __init__(self):
        self.keys: list[str] = []
        self.data: list[dict] = []

    def at(self, idx: int) -> dict:
        return self.data[idx]

    def add(self, key: str, value: dict):
        self.keys.append(key)
        self.data.append(value)

    def set(self, idx: int, col: str, val: Any):
        self.data[idx][col] = val

    def put(self, row: str, col: str, val: Any):
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
            self.begin_time = '09:40:00' #交易开始时间
            self.base_funds = 8000  # 基准资金
            self.cost_limit = 1.50  # 成本上限（比例）
            self.loss_limit = 0.15  # 亏损上限（比例）
            self.gain_limit = 0.05  # 盈利上限（比例）
            self.start_qty = 3000  # 起步买卖金额
            self.least_qty = 1000  # 最低卖出金额

    class Sma:
        def __init__(self):
            self.p05 = 5
            self.p10 = 10
            self.p20 = 20
            self.p30 = 30
            self.p60 = 60

    class Macd:
        def __init__(self):
            self.fast = 13  # 快线周期
            self.slow = 60  # 慢线周期
            self.sign = 5  # 信号线周期

    class Turn:
        def __init__(self):
            self.least_wave = 0.004  # 最小摆动（比例）
            self.value_key = K.Sma.sma10

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
            self.sma: Var.Sma = Var.Sma()
            self.macd: Var.Macd = Var.Macd()
            self.turn: Var.Turn = Var.Turn()
            self.rise: Var.Rise = Var.Rise()
            self.fall: Var.Fall = Var.Fall()


class Biz:
    class Base:
        def __init__(self, cfg: Var.Config, bus: Bus):
            self.way = 'PTrader'
            self.cfg = cfg
            self.bus = bus

        def live_pos(self) -> dict:
            return self.bus.posLot.at(-1)

        def live_day(self) -> dict:
            return self.bus.dayLot.at(-1)

        def live_min(self) -> dict:
            return self.bus.minLot.at(-1)

        def fit_sell_qty(self, plan_qty: int):
            """适配卖出的数量"""
            pos = self.live_pos()
            base_price = self.bus.ctxMap.get(K.Ctx.base_price)
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
            # 配额已经用完
            acts = [act for act in self.bus.actSet if act.status == status]
            length = len(acts)
            if len(acts) >= len(quotas):
                return False, 0

            # 当前信息
            live_min = self.live_min()
            turn_idx = live_min[K.Turn.turn_idx]  # 拐点
            node_val = live_min[K.Turn.node_val]  # 节点值
            base_price = self.bus.ctxMap.get(K.Ctx.base_price)  # 基准价格
            threshold_val = thresholds[length] * base_price  # 阈值

            # 首次交易，或者前后两次交易的拐点不一样
            if length == 0 or turn_idx != acts[-1].turn_idx:
                if (node_val - live_min[K.Turn.turn_val]) * status > threshold_val:
                    return True, quotas[length]
                return False, 0

            # 前后两次交易的拐点相同
            if (node_val - acts[-1].node_val) * status > threshold_val:
                return True, quotas[length]
            return False, 0

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
                self.__do_buy(buy, self.bus.symbol, amount)
            if self.is_sell():
                amount = self.sell_amount()
                self.__do_sell(sell, self.bus.symbol, amount)

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
            self.bus.actSet.append(live_act)
            self.bus.ctxMap[K.Ctx.live_act] = live_act

    class TurnTrader(Trader):
        def is_buy(self) -> bool:
            """判断是否买入"""
            live_min = self.live_min()
            if self.is_over_budget():  return False
            if self.bus.ctxMap.get(K.Ctx.stop_buying, True): return False
            if live_min[K.Bar.instant] < self.cfg.bas.begin_time: return False

            # MACD大于设定的上限时，直接买入
            if live_min[K.Macd.macd] >= self.cfg.rise.upper_macd: return True
            # 分钟线
            if live_min[K.Sma.sma10] <= max(live_min[K.Sma.sma20], live_min[K.Sma.sma30]): return False
            if live_min[K.Macd.dif] < 0 or live_min[K.Macd.dea] < 0: return False
            if live_min[K.Macd.macd] < 0: return False

            # 决定买入
            return True

        def buy_amount(self) -> float:
            """买入数量"""
            has_quota, quota = self.check_buy_quota()
            if not has_quota: return 0
            base_price = self.bus.ctxMap.get(K.Ctx.base_price)
            buy_amount = max(self.cfg.bas.base_funds * quota, self.cfg.bas.start_qty) / base_price
            return round(buy_amount / 100) * 100

        def is_sell(self) -> bool:
            """判断是否卖出"""
            live_min = self.live_min()
            if live_min[K.Bar.instant] < self.cfg.bas.begin_time:
                return False

            # MACD小于设定的下限，直接卖出
            if live_min[K.Macd.macd] < self.cfg.fall.lower_macd: return True
            # 分钟线
            if live_min[K.Sma.sma10] >= live_min[K.Sma.sma20]: return False
            if live_min[K.Macd.macd] > 0: return False
            # 决定卖出
            return True

        def sell_amount(self) -> float:
            """卖出数量"""
            has_quota, quota = self.check_sell_quota()
            if not has_quota: return 0
            plan_amount = self.cfg.bas.base_funds * quota
            sell_amount = self.fit_sell_qty(plan_amount)
            return -sell_amount


class Line:
    class Sma:
        def __init__(self, cfg: Var.Config):
            self.cfg: Var.Config = cfg

        def first(self, lot: Lot, ext: dict):
            Line.Sma.__first(lot, ext, K.Sma.sma05, self.cfg.sma.p05)
            Line.Sma.__first(lot, ext, K.Sma.sma10, self.cfg.sma.p10)
            Line.Sma.__first(lot, ext, K.Sma.sma20, self.cfg.sma.p20)
            Line.Sma.__first(lot, ext, K.Sma.sma30, self.cfg.sma.p30)
            Line.Sma.__first(lot, ext, K.Sma.sma60, self.cfg.sma.p60)
            return self

        def next(self, lot: Lot, ext: dict):
            Line.Sma.__next(lot, ext, K.Sma.sma05)
            Line.Sma.__next(lot, ext, K.Sma.sma10)
            Line.Sma.__next(lot, ext, K.Sma.sma20)
            Line.Sma.__next(lot, ext, K.Sma.sma30)
            Line.Sma.__next(lot, ext, K.Sma.sma60)
            return self

        @staticmethod
        def __first(lot: Lot, ext: dict, key: str, period: int):
            price = lot.at(-1).get(K.Bar.close)
            prices = deque(maxlen=period)
            prices.append(price)
            ext[key] = prices
            lot.set(-1, key, price)

        @staticmethod
        def __next(lot: Lot, ext: dict, key: str):
            price = lot.at(-1).get(K.Bar.close)
            prices = ext.get(key)
            prices.append(price)
            sma = round(sum(prices) / len(prices), 4)
            lot.set(-1, key, sma)

    class Macd:
        def __init__(self, cfg: Var.Config):
            self.cfg: Var.Config = cfg

        def first(self, lot: Lot):
            price = lot.at(-1).get(K.Bar.close)
            lot.set(-1, K.Macd.fast, price)
            lot.set(-1, K.Macd.slow, price)
            lot.set(-1, K.Macd.dif, 0)
            lot.set(-1, K.Macd.dea, 0)
            lot.set(-1, K.Macd.macd, 0)
            return self

        def next(self, lot: Lot):
            self.__next(lot, K.Macd.fast, self.cfg.macd.fast)
            self.__next(lot, K.Macd.slow, self.cfg.macd.slow)
            self.__next_macd(lot, self.cfg.macd.sign)

        @staticmethod
        def __next(lot: Lot, key: str, period: int):
            price = lot.at(-1).get(K.Bar.close)
            prev_val = lot.at(-2).get(key)
            next_val = Line.Macd.__calc(price, period, prev_val)
            lot.set(-1, key, next_val)

        @staticmethod
        def __next_macd(lot: Lot, period: int):
            fast_ema = lot.at(-1).get(K.Macd.fast)
            slow_ema = lot.at(-1).get(K.Macd.slow)
            prev_dea = lot.at(-2).get(K.Macd.dea)

            dif = round(fast_ema - slow_ema, 4)
            dea = Line.Macd.__calc(dif, period, prev_dea)
            macd = round((dif - dea) * 2, 4)

            lot.set(-1, K.Macd.dif, dif)
            lot.set(-1, K.Macd.dea, dea)
            lot.set(-1, K.Macd.macd, macd)

        @staticmethod
        def __calc(price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

    class Turn:
        def __init__(self, cfg: Var.Config):
            self.cfg: Var.Config = cfg
            self.threshold = 10000
            self.value_key = cfg.turn.value_key

        def new_node(self, lot: Lot) -> Bar.Node:
            node = Bar.Node()
            node.node_idx = lot.key(-1)
            node.node_val = lot.at(-1).get(self.value_key)
            return node

        def new_turn(self, node: Bar.Node) -> Bar.Turn:
            turn = Bar.Turn()
            turn.turn_idx = node.node_idx
            turn.turn_val = node.node_val
            turn.turn_lvl = node.node_lvl
            turn.add_quotas = copy.copy(self.cfg.rise.add_quotas)
            turn.sub_quotas = copy.copy(self.cfg.fall.sub_quotas)
            return turn

        def first(self, lot: Lot, ext: dict):
            node = self.new_node(lot)
            turn = self.new_turn(node)
            turn.turn_lvl = -2
            ext[K.Ctx.turn_nodes] = [node]
            ext[K.Ctx.turn_turns] = [turn]

            # 最小振幅价格差
            base_price = ext.get(K.Ctx.base_price)
            self.threshold = round(base_price * self.cfg.turn.least_wave, 4)

            bar = lot.at(-1)
            bar[K.Turn.node_idx] = node.node_idx
            bar[K.Turn.node_val] = node.node_val
            bar[K.Turn.turn_idx] = turn.turn_idx
            bar[K.Turn.turn_val] = turn.turn_val
            bar[K.Turn.turn_lvl] = turn.turn_lvl
            return self

        def next(self, lot: Lot, ext: dict):
            # 先预设值
            bar = lot.at(-1)
            bar[K.Turn.node_idx] = lot.key(-1)
            bar[K.Turn.node_val] = bar.get(self.value_key)
            bar[K.Turn.turn_idx] = lot.at(-2).get(K.Turn.turn_idx)
            bar[K.Turn.turn_val] = lot.at(-2).get(K.Turn.turn_val)
            bar[K.Turn.turn_lvl] = 0

            # 跳过连续相等的节点
            nodes = ext.get(K.Ctx.turn_nodes)
            node = self.new_node(lot)
            if node.node_val == nodes[-1].node_val:
                return

            # 节点数据不得小于3条
            nodes.append(node)
            if len(nodes) < 3:
                return

            # 计算顶点、拐点
            turns = ext.get(K.Ctx.turn_turns)
            self._calc_apex(nodes, turns)
            self._calc_turn(nodes, turns, lot)

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

        def _calc_turn(self, nodes: list[Bar.Node], turns: list[Bar.Turn], lot: Lot):
            # 起始拐点
            first_turn = turns[0]
            if first_turn.turn_lvl == -2:
                diff = round(first_turn.turn_val - nodes[-1].node_val, 4)
                if abs(diff) > self.threshold:
                    first_turn.turn_lvl = 1 if diff > 0 else -1
                    lot.set(0, K.Turn.turn_lvl, first_turn.turn_lvl)
                return

            # 计算拐点
            turn = self._get_turn(nodes[-1], turns[-1])
            if turn is not None:
                turns.append(turn)
                lot.put(turn.turn_idx, K.Turn.turn_lvl, turn.turn_lvl)
                lot.set(-1, K.Turn.turn_idx, turn.turn_idx)
                lot.set(-1, K.Turn.turn_val, turn.turn_val)

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
class Bus:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.posLot = Lot()  # 仓位数据
        self.dayLot = Lot()  # 日线数据
        self.minLot = Lot()  # 分钟数据
        self.ctxMap = {}  # 上下文数据
        self.actSet = []  # 操作数据

    def clear(self):
        self.posLot.clear()
        self.dayLot.clear()
        self.minLot.clear()
        self.ctxMap.clear()
        self.actSet.clear()


class Market:
    def __init__(self, config: Type[Var.Config], trader: Type[Biz.Trader], bus: Bus):
        self.bus: Bus = bus
        self.cfg: Var.Config = config()
        self.biz: Biz.Trader = trader(self.cfg, self.bus)
        self.status = 0  # 状态：-1暂停、0初始、1就绪、2执行中

        self.line_day_sma = None
        self.line_day_macd = None
        self.line_sma = None
        self.line_macd = None
        self.line_turn = None

    def prepare(self, pos, bars):
        self.bus.clear()
        # 日线数据
        Bar(bars[0]).into(self.bus.dayLot)
        self.line_day_sma = Line.Sma(self.cfg).first(self.bus.dayLot, self.bus.ctxMap)
        self.line_day_macd = Line.Macd(self.cfg).first(self.bus.dayLot)
        for bar in bars[1:]:
            Bar(bar).into(self.bus.dayLot)
            self.line_day_sma.next(self.bus.dayLot, self.bus.ctxMap)
            self.line_day_macd.next(self.bus.dayLot)

        # 今日是否允许买卖
        stop_buying = self.__stop_buying()
        self.bus.ctxMap[K.Ctx.stop_buying] = stop_buying
        amount = getattr(pos, 'amount', 0.0)  # 总持仓数量
        if stop_buying and amount <= 100:
            self.status = -1
            return self

        # 今日基准价格
        self.bus.ctxMap[K.Ctx.base_price] = round(bars[-1].close, 4)
        self.status = 1
        return self

    def running(self, pos, bar):
        if self.status == 2:
            Pos(pos, bar).into(self.bus.posLot)
            Bar(bar).into(self.bus.minLot)
            self.line_sma.next(self.bus.minLot, self.bus.ctxMap)
            self.line_turn.next(self.bus.minLot, self.bus.ctxMap)
            self.line_macd.next(self.bus.minLot)
            return

        if self.status == -1:
            return

        if self.status == 0:
            self.prepare(pos,[bar])
            return

        if self.status == 1:
            # 仓位数据、基准持仓
            position = Pos(pos, bar)
            position.into(self.bus.posLot)
            self.bus.ctxMap[K.Ctx.base_amount] = position.avail_amount
            # 分钟数据
            Bar(bar).into(self.bus.minLot)
            self.line_sma = Line.Sma(self.cfg).first(self.bus.minLot, self.bus.ctxMap)
            self.line_turn = Line.Turn(self.cfg).first(self.bus.minLot, self.bus.ctxMap)
            self.line_macd = Line.Macd(self.cfg).first(self.bus.minLot)
            self.status = 2

    def trading(self, buy: Callable, sell: Callable):
        self.biz.trading(buy, sell)

    def __stop_buying(self) -> bool:
        live_day = self.bus.dayLot.at(-1)
        max_sma = max(live_day[K.Sma.sma10], live_day[K.Sma.sma20], live_day[K.Sma.sma30], live_day[K.Sma.sma60])
        if live_day[K.Sma.sma05] <= max_sma: return True
        if live_day[K.Macd.dif] < 0: return True
        if live_day[K.Macd.dea] < 0: return True
        return False


class Manager:
    busList: dict[str, Bus] = {}
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

        bus = Manager.busList.setdefault(symbol, Bus(symbol))
        biz = Manager.whites.get(symbol, Env('config', 'trader'))
        config = Manager.classes.get(biz.config)
        trader = Manager.classes.get(biz.trader)
        market = Manager.markets.setdefault(symbol, Market(config, trader, bus))
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

    history = get_history(120, frequency='1d')
    for symbol in symbols:
        pos = positions.get(symbol)
        df = history.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in df.iterrows()]
        Manager.market(symbol).prepare(pos,bars)


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
