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
from collections import deque
from types import SimpleNamespace
from typing import Callable, Type


class K:
    is_stop_buy = 'is_stop_buy'
    base_amount = 'base_amount'
    base_price = 'base_price'
    sup_quotas = 'sup_quotas'
    sub_quotas = 'sub_quotas'
    last_act = 'last_act'


class Var:
    class Base:
        def __init__(self):
            self.begin_time = '09:40:00'  # 开始交易时间
            self.basic_fund = 10000  # 交易基准金额
            self.start_fund = 3000  # 交易起步金额
            self.least_fund = 1000  # 交易最低金额
            self.cost_limit = 1.50  # 成本上限（比例）
            self.loss_limit = 0.15  # 亏损上限（比例）
            self.gain_limit = 0.05  # 盈利上限（比例）

    class Macd:
        def __init__(self):
            self.fast = 13  # 快线周期
            self.slow = 60  # 慢线周期
            self.sign = 5  # 信号线周期

    class Turn:
        def __init__(self):
            self.least_wave = 0.005  # 最小摆动（比例）
            self.sma_period = 10  # 所参照sma的周期

    class Trad:
        def __init__(self):
            self.sup_quotas = [0.300, 0.300, 0.300]  # 加仓额度（比例）
            self.sup_bounds = [0.006, 0.010]  # 加仓阈值（比例）
            self.upper_macd = 0.0025  # macd限制（比例）

            self.sub_quotas = [0.300, 0.400, 0.300]  # 减仓额度（比例）
            self.sub_bounds = [0.010, 0.015, 0.020]  # 减仓阈值（比例）
            self.lower_macd = -0.0025  # macd限制（比例）

    class Config:
        def __init__(self):
            self.env = 'Live'  # 测试--Test、线上--Live
            self.base: Var.Base = Var.Base()
            self.macd: Var.Macd = Var.Macd()
            self.turn: Var.Turn = Var.Turn()
            self.trad: Var.Trad = Var.Trad()


class Act:
    def __init__(self, symbol: str, node: Bin.Node):
        self.node_idx = node.index
        self.node_val = node.turn.val
        self.turn_idx = node.turn.prvTurn.idx
        self.turn_val = node.turn.prvTurn.val
        self.turn_lvl = node.turn.prvTurn.lvl
        self.symbol = symbol
        self.amount = 0
        self.status = 0

    def trade(self, amount: float):
        self.status = 1 if amount > 0 else -1 if amount < 0 else 0
        self.amount = amount
        return self


class Bin:
    class Bar:
        def __init__(self, bar):
            self.datetime = bar.datetime
            self.instant = bar.datetime.strftime('%H:%M:%S')
            self.volume: float = round(bar.volume, 2)  # 交易量
            self.money: float = round(bar.money, 2)  # 交易金额
            self.price: float = round(bar.price, 4)  # 最新价
            self.close: float = round(bar.close, 4)  # 收盘价
            self.open: float = round(bar.open, 4)  # 开盘价
            self.high: float = round(bar.high, 4)  # 最高价
            self.low: float = round(bar.low, 4)  # 最低价

    class Sma:
        _attrs = {5: 'sma05', 10: 'sma10', 20: 'sma20', 30: 'sma30', 60: 'sma60'}

        def __init__(self):
            self.sma05: float = 0.0
            self.sma10: float = 0.0
            self.sma20: float = 0.0
            self.sma30: float = 0.0
            self.sma60: float = 0.0

        def get(self, period: int) -> float:
            return getattr(self, Bin.Sma._attrs.get(period, ''), 0.0)

        def set(self, period: int, val: float):
            attr = Bin.Sma._attrs.get(period)
            if attr is not None:
                setattr(self, attr, val)

    class Macd:
        def __init__(self):
            self.fast: float = 0.0
            self.slow: float = 0.0
            self.sign: float = 0.0
            self.diff: float = 0.0
            self.dea_: float = 0.0
            self.macd: float = 0.0

    class Turn:
        class Dot:
            def __init__(self, idx: str, val: float, lvl: int):
                self.lvl: int = lvl
                self.idx: str = idx
                self.val: float = val

        def __init__(self):
            self.lvl: int = 0
            self.val: float = 0.0
            self.prev_idx: str = ''
            self.prvTurn: Bin.Turn.Dot | None = None
            self.maxApex: Bin.Turn.Dot | None = None
            self.minApex: Bin.Turn.Dot | None = None

    class Node:
        def __init__(self, bar):
            self.index: str = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
            self.bar: Bin.Bar = Bin.Bar(bar)
            self.sma: Bin.Sma = Bin.Sma()
            self.macd: Bin.Macd = Bin.Macd()
            self.turn: Bin.Turn = Bin.Turn()


class Pos:
    def __init__(self, pos):
        self.total_amount = getattr(pos, 'amount', 0.0)  # 总持仓数量
        self.avail_amount = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
        self.last_price = getattr(pos, 'last_sale_price', 0.0)  # 最新价格
        self.cost_price = getattr(pos, 'cost_basis', 0.0)  # 成本价格
        self.valuation = round(self.total_amount * self.last_price, 2)  # 市值
        self.principal = round(self.total_amount * self.cost_price, 2)  # 本金


class Silo:
    def __init__(self):
        self.keys: list[str] = []
        self.data: list[Bin.Node] = []
        self.dict: dict = {}

    def at(self, idx: int) -> Bin.Node:
        return self.data[idx]

    def get(self, key: str) -> Bin.Node:
        idx = self.keys.index(key)
        return self.data[idx]

    def add(self, node: Bin.Node):
        self.keys.append(node.index)
        self.data.append(node)

    def clear(self):
        self.keys.clear()
        self.data.clear()
        self.dict.clear()


############################################################
class Biz:
    class Bas:
        def __init__(self, cfg: Var.Config, bus: Bus):
            self.cfg = cfg
            self.bus = bus

        def last_pos(self) -> Pos:
            return self.bus.posArr[-1]

        def last_day(self) -> Bin.Node:
            return self.bus.daySet.at(-1)

        def last_min(self) -> Bin.Node:
            return self.bus.minSet.at(-1)

        def is_out_budget(self) -> bool:
            """是否超过本金上限/亏损上限"""
            pos = self.last_pos()
            if pos.principal > self.cfg.base.basic_fund * self.cfg.base.cost_limit:
                return True
            if pos.principal - pos.valuation >= self.cfg.base.basic_fund * self.cfg.base.loss_limit:
                return True
            return False

        def is_hit_bound(self, status) -> bool:
            """是否达到阈值边界"""
            bounds = self.cfg.trad.sup_bounds if status > 0 else self.cfg.trad.sub_bounds
            last_min = self.last_min()
            turn_idx = last_min.turn.prev_idx

            # 交易记录：类型相同、拐点相同
            acts = [act for act in self.bus.actArr if act.status == status and act.turn_idx == turn_idx]
            if len(acts) < len(bounds):
                threshold = bounds[len(acts)] * self.bus.ctxMap.get(K.base_price)
                if (last_min.turn.val - last_min.turn.prvTurn.val) * status > threshold:
                    return True
            return False

    class Qty(Bas):
        def buy_amount(self) -> float:
            """买入数量"""
            quotas = self.bus.ctxMap.get(K.sup_quotas)
            if quotas:
                quota = quotas.pop(0)
                base_price = self.bus.ctxMap.get(K.base_price)
                buy_amount = max(self.cfg.base.basic_fund * quota, self.cfg.base.start_fund) / base_price
                return round(buy_amount / 100) * 100
            return 0

        def sell_amount(self) -> float:
            """卖出数量"""
            quotas = self.bus.ctxMap.get(K.sub_quotas)
            if quotas:
                quota = quotas.pop(0)
                plan_amount = self.cfg.base.basic_fund * quota
                sell_amount = self.fit_sell_qty(plan_amount)
                return -sell_amount
            return 0

        def fit_sell_qty(self, plan_qty: int):
            """适配卖出的数量"""
            pos = self.last_pos()
            base_price = self.bus.ctxMap.get(K.base_price)
            start_qty = round(self.cfg.base.start_fund / base_price / 100) * 100
            least_qty = round(self.cfg.base.least_fund / base_price / 100) * 100
            avail_qty = pos.avail_amount

            # 计算需要保留的数量
            profit = pos.valuation - pos.principal  # 当前盈利
            unavail_qty = pos.total_amount - avail_qty  # 不可用持仓数量
            profit_target = self.cfg.base.basic_fund * self.cfg.base.gain_limit  # 盈利目标
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

    class Trader(Qty):
        def is_buy(self) -> bool:
            """判断是否买入"""
            last_min = self.last_min()
            if self.is_out_budget():  return False
            if self.bus.ctxMap.get(K.is_stop_buy, True): return False
            if last_min.bar.instant < self.cfg.base.begin_time: return False

            # MACD大于设定的上限时，直接买入
            if last_min.macd.macd >= self.cfg.trad.upper_macd:
                return True

            if last_min.sma.sma10 <= max(last_min.sma.sma20, last_min.sma.sma30): return False
            if last_min.macd.diff < 0 or last_min.macd.dea_ < 0: return False
            if last_min.macd.macd < 0: return False

            # 是否达到阈值边界
            return self.is_hit_bound(1)

        def is_sell(self) -> bool:
            """判断是否卖出"""
            last_min = self.last_min()
            if last_min.bar.instant < self.cfg.base.begin_time: return False

            # MACD小于设定的下限，直接卖出
            if last_min.macd.macd < self.cfg.trad.lower_macd:
                return True

            if last_min.sma.sma10 >= last_min.sma.sma20: return False
            if last_min.macd.macd > 0: return False

            # 是否达到阈值边界
            return self.is_hit_bound(-1)

        def trading(self, buy: Callable, sell: Callable):
            """执行交易"""
            symbol = self.bus.symbol
            is_live = self.cfg.env == 'Live'
            is_test = self.cfg.env == 'Test'

            # 买入
            if self.is_buy():
                amount = self.buy_amount()
                if is_live and amount > 0:
                    buy(symbol, amount)
                    self.log(symbol, amount)
                if is_test and amount > 0:
                    buy(size=amount)
                    self.log(symbol, amount)

            # 卖出
            if self.is_sell():
                amount = self.sell_amount()
                if is_live and amount < 0:
                    sell(symbol, amount)
                    self.log(symbol, amount)
                if is_test and amount < 0:
                    sell(size=-amount)
                    self.log(symbol, amount)

        def log(self, symbol: str, amount: float):
            last_min = self.last_min()
            last_act = Act(symbol, last_min).trade(amount)
            self.bus.actArr.append(last_act)
            self.bus.ctxMap[K.last_act] = last_act


class Line:
    class Sma:
        @staticmethod
        def first(silo: Silo):
            Line.Sma.__first(silo, 5)
            Line.Sma.__first(silo, 10)
            Line.Sma.__first(silo, 20)
            Line.Sma.__first(silo, 30)
            Line.Sma.__first(silo, 60)

        @staticmethod
        def next(silo: Silo):
            Line.Sma.__next(silo, 5)
            Line.Sma.__next(silo, 10)
            Line.Sma.__next(silo, 20)
            Line.Sma.__next(silo, 30)
            Line.Sma.__next(silo, 60)
            pass

        @staticmethod
        def __first(silo: Silo, period: int):
            key = 'SMA' + str(period)
            dqe = deque(maxlen=period)
            node = silo.at(-1)
            price = node.bar.close
            dqe.append(price)
            node.sma.set(period, price)
            silo.dict.setdefault(key, dqe)

        @staticmethod
        def __next(silo: Silo, period: int):
            key = 'SMA' + str(period)
            dqe = silo.dict.get(key)
            node = silo.at(-1)
            price = node.bar.close
            dqe.append(price)
            value = round(sum(dqe) / len(dqe), 4)
            node.sma.set(period, value)

    class Macd:
        @staticmethod
        def first(silo: Silo):
            node = silo.at(-1)
            price = node.bar.close
            node.macd.fast = price
            node.macd.slow = price
            node.macd.diff = 0.0
            node.macd.dea_ = 0.0
            node.macd.macd = 0.0

        @staticmethod
        def next(silo: Silo, cfg: Var.Config):
            node = silo.at(-1)
            price = node.bar.close
            prev_node = silo.at(-2)

            fast = Line.Macd.ema(price, cfg.macd.fast, prev_node.macd.fast)
            slow = Line.Macd.ema(price, cfg.macd.slow, prev_node.macd.slow)
            diff = round(fast - slow, 4)
            dea_ = Line.Macd.ema(diff, cfg.macd.sign, prev_node.macd.dea_)
            macd = round((diff - dea_) * 2, 4)

            node.macd.fast = fast
            node.macd.slow = slow
            node.macd.diff = diff
            node.macd.dea_ = dea_
            node.macd.macd = macd

        @staticmethod
        def ema(price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

    class Turn:
        @staticmethod
        def first(silo: Silo, cfg: Var.Config, ctx: dict):
            # 节点集合
            node = silo.at(-1)
            dot = Bin.Turn.Dot(node.index, node.bar.close, -2)
            dots = deque(maxlen=3)
            dots.append(dot)
            silo.dict.setdefault('dots', dots)

            # 拐点信息
            node.turn.lvl = -2
            node.turn.val = node.bar.close
            node.turn.prev_idx = node.index
            node.turn.prvTurn = copy.copy(dot)

            # 价格最小振幅阈值
            base_price = ctx.get(K.base_price)
            threshold = round(base_price * cfg.turn.least_wave, 4)
            silo.dict.setdefault('threshold', threshold)

        @staticmethod
        def next(silo: Silo, cfg: Var.Config):
            # 预先设值
            node = silo.at(-1)
            curr_turn = node.turn
            prev_turn = silo.at(-2).turn
            curr_turn.lvl = 0
            curr_turn.val = node.sma.get(cfg.turn.sma_period)
            curr_turn.prev_idx = prev_turn.prev_idx
            curr_turn.prvTurn = prev_turn.prvTurn
            curr_turn.maxApex = prev_turn.maxApex
            curr_turn.minApex = prev_turn.minApex

            # 跳过连续相等的点
            dots = silo.dict.get('dots')
            if curr_turn.val == dots[-1].val:
                return

            # 数据不得小于3条
            dot = Bin.Turn.Dot(node.index, curr_turn.val, 0)
            dots.append(dot)
            if len(dots) < 3:
                return

            # 计算顶点、拐点
            Line.Turn.__apex(silo)
            Line.Turn.__turn(silo)

        @staticmethod
        def __apex(silo: Silo):
            turn = silo.at(-1).turn
            dots = silo.dict.get('dots')
            threshold = silo.dict.get('threshold')

            # 计算顶点
            dot3 = dots[-3]
            dot2 = dots[-2]
            dot1 = dots[-1]
            if dot3.val < dot2.val > dot1.val:
                if dot2.val - turn.prvTurn.val >= threshold:
                    if turn.maxApex is None or dot2.val > turn.maxApex.val:
                        dot2.lvl = 1
                        turn.maxApex = dot2
            elif dot3.val > dot2.val < dot1.val:
                if turn.prvTurn.val - dot2.val >= threshold:
                    if turn.minApex is None or dot2.val < turn.minApex.val:
                        dot2.lvl = -1
                        turn.minApex = dot2

        @staticmethod
        def __turn(silo: Silo):
            # 起始拐点
            turn = silo.at(-1).turn
            dots = silo.dict.get('dots')
            threshold = silo.dict.get('threshold')
            if turn.prvTurn.lvl == -2:
                diff = turn.prvTurn.val - dots[-1].val
                if abs(diff) > threshold:
                    turn.prvTurn.lvl = 1 if diff > 0 else -1
                    silo.at(0).turn.lvl = turn.prvTurn.lvl
                return

            # 计算拐点
            dot = None
            value = dots[-1].val
            if turn.maxApex and turn.maxApex.val - value >= threshold:
                dot = turn.maxApex
            if turn.minApex and value - turn.minApex.val >= threshold:
                dot = turn.minApex
            if dot is not None:
                silo.get(dot.idx).turn.lvl = dot.lvl
                turn.prev_idx = dot.idx
                turn.prvTurn = dot
                turn.maxApex = None
                turn.minApex = None


class Market:
    def __init__(self, config: Var.Config, trader: Biz.Trader, bus: Bus):
        self.status = 0  # 状态：-1暂停、0初始、1就绪、2执行中
        self.cfg: Var.Config = config
        self.biz: Biz.Trader = trader
        self.bus: Bus = bus

    def prepare(self, pos, bars):
        self.bus.clear()
        self.pre_line(self.bus.daySet, bars)
        self.bus.ctxMap[K.base_price] = round(bars[-1].close, 4)
        self.bus.ctxMap[K.sup_quotas] = self.cfg.trad.sup_quotas.copy()
        self.bus.ctxMap[K.sub_quotas] = self.cfg.trad.sub_quotas.copy()
        self.status = 1

        # 今日是否允许买卖
        is_stop_buy = self.is_stop_buy()
        self.bus.ctxMap[K.is_stop_buy] = is_stop_buy
        amount = getattr(pos, 'amount', 0.0)  # 总持仓数量
        if is_stop_buy and amount <= 100:
            self.status = -1

    def running(self, pos, bar):
        if self.status == 2:
            position = Pos(pos)
            self.bus.posArr.append(position)
            self.bus.minSet.add(Bin.Node(bar))
            Line.Sma.next(self.bus.minSet)
            Line.Macd.next(self.bus.minSet, self.cfg)
            Line.Turn.next(self.bus.minSet, self.cfg)
            return

        if self.status == 1:
            position = Pos(pos)
            self.bus.posArr.append(position)
            self.bus.minSet.add(Bin.Node(bar))
            Line.Sma.first(self.bus.minSet)
            Line.Macd.first(self.bus.minSet)
            Line.Turn.first(self.bus.minSet, self.cfg, self.bus.ctxMap)
            self.bus.ctxMap[K.base_amount] = position.avail_amount
            self.status = 2
        elif self.status == 0:
            self.prepare(pos, [bar])
        elif self.status == -1:
            return

    def trading(self, buy: Callable, sell: Callable):
        if self.status == 2:
            self.biz.trading(buy, sell)

    def is_stop_buy(self) -> bool:
        last_day = self.bus.daySet.at(-1)
        if last_day.sma.sma05 <= last_day.sma.sma10: return True
        if last_day.sma.sma05 <= last_day.sma.sma20: return True
        if last_day.sma.sma05 <= last_day.sma.sma30: return True
        if last_day.sma.sma05 <= last_day.sma.sma60: return True
        if last_day.macd.diff < 0: return True
        if last_day.macd.dea_ < 0: return True
        if last_day.macd.macd < 0: return True
        return False

    def pre_line(self, silo: Silo, bars):
        if not bars: return
        silo.add(Bin.Node(bars[0]))
        Line.Sma.first(silo)
        Line.Macd.first(silo)
        for bar in bars[1:]:
            silo.add(Bin.Node(bar))
            Line.Sma.next(silo)
            Line.Macd.next(silo, self.cfg)


############################################################
class Kit:
    def __init__(self, config: str, trader: str):
        self.config = config  # 配置信息
        self.trader = trader  # 交易规则


class Bus:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.daySet = Silo()  # 日线数据
        self.minSet = Silo()  # 分钟数据
        self.posArr = []  # 仓位数据
        self.actArr = []  # 操作数据
        self.ctxMap = {}  # 上下文数据

    def clear(self):
        self.daySet.clear()
        self.minSet.clear()
        self.posArr.clear()
        self.actArr.clear()
        self.ctxMap.clear()


class Env:
    markets: dict[str, Market] = {}
    classes: dict[str, Type[Var.Config | Biz.Trader]] = {
        "config": Var.Config,
        "trader": Biz.Trader,
    }
    # 黑名单、白名单
    blacks: list[str] = ['515450.SS', '515100.SS']
    whites: dict[str, Kit] = {
        '159857.SZ': Kit('config', 'trader'),
    }

    @staticmethod
    def symbols(positions: dict) -> list[str]:
        codes = list(Env.whites)
        if positions:
            sids = [pos.sid for pos in positions.values()]
            codes.extend(sids)
        symbols = list(set(codes))
        for symbol in Env.blacks:
            Env.markets.pop(symbol, None)
            if symbol in symbols:
                symbols.remove(symbol)
        return symbols

    @staticmethod
    def market(symbol: str) -> Market:
        market = Env.markets.get(symbol)
        if market is not None:
            return market

        bus = Bus(symbol)
        kit = Env.whites.get(symbol, Kit('config', 'trader'))
        config = Env.classes.get(kit.config)()
        trader = Env.classes.get(kit.trader)(config, bus)
        market = Env.markets.setdefault(symbol, Market(config, trader, bus))
        return market


############################################################
def initialize(context):
    """启动时执行一次"""
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    positions = get_positions()
    symbols = Env.symbols(positions)
    g.symbols = symbols
    set_universe(symbols)

    history = get_history(120, frequency='1d')
    for symbol in symbols:
        pos = positions.get(symbol)
        days = history.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in days.iterrows()]
        Env.market(symbol).prepare(pos, bars)


def handle_data(context, data):
    """每个单位周期执行一次"""
    positions = context.portfolio.positions
    for symbol in g.symbols:
        bar = data[symbol]
        pos = positions.get(symbol)
        Env.market(symbol).running(pos, bar)
        Env.market(symbol).trading(order, order)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
