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

from collections import deque
from types import SimpleNamespace
from typing import Callable, Type


class K:
    base_price = 'base_price'
    base_amount = 'base_amount'
    rise_quotas = 'rise_quotas'
    fall_quotas = 'fall_quotas'
    last_log = 'last_log'


class Bar:
    def __init__(self, bar):
        self.datetime = bar.datetime
        self.instant = bar.datetime.strftime('%H:%M:%S')
        self.volume: float = round(bar.volume, 2)  # 交易量
        self.money: float = round(bar.money, 2)  # 交易金额
        self.price: float = round(bar.price, 5)  # 最新价
        self.close: float = round(bar.close, 5)  # 收盘价
        self.open: float = round(bar.open, 5)  # 开盘价
        self.high: float = round(bar.high, 5)  # 最高价
        self.low: float = round(bar.low, 5)  # 最低价

    def agg(self, bar: Bar):
        if bar is not None:
            self.volume = self.volume + bar.volume
            self.money = self.money + bar.money
            self.open = bar.open
            self.high = max(self.high, bar.high)
            self.low = min(self.low, bar.low)
        return self


class Pos:
    def __init__(self, pos):
        self.total_amount = getattr(pos, 'amount', 0.0)  # 总持仓数量
        self.avail_amount = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
        self.last_price = getattr(pos, 'last_sale_price', 0.0)  # 最新价格
        self.cost_price = getattr(pos, 'cost_basis', 0.0)  # 成本价格
        self.valuation = round(self.total_amount * self.last_price, 2)  # 市值
        self.principal = round(self.total_amount * self.cost_price, 2)  # 本金


class Log:
    def __init__(self, symbol: str, node: Bin.Node):
        self.node_idx = node.index
        self.node_val = node.turn.curNode.val
        self.turn_idx = node.turn.preTurn.idx
        self.turn_val = node.turn.preTurn.val
        self.turn_tag = node.turn.preTurn.tag
        self.symbol = symbol
        self.amount = 0
        self.status = 0

    def act(self, amount: float):
        self.status = 1 if amount > 0 else -1 if amount < 0 else 0
        self.amount = amount
        return self


class Silo:
    def __init__(self, maxlen=None):
        self.keys: deque[str] = deque(maxlen=maxlen)
        self.data: deque[Bin.Node] = deque(maxlen=maxlen)
        self.dict: dict = {}

        self._agg_per = None  # 聚合周期
        self._agg_bar = None  # 聚合bar
        self._agg_tmp = None  # 聚合node

    def __len__(self):
        if self._agg_tmp is None:
            return len(self.data)
        return len(self.data) + 1

    def at(self, idx: int) -> Bin.Node:
        if self._agg_tmp is None:
            return self.data[idx]
        if idx == -1 or idx == len(self.data):
            return self._agg_tmp
        diff = 1 if idx < 0 else 0
        return self.data[idx + diff]

    def get(self, key: str) -> Bin.Node:
        idx = self.keys.index(key)
        return self.data[idx]

    def add(self, node: Bin.Node):
        self.keys.append(node.index)
        self.data.append(node)

    def agg_per(self, per):
        self._agg_per = per
        return self

    def agg_add(self, node: Bin.Node):
        bar = node.bar.agg(self._agg_bar)
        if self.__is_cut(bar):
            self.keys.append(node.index)
            self.data.append(node)
            self._agg_tmp = None
            self._agg_bar = None
        else:
            self._agg_tmp = node
            self._agg_bar = bar

    def __is_cut(self, bar: Bar) -> bool:
        dt = bar.datetime
        if self._agg_per == '1d':
            return dt.hour == 15 and dt.minute == 0 and dt.second == 0
        if self._agg_per == '5m':
            return dt.minute % 5 == 0
        return True


############################################################
class Var:
    class Base:
        def __init__(self):
            self.begin_time = '09:40:00'  # 开始交易时间
            self.close_time = '14:55:00'  # 关闭交易时间
            self.basic_fund = 10000  # 交易基准金额
            self.start_fund = 3000  # 交易起步金额
            self.least_fund = 1000  # 交易最低金额
            self.cost_limit = 1.50  # 成本上限（比例）
            self.loss_limit = 0.15  # 亏损上限（比例）
            self.gain_limit = 0.10  # 盈利上限（比例）

    class Macd:
        def __init__(self):
            self.fast = 13  # 快线周期
            self.slow = 60  # 慢线周期
            self.sign = 5  # 信号线周期

    class Turn:
        def __init__(self):
            self.least_wave = 0.01  # 最小摆动（比例）
            self.sma_period = 10  # 所参照sma的周期

    class Trad:
        def __init__(self):
            self.rise_bounds = [0.010, 0.015]  # 加仓阈值（比例）
            self.rise_quotas = [0.300, 0.300, 0.300]  # 加仓额度（比例）
            self.rise_macd = 0.003  # macd限制（比例）

            self.fall_bounds = [0.015, 0.020, 0.025]  # 减仓阈值（比例）
            self.fall_quotas = [0.300, 0.400, 0.300]  # 减仓额度（比例）
            self.fall_macd = -0.003  # macd限制（比例）

    class Config:
        def __init__(self):
            self.env = 'Live'  # 测试--Test、线上--Live
            self.base: Var.Base = Var.Base()
            self.macd: Var.Macd = Var.Macd()
            self.turn: Var.Turn = Var.Turn()
            self.trad: Var.Trad = Var.Trad()


class Bin:
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
        class Point:
            def __init__(self, idx: str, val: float, tag: int):
                self.idx = idx  # 索引
                self.val = val  # 值
                self.tag = tag  # 标签
                self.maxApex = None  # 最大顶点
                self.minApex = None  # 最小顶点

        def __init__(self):
            self.tag = 0  # 标签
            self.curNode = None  # 当前节点
            self.preTurn = None  # 前一拐点

    class Node:
        def __init__(self, bar):
            self.index: str = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
            self.bar: Bar = Bar(bar)
            self.sma: Bin.Sma = Bin.Sma()
            self.macd: Bin.Macd = Bin.Macd()
            self.turn: Bin.Turn = Bin.Turn()


class Biz:
    class Bas:
        def __init__(self, cfg: Var.Config, bus: Bus):
            self.cfg = cfg
            self.bus = bus

        def last_pos(self) -> Pos:
            return self.bus.posQue[-1]

        def last_min(self) -> Bin.Node:
            return self.bus.minSet.at(-1)

        def last_day(self) -> Bin.Node:
            return self.bus.daySet.at(-1)

        def is_out_budget(self) -> bool:
            """是否超过本金上限/亏损上限"""
            pos = self.last_pos()
            if pos.principal > self.cfg.base.basic_fund * self.cfg.base.cost_limit:
                return True
            if pos.principal - pos.valuation >= self.cfg.base.basic_fund * self.cfg.base.loss_limit:
                return True
            return False

        def is_out_schedule(self) -> bool:
            """是否在日程时间外"""
            instant = self.last_min().bar.instant
            return instant < self.cfg.base.begin_time or instant > self.cfg.base.close_time

        def is_stop_buy(self) -> bool:
            """是否暂停买入"""
            today = self.last_day()
            if today.sma.sma05 <= today.sma.sma10: return True
            if today.sma.sma05 <= today.sma.sma20: return True
            if today.sma.sma05 <= today.sma.sma30: return True
            if today.sma.sma05 <= today.sma.sma60: return True
            return False

        def is_hit_bound(self, status) -> bool:
            """是否达到阈值边界"""
            turn = self.last_min().turn
            bounds = self.cfg.trad.rise_bounds if status > 0 else self.cfg.trad.fall_bounds
            logs = [log for log in self.bus.logQue if log.status == status and log.turn_idx == turn.preTurn.idx]
            if len(logs) < len(bounds):
                threshold = bounds[len(logs)] * self.bus.ctxMap.get(K.base_price)
                if (turn.curNode.val - turn.preTurn.val) * status > threshold:
                    return True
            return False

    class Qty(Bas):
        def buy_amount(self) -> float:
            """买入数量"""
            quotas = self.bus.ctxMap.get(K.rise_quotas)
            if not quotas:
                return 0

            last_price = self.last_pos().last_price
            base_price = self.bus.ctxMap.get(K.base_price)
            price = base_price if last_price == 0 else last_price
            quota = quotas.pop(0)
            buy_amount = max(self.cfg.base.basic_fund * quota, self.cfg.base.start_fund) / price
            return round(buy_amount / 100) * 100

        def sell_amount(self) -> float:
            """卖出数量"""
            quotas = self.bus.ctxMap.get(K.fall_quotas)
            if quotas:
                quota = quotas.pop(0)
                plan_amount = self.cfg.base.basic_fund * quota
                sell_amount = self.fit_sell_qty(plan_amount)
                return -sell_amount
            return 0

        def fit_sell_qty(self, plan_qty: int):
            """适配卖出的数量"""
            pos = self.last_pos()
            avail_qty = pos.avail_amount
            if avail_qty == 0:
                return 0

            # 计算需要保留的数量
            profit = pos.valuation - pos.principal  # 当前盈利
            start_qty = round(self.cfg.base.start_fund / pos.last_price / 100) * 100 # 起始数量
            least_qty = round(self.cfg.base.least_fund / pos.last_price / 100) * 100 # 最小数量
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
            if self.is_stop_buy(): return False
            if self.is_out_budget():  return False
            if self.is_out_schedule(): return False
            if last_min.sma.sma05 <= last_min.sma.sma10: return False
            if last_min.sma.sma05 <= last_min.sma.sma20: return False
            if last_min.sma.sma05 <= last_min.sma.sma30: return False
            if last_min.sma.sma05 <= last_min.sma.sma60: return False
            if last_min.macd.diff < 0: return False
            if last_min.macd.dea_ < 0: return False
            if last_min.macd.macd < 0: return False

            # 是否达到阈值边界
            return self.is_hit_bound(1)

        def is_sell(self) -> bool:
            """判断是否卖出"""
            last_min = self.last_min()
            if self.is_out_schedule(): return False
            if last_min.sma.sma05 >= last_min.sma.sma10: return False
            if last_min.sma.sma05 >= last_min.sma.sma20: return False
            if last_min.sma.sma05 >= last_min.sma.sma30: return False
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
            last_log = Log(symbol, last_min).act(amount)
            self.bus.logQue.append(last_log)
            self.bus.ctxMap[K.last_log] = last_log


class Line:
    class Sma:
        @staticmethod
        def first(silo: Silo):
            node = silo.at(-1)
            price = node.bar.close
            node.sma.sma05 = price
            node.sma.sma10 = price
            node.sma.sma20 = price
            node.sma.sma30 = price
            node.sma.sma60 = price

        @staticmethod
        def next(silo: Silo):
            Line.Sma.__next(silo, 5)
            Line.Sma.__next(silo, 10)
            Line.Sma.__next(silo, 20)
            Line.Sma.__next(silo, 30)
            Line.Sma.__next(silo, 60)
            pass

        @staticmethod
        def __next(silo: Silo, period: int):
            node = silo.at(-1)
            price = node.bar.close
            pre_sma = silo.at(-2).sma.get(period)
            if len(silo) > period:
                val_1st = silo.at(-period - 1).bar.close
                cur_sma = pre_sma + (price - val_1st) / period
            else:
                cur_sma = (pre_sma * (len(silo) - 1) + price) / len(silo)
            node.sma.set(period, round(cur_sma, 5))

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
            diff = round(fast - slow, 5)
            dea_ = Line.Macd.ema(diff, cfg.macd.sign, prev_node.macd.dea_)
            macd = round((diff - dea_) * 2, 5)

            node.macd.fast = fast
            node.macd.slow = slow
            node.macd.diff = diff
            node.macd.dea_ = dea_
            node.macd.macd = macd

        @staticmethod
        def ema(price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 5)

    class Turn:
        @staticmethod
        def first(silo: Silo):
            node = silo.at(-1)
            node.turn.tag = -2
            node.turn.curNode = Bin.Turn.Point(node.index, node.bar.close, -2)
            node.turn.preTurn = node.turn.curNode

        @staticmethod
        def next(silo: Silo, cfg: Var.Config):
            node = silo.at(-1)
            value = node.sma.get(cfg.turn.sma_period)
            node.turn.tag = 0
            node.turn.curNode = Bin.Turn.Point(node.index, value, 0)
            node.turn.preTurn = silo.at(-2).turn.preTurn
            if len(silo) < 3:
                return

            # 价格最小振幅阈值
            threshold = round(value * cfg.turn.least_wave, 5)
            # 计算顶点、拐点
            Line.Turn.__apex(silo, threshold)
            Line.Turn.__turn(silo, threshold)

        @staticmethod
        def __apex(silo: Silo, threshold):
            point1 = silo.at(-1).turn.curNode
            point2 = silo.at(-2).turn.curNode
            point3 = None
            if point1.val == point2.val:
                return
            for i in range(-3, -len(silo) - 1, -1):
                point3 = silo.at(i).turn.curNode
                if point3.val != point2.val:
                    break
            if point3 is None:
                return

            # 计算顶点
            pre_turn = silo.at(-1).turn.preTurn
            if point3.val < point2.val > point1.val:
                if point2.val - pre_turn.val >= threshold:
                    if pre_turn.maxApex is None or point2.val > pre_turn.maxApex.val:
                        point2.tag = 1
                        pre_turn.maxApex = point2
            elif point3.val > point2.val < point1.val:
                if pre_turn.val - point2.val >= threshold:
                    if pre_turn.minApex is None or point2.val < pre_turn.minApex.val:
                        point2.tag = -1
                        pre_turn.minApex = point2

        @staticmethod
        def __turn(silo: Silo, threshold):
            # 起始拐点
            turn = silo.at(-1).turn
            if turn.preTurn.tag == -2:
                diff = turn.preTurn.val - turn.curNode.val
                if abs(diff) > threshold:
                    turn.preTurn.tag = 1 if diff > 0 else -1
                    silo.at(0).turn.tag = turn.preTurn.tag
                return

            # 计算拐点
            point = None
            pre_turn = turn.preTurn
            if pre_turn.maxApex and pre_turn.maxApex.val - turn.curNode.val >= threshold:
                point = pre_turn.maxApex
            if pre_turn.minApex and turn.curNode.val - pre_turn.minApex.val >= threshold:
                point = pre_turn.minApex
            if point is not None:
                turn.preTurn = point
                silo.get(point.idx).turn.tag = point.tag


############################################################
class Bus:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.daySet = Silo(maxlen=120)  # 日线数据
        self.minSet = Silo(maxlen=240)  # 分线数据
        self.logQue = deque(maxlen=360)  # 操作数据
        self.posQue = deque(maxlen=360)  # 仓位数据
        self.ctxMap = {}  # 上下文数据


class Market:
    def __init__(self, config: Var.Config, trader: Biz.Trader, bus: Bus):
        self.cfg: Var.Config = config
        self.biz: Biz.Trader = trader
        self.bus: Bus = bus
        self.status = 0

    def initialize(self, days, mins):
        self.__handle_bars(self.bus.daySet, days)
        self.__handle_bars(self.bus.minSet, mins, turn=True)
        self.bus.daySet.agg_per('1d')
        self.bus.minSet.agg_per('5m')
        self.status = 1

    def prepare(self, pos):
        if len(self.bus.minSet) == 0:
            self.status = 0
            return
        self.bus.ctxMap[K.rise_quotas] = self.cfg.trad.rise_quotas.copy()
        self.bus.ctxMap[K.fall_quotas] = self.cfg.trad.fall_quotas.copy()
        self.bus.ctxMap[K.base_price] = self.bus.minSet.at(-1).bar.close
        self.bus.ctxMap[K.base_amount] = getattr(pos, 'amount', 0.0)
        self.status = 2

    def running(self, pos, bar):
        if self.status != 2:
            return
        self.bus.posQue.append(Pos(pos))
        self.bus.minSet.agg_add(Bin.Node(bar))
        Line.Sma.next(self.bus.minSet)
        Line.Macd.next(self.bus.minSet, self.cfg)
        Line.Turn.next(self.bus.minSet, self.cfg)
        self.bus.daySet.agg_add(Bin.Node(bar))
        Line.Sma.next(self.bus.daySet)
        Line.Macd.next(self.bus.daySet, self.cfg)

    def trading(self, buy: Callable, sell: Callable):
        if self.status == 2:
            self.biz.trading(buy, sell)

    def __handle_bars(self, silo: Silo, bars, sma=True, macd=True, turn=False):
        if not bars: return
        silo.add(Bin.Node(bars[0]))
        if sma: Line.Sma.first(silo)
        if macd: Line.Macd.first(silo)
        if turn: Line.Turn.first(silo)
        for bar in bars[1:]:
            silo.add(Bin.Node(bar))
            if sma: Line.Sma.next(silo)
            if macd: Line.Macd.next(silo, self.cfg)
            if turn: Line.Turn.next(silo, self.cfg)


############################################################
class Kit:
    def __init__(self, config: str, trader: str):
        self.config = config  # 配置信息
        self.trader = trader  # 交易规则


class Env:
    markets: dict[str, Market] = {}
    classes: dict[str, Type[Var.Config | Biz.Trader]] = {
        "config": Var.Config,
        "trader": Biz.Trader,
    }
    # 黑名单、白名单
    blacks: list[str] = ['515450.SS', '515100.SS']
    whites: dict[str, Kit] = {
        #'515790.SS': Kit('config', 'trader'),
    }


class Manager:
    @staticmethod
    def symbols(positions: dict) -> list[str]:
        codes = list(Env.whites)
        if positions is not None and positions:
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
    symbols = Manager.symbols(positions)
    set_universe(symbols)

    # 初始化
    codes = [symbol for symbol in symbols if Manager.market(symbol).status == 0]
    if len(codes) != 0:
        day_his = get_history(120, frequency='1d', security_list=codes)
        min_his = get_history(240, frequency='5m', security_list=codes)
        for symbol in symbols:
            day_df = day_his.query(f'code in ["{symbol}"]')
            min_df = min_his.query(f'code in ["{symbol}"]')
            days = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in day_df.iterrows()]
            mins = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in min_df.iterrows()]
            Manager.market(symbol).initialize(days, mins)

    # 准备
    for symbol in symbols:
        pos = positions.get(symbol)
        Manager.market(symbol).prepare(pos)


def handle_data(context, data):
    """每个单位周期执行一次"""
    symbols = Env.markets.keys()
    positions = context.portfolio.positions
    for symbol in symbols:
        bar = data[symbol]
        pos = positions.get(symbol)
        Manager.market(symbol).running(pos, bar)
        Manager.market(symbol).trading(order, order)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
