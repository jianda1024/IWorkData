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

from abc import ABC, abstractmethod
from datetime import time
from types import SimpleNamespace
from typing import Self, Callable, Type

import pandas as pd


class K:
    datetime = 'datetime'

    class Pos:
        base_funds = 'pos.base_funds'
        cost_limit = 'pos.cost_limit'
        loss_limit = 'pos.loss_limit'
        gain_limit = 'pos.gain_limit'

        avail_amount = 'pos.avail_amount'
        total_amount = 'pos.total_amount'
        cost_price = 'pos.cost_price'
        last_price = 'pos.last_price'

    class Bar:
        volume = 'bar.volume'
        money = 'bar.money'
        price = 'bar.price'
        close = 'bar.close'
        open = 'bar.open'
        high = 'bar.high'
        low = 'bar.low'

    class Ema:
        fast = 'ema.fast'
        slow = 'ema.slow'

    class Smma:
        fast = 'smma.fast'
        slow = 'smma.slow'

    class Macd:
        fast = 'macd.fast'
        slow = 'macd.slow'
        sign = 'macd.sign'

        dif = 'macd.dif'
        dea = 'macd.dea'
        macd = 'macd.macd'

    class Turn:
        apex_val = 'turn.apex_val'
        turn_val = 'turn.turn_val'
        prev_idx = 'turn.prev_idx'

    class Rise:
        add_quotas = 'rise.add_quotas'
        thresholds = 'rise.thresholds'
        macd_limit = 'rise.macd_limit'

    class Fall:
        cut_quotas = 'fall.cut_quotas'
        thresholds = 'fall.thresholds'

    class Wave:
        min_turn = 'wave.min_turn'
        rise_lvl = 'wave.rise_lvl'
        fall_lvl = 'wave.fall_lvl'


class Pos:
    def __init__(self, pos, bar):
        self.datetime_str = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.avail_amount = getattr(pos, 'enable_amount', 0.0)
        self.total_amount = getattr(pos, 'amount', 0.0)
        self.cost_price = getattr(pos, 'cost_basis', 0.0)
        self.last_price = getattr(pos, 'last_sale_price', 0.0)
        self.datetime = bar.datetime

    def pos_df(self):
        pos_dict = {
            K.datetime: self.datetime,
            K.Pos.avail_amount: self.avail_amount,
            K.Pos.total_amount: self.total_amount,
            K.Pos.cost_price: self.cost_price,
            K.Pos.last_price: self.last_price,
        }
        return pd.DataFrame([pos_dict], index=[self.datetime_str])


class Bar:
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

    def bar_dict(self):
        return {
            K.datetime: self.datetime,
            K.Bar.volume: self.volume,
            K.Bar.money: self.money,
            K.Bar.price: self.price,
            K.Bar.close: self.close,
            K.Bar.open: self.open,
            K.Bar.high: self.high,
            K.Bar.low: self.low,
        }

    def day_df(self):
        bar_dict = self.bar_dict()
        day_dict = {
            K.Smma.fast: None,
            K.Smma.slow: None,
        }
        row_dict = bar_dict | day_dict
        return pd.DataFrame([row_dict], index=[self.time_str])

    def min_df(self):
        bar_dict = self.bar_dict()
        min_dict = {
            K.Ema.fast: None,
            K.Ema.slow: None,

            K.Macd.fast: None,
            K.Macd.slow: None,
            K.Macd.dif: None,
            K.Macd.dea: None,
            K.Macd.macd: None,

            K.Turn.apex_val: 0,
            K.Turn.turn_val: 0,
            K.Turn.prev_idx: '',

            K.Wave.rise_lvl: -2,
            K.Wave.fall_lvl: -2,
        }
        row_dict = bar_dict | min_dict
        return pd.DataFrame([row_dict], index=[self.time_str])


class Line:
    class L(ABC):
        def __init__(self, cfg: pd.DataFrame, df: pd.DataFrame):
            self.cfg = cfg
            self.df = df

        def _first(self, key: str):
            price = self.df[K.Bar.close].iloc[-1]
            self.df[key].iloc[-1] = price

        def _next(self, key: str):
            price = self.df[K.Bar.close].iloc[-1]
            period = self.cfg[key].iloc[-1]
            prev_val = self.df[key].iloc[-2]
            next_val = self._calc(price, period, prev_val)
            self.df[key].iloc[-1] = next_val

        @abstractmethod
        def _calc(self, price: float, period: int, prev_val: float):
            pass

    class Ema(L):
        def first(self):
            self._first(K.Ema.fast)
            self._first(K.Ema.slow)

        def next(self):
            self._next(K.Ema.fast)
            self._next(K.Ema.slow)

        def _calc(self, price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

    class Smma(L):
        def first(self):
            self._first(K.Smma.fast)
            self._first(K.Smma.slow)

        def next(self):
            self._next(K.Smma.fast)
            self._next(K.Smma.slow)

        def _calc(self, price: float, period: int, prev_val: float):
            value = (prev_val * (period - 1) + price) / period
            return round(value, 4)

    class Macd(L):
        def first(self):
            self._first(K.Macd.fast)
            self._first(K.Macd.slow)
            self.df[K.Macd.dif].iloc[-1] = 0
            self.df[K.Macd.dea].iloc[-1] = 0
            self.df[K.Macd.macd].iloc[-1] = 0

        def next(self):
            self._next(K.Macd.fast)
            self._next(K.Macd.slow)
            self._next_macd()

        def _next_macd(self):
            period = self.cfg[K.Macd.sign].iloc[-1]
            prev_dea = self.df[K.Macd.dea].iloc[-2]
            fast_ema = self.df[K.Macd.fast].iloc[-1]
            slow_ema = self.df[K.Macd.slow].iloc[-1]

            dif = round(fast_ema - slow_ema, 4)
            dea = self._calc(dif, period, prev_dea)
            macd = round((dif - dea) * 2, 4)

            self.df[K.Macd.dif].iloc[-1] = dif
            self.df[K.Macd.dea].iloc[-1] = dea
            self.df[K.Macd.macd].iloc[-1] = macd

        def _calc(self, price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

    class Turn(L):
        def next(self):
            # 判断顶点

            pass


        def _calc(self, price: float, period: int, prev_val: float):
            pass


class Config:
    def __init__(self):
        self.cfg = {
            # 仓位
            K.Pos.base_funds: 8000,  # 基础资金
            K.Pos.cost_limit: 1.50,  # 成本上限（比例）
            K.Pos.loss_limit: 0.15,  # 亏损上限（比例）
            K.Pos.gain_limit: 0.05,  # 盈利上限（比例）

            # 指数移动平均线
            K.Ema.fast: 10,
            K.Ema.slow: 30,

            # 平滑移动平均线
            K.Smma.fast: 10,
            K.Smma.slow: 30,

            # 指数平滑异同移动平均线
            K.Macd.fast: 13,
            K.Macd.slow: 60,
            K.Macd.sign: 5,

            # 加仓
            K.Rise.add_quotas: [0.300],  # 加仓额度（比例）
            K.Rise.thresholds: [0.004],  # 加仓阈值（比例）
            K.Rise.macd_limit: 0.0015,  # macd下限（比例）

            # 减仓
            K.Fall.cut_quotas: [0.300, 0.400, 0.300],  # 减仓额度（比例）
            K.Fall.thresholds: [0.004, 0.007, 0.010],  # 减仓阈值（比例）

            # 波动
            K.Wave.min_turn: 0.04  # 最小摆动（比例）
        }

    def set(self, key, value):
        self.cfg[key] = value
        return self

    def df(self) -> pd.DataFrame:
        return pd.DataFrame([self.cfg])


############################################################
class NodeBar:
    def __init__(self, cfg, bar):
        self.Cfg: Config = cfg
        self.Bar: Bar = Bar(bar)
        self.Ema: Bar.Ema = Bar.Ema()
        self.Smma: Bar.Smma = Bar.Smma()
        self.Macd: Bar.Macd = Bar.Macd()
        self.Mark: NodeBar.Mark = NodeBar.Mark()

        self.datetime = self.Bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.price = self.Bar.price
        self.turn_ema = 0.0
        self.turn_val = 0

    def first(self) -> Self:
        """初始节点，NodeBar"""
        self.Ema.first(self.Bar)
        self.Smma.first(self.Bar)
        self.Macd.first(self.Bar)
        self.turn_ema = self.Ema.fast
        self.turn_val = -2
        return self

    def next(self, nodes: list[Self]) -> Self:
        """下一个节点，NodeBar"""
        self.Ema.next(self.Cfg, self.Bar, nodes[-1].Ema)
        self.Smma.next(self.Cfg, self.Bar, nodes[-1].Smma)
        self.Macd.next(self.Cfg, self.Bar, nodes[-1].Macd)
        self.turn_ema = self.Ema.fast
        return self

    def turn(self):
        """转换为拐点，TurnBar"""
        return TurnBar(self.Cfg, self)

    class Mark:
        def __init__(self):
            # 涨跌等级，标记在节点上，用于避免重复计算
            self.rise_level = -2
            self.fall_level = -2


class TurnBar:
    def __init__(self, cfg, node: NodeBar):
        self.datetime = node.datetime
        self.turn_ema = node.turn_ema
        self.turn_val = node.turn_val

        # 最大/小值点，用于判断是否有效的拐点
        self._max_node: NodeBar | None = None
        self._min_node: NodeBar | None = None

        # 涨跌批次，标记在拐点上，用于避免重复交易
        self.rise_lots = cfg.Rise.add_quotas.copy()
        self.fall_lots = cfg.Fall.sub_quotas.copy()

    def max_node(self, node: NodeBar, threshold):
        """到下一拐点前：最大的凸点"""
        if node.turn_ema - self.turn_ema > threshold:
            if self._max_node is None or node.turn_ema > self._max_node.turn_ema:
                self._max_node = node

    def min_node(self, node: NodeBar, threshold):
        """到下一拐点前：最小的凹点"""
        if self.turn_ema - node.turn_ema > threshold:
            if self._min_node is None or node.turn_ema < self._min_node.turn_ema:
                self._min_node = node

    def next_turn(self, node: NodeBar, threshold):
        """下一拐点"""
        if self.turn_val == -2:
            diff = round(self.turn_ema - node.turn_ema, 4)
            if abs(diff) > threshold:
                self.turn_val = 1 if diff > 0 else -1
            return None
        if self._max_node and self._max_node.turn_ema - node.turn_ema > threshold:
            return self._max_node.turn()
        if self._min_node and node.turn_ema - self._min_node.turn_ema > threshold:
            return self._min_node.turn()
        return None


############################################################
class PosSet:
    def __init__(self, cfg):
        self.Cfg: Config = cfg  # 全局配置
        self.avail_amount = 0.0  # 可用持仓数量
        self.total_amount = 0.0  # 总持仓数量
        self.cost_price = 0.0  # 成本价格
        self.last_price = 0.0  # 最新价格
        self.valuation = 0.0  # 市值
        self.principal = 0.0  # 本金

    def update(self, pos):
        self.avail_amount = getattr(pos, 'enable_amount', 0.0)
        self.total_amount = getattr(pos, 'amount', 0.0)
        self.cost_price = getattr(pos, 'cost_basis', 0.0)
        self.last_price = getattr(pos, 'last_sale_price', 0.0)
        self.valuation = round(self.total_amount * self.last_price, 2)
        self.principal = round(self.total_amount * self.cost_price, 2)

    def over_limit(self) -> bool:
        """超过亏损上限 or 超过本金上限"""
        if self.principal - self.valuation >= self.Cfg.Pos.base_principal * self.Cfg.Pos.loss_limit:
            return True
        return self.principal > self.Cfg.Pos.base_principal * self.Cfg.Pos.cost_limit

    def has_no_amount(self) -> bool:
        """没有可用持仓"""
        return self.avail_amount <= self.remain_amount()

    def remain_amount(self) -> int:
        """获取保留的股票数量"""
        if self.total_amount > self.avail_amount:
            return 0
        if self.valuation - self.principal > self.Cfg.Pos.base_principal * self.Cfg.Pos.gain_limit:
            return 0
        return 100


class DaySet:
    def __init__(self, cfg):
        self.Cfg: Config = cfg  # 全局配置
        self.nodes: list[NodeBar] = []  # 节点集合

    def prepare(self, bars):
        self.Cfg.shift("ByDay")
        self.nodes.clear()
        self.nodes.append(NodeBar(self.Cfg, bars[0]).first())
        for bar in bars[1:]:
            node = NodeBar(self.Cfg, bar).next(self.nodes)
            self.nodes.append(node)
        self.Cfg.shift()


class MinSet:
    def __init__(self, cfg):
        self.Cfg: Config = cfg  # 全局配置
        self.base_price: float = 0.0  # 基准价格
        self.base_amount: float = 0.0  # 基准持仓
        self.nodes: list[NodeBar] = []  # 节点集合
        self.turns: list[TurnBar] = []  # 拐点集合

    def prepare(self, bar):
        self.base_price = round(bar.close, 4)
        self.nodes.clear()
        self.turns.clear()

    def first(self, bar, amount):
        self.base_amount = amount
        node = NodeBar(self.Cfg, bar).first()
        turn = node.turn()
        self.nodes.append(node)
        self.turns.append(turn)

    def next(self, bar):
        node = NodeBar(self.Cfg, bar).next(self.nodes)
        if node.turn_ema == self.nodes[-1].turn_ema:
            return
        self.nodes.append(node)
        if len(self.nodes) < 3:
            return

        # 最小振幅价格差
        threshold = round(self.base_price * self.Cfg.Wave.min_swing, 4)

        # 计算凹凸点
        prev = self.nodes[-3]
        node = self.nodes[-2]
        post = self.nodes[-1]
        if prev.turn_ema < node.turn_ema > post.turn_ema:
            node.turn_val = 1
            self.turns[-1].max_node(node, threshold)
        elif prev.turn_ema > node.turn_ema < post.turn_ema:
            node.turn_val = -1
            self.turns[-1].min_node(node, threshold)

        # 计算拐点
        turn = self.turns[-1].next_turn(post, threshold)
        self.turns.append(turn) if turn else None


############################################################
class Broker(ABC):

    @abstractmethod
    def is_buy(self) -> bool:
        pass

    @abstractmethod
    def is_sell(self) -> bool:
        pass

    @abstractmethod
    def do_buy(self, func: Callable):
        pass

    @abstractmethod
    def do_sell(self, func: Callable):
        pass


class TurnBroker(Broker):
    def __init__(self, bus):
        self.Bus: StockBus = bus

    def is_buy(self) -> bool:
        yest, node, turn = self.Bus.bars()
        # 非波谷
        if turn.turn_val != -1:
            return False
        # 前10分钟不买入
        if node.Bar.datetime.time() < time(9, 40, 0):
            return False
        # 超过亏损上限 or 超过本金上限
        if self.Bus.PosSet.over_limit():
            return False
        # 日线下跌 or 分钟线下跌
        if yest.Smma.is_fall() or node.Smma.is_fall():
            return False
        # 小于MACD值下限
        if node.Macd.macd < self.Bus.Config.Rise.macd_limit:
            return False
        # 涨幅未达到阈值 or 重复操作
        level = self.__rise_level()
        if level == -1 or turn.rise_lots[level] == 0:
            return False
        # 决定买入
        return True

    def is_sell(self) -> bool:
        yest, node, turn = self.Bus.bars()
        # 非波峰
        if turn.turn_val != 1:
            return False
        # 没有可用持仓
        if self.Bus.PosSet.has_no_amount():
            return False
        # 分钟线上涨
        if node.Smma.is_rise():
            return False
        # 跌幅未达到阈值 or 重复操作
        level = self.__fall_level()
        if level == -1 or turn.fall_lots[level] == 0:
            return False
        # 决定卖出
        return True

    def do_buy(self, func: Callable):
        """执行买入"""
        _, _, turn = self.Bus.bars()
        level = self.__rise_level()
        lots = turn.rise_lots
        buy_amount = self.Bus.Config.Pos.base_principal / self.Bus.MinSet.base_price * lots[level]
        amount = round(buy_amount / 100) * 100

        # 执行买入
        func(self.Bus.symbol, amount)
        lots[level] = 0.0

    def do_sell(self, func: Callable):
        """执行卖出"""
        _, _, turn = self.Bus.bars()
        level = self.__fall_level()
        lots = turn.fall_lots

        # 最小数量：根据基准本金
        today = self.Bus.MinSet
        min_qty = self.Bus.Config.Pos.base_principal / today.base_price * lots[level]
        # 减仓数量：根据当日初始可用持仓
        cur_qty = today.base_amount * lots[level]
        # 避免低仓位时，还分多次减仓
        sell_qty = max(min_qty, cur_qty)
        # 不得超过当前可用持仓
        sell_amount = min(sell_qty, self.Bus.PosSet.avail_amount)
        # 调整到100的倍数，并留下底仓
        amount = round(sell_amount / 100) * 100 - self.Bus.PosSet.remain_amount()

        # 执行卖出
        func(self.Bus.symbol, -amount)
        lots[level] = 0.0

    def __rise_level(self):
        """当前上涨等级"""
        _, node, _ = self.Bus.bars()
        mark = node.Mark
        if mark.rise_level == -2:
            mark.rise_level = self.__calc_level(self.Bus.Config.Rise.thresholds)
        return mark.rise_level

    def __fall_level(self):
        """当前下跌等级"""
        _, node, _ = self.Bus.bars()
        mark = node.Mark
        if mark.fall_level == -2:
            mark.fall_level = self.__calc_level(self.Bus.Config.Fall.thresholds)
        return mark.fall_level

    def __calc_level(self, thresholds):
        """计算涨跌等级"""
        _, node, turn = self.Bus.bars()
        diff_value = abs(node.turn_ema - turn.turn_ema)
        diff_ratio = round(diff_value / self.Bus.MinSet.base_price, 4)
        for threshold in reversed(thresholds):
            if diff_ratio > threshold:
                return thresholds.index(threshold)
        return -1


############################################################
class StockMarket:
    @staticmethod
    def prepare(bus: StockBus, bars):
        bus.DaySet.prepare(bars)
        bus.MinSet.prepare(bars[-1])
        bus.status = 1

    @staticmethod
    def running(bus: StockBus, bar, pos):
        bus.PosSet.update(pos)
        if bus.status == 2:
            bus.MinSet.next(bar)
            return
        if bus.status == 1:
            amount = bus.PosSet.avail_amount
            bus.MinSet.first(bar, amount)
            bus.status = 2
            return
        if bus.status == 0:
            StockMarket.prepare(bus, [bar])


class StockTrader:
    @staticmethod
    def trading(bus, func: Callable):
        """执行交易"""
        clazz = bus.Broker
        broker = clazz(bus)
        if broker.is_buy():
            broker.do_buy(func)
            return
        if broker.is_sell():
            broker.do_sell(func)


############################################################
class StockBus:
    def __init__(self, symbol: str, cfg: Config, broker: Type[Broker]):
        self.status: int = 0  # 状态：0-Initial、1-Started、2-Running
        self.symbol: str = symbol  # 股票代码
        self.Config: Config = cfg  # 配置信息
        self.PosSet: PosSet = PosSet(cfg)  # 当前仓位
        self.DaySet: DaySet = DaySet(cfg)  # 日线数据
        self.MinSet: MinSet = MinSet(cfg)  # 分钟数据
        self.Broker: Type[Broker] = broker  # 经纪人

    def bars(self) -> tuple[NodeBar, NodeBar, TurnBar]:
        return self.DaySet.nodes[-1], self.MinSet.nodes[-1], self.MinSet.turns[-1]


def init() -> list[StockBus]:
    buses: list[StockBus] = []
    codes = ['159857.SZ']
    config = Config()
    for code in codes:
        bus = StockBus(code, config, TurnBroker)
        buses.append(bus)
    return buses


############################################################
def initialize(context):
    """启动时执行一次"""
    buses = init()
    symbols = [bus.symbol for bus in buses]
    g.buses = buses
    g.symbols = symbols
    set_universe(symbols)
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    history = get_history(60, frequency='1d')
    for bus in g.buses:
        df = history.query(f'code in ["{bus.symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in df.iterrows()]
        StockMarket.prepare(bus, bars)


def handle_data(context, data):
    """每个单位周期执行一次"""
    positions = context.portfolio.positions
    for bus in g.buses:
        bar = data[bus.symbol]
        pos = positions.get(bus.symbol)
        StockMarket.running(bus, bar, pos)
        StockTrader.trading(bus, order)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
