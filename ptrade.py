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
from typing import Callable, Type

import pandas as pd


class K:
    class Bas:
        symbol = 'symbol'
        datetime = 'datetime'
        base_price = 'base_price'
        base_amount = 'base_amount'

    class Bar:
        volume = 'bar.volume'
        money = 'bar.money'
        price = 'bar.price'
        close = 'bar.close'
        open = 'bar.open'
        high = 'bar.high'
        low = 'bar.low'

    class Pos:
        avail_amount = 'avail_amount'
        total_amount = 'total_amount'
        cost_price = 'cost_price'
        last_price = 'last_price'
        valuation = 'valuation'
        principal = 'principal'

    class Log:
        node_idx = 'node_idx'
        turn_idx = 'turn_idx'
        turn_val = 'turn_val'
        amount = 'amount'
        level = 'level'
        type = 'type'

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
        turn_lvl = 'turn.turn_lvl'
        turn_idx = 'turn.turn_idx'
        turn_val = 'turn.turn_val'
        nodes = 'turn.nodes'
        turns = 'turn.turns'


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

    def new_df(self) -> pd.DataFrame:
        bar_dict = self._bar_dict()
        return pd.DataFrame([bar_dict], index=[self.time_str])

    def add_to(self, df: pd.DataFrame):
        df.loc[self.time_str] = self._bar_dict()

    def _bar_dict(self):
        return {
            K.Bas.datetime: self.datetime,
            K.Bar.volume: self.volume,
            K.Bar.money: self.money,
            K.Bar.price: self.price,
            K.Bar.close: self.close,
            K.Bar.open: self.open,
            K.Bar.high: self.high,
            K.Bar.low: self.low,
        }

    class Node:
        def __init__(self):
            self.node_idx = 0  # 节点索引
            self.node_val = 0  # 节点均值
            self.node_lvl = 0  # 节点标记

    class Turn:
        def __init__(self):
            self.turn_idx = 0  # 拐点索引
            self.turn_val = 0  # 拐点均值
            self.turn_lvl = 0  # 拐点标记

            # 加/减仓额度，用于避免重复交易
            self.add_quotas = []
            self.sub_quotas = []

            # 最大/小顶点，用于判断是否有效的拐点
            self.MaxApex: Bar.Node | None = None
            self.MinApex: Bar.Node | None = None


class Pos:
    def __init__(self, pos, bar):
        self.datetime_str = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.avail_amount = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
        self.total_amount = getattr(pos, 'amount', 0.0)  # 总持仓数量
        self.cost_price = getattr(pos, 'cost_basis', 0.0)  # 成本价格
        self.last_price = getattr(pos, 'last_sale_price', 0.0)  # 最新价格
        self.valuation = round(self.total_amount * self.last_price, 2)  # 市值
        self.principal = round(self.total_amount * self.cost_price, 2)  # 本金

    def new_df(self) -> pd.DataFrame:
        pos_dict = self._pos_dict()
        return pd.DataFrame([pos_dict], index=[self.datetime_str])

    def add_to(self, df: pd.DataFrame):
        df.loc[self.datetime_str] = self._pos_dict()

    def _pos_dict(self):
        return {
            K.Pos.avail_amount: self.avail_amount,
            K.Pos.total_amount: self.total_amount,
            K.Pos.cost_price: self.cost_price,
            K.Pos.last_price: self.last_price,
            K.Pos.valuation: self.valuation,
            K.Pos.principal: self.principal,
        }


class Log:
    def __init__(self, info: dict):
        self.node_idx = info[K.Bas.datetime]
        self.turn_idx = info[K.Turn.turn_idx]
        self.turn_lvl = info[K.Turn.turn_lvl]
        self.amount = 0
        self.level = 0
        self.type = 0

    def info(self, amount: float, level: int):
        self.amount = amount
        self.level = level
        self.type = 'Buy' if amount > 0 else 'Sell' if amount < 0 else 'Null'
        return self

    def add_to(self, df: pd.DataFrame):
        df.loc[len(df)] = self._log_dict()

    @staticmethod
    def empty_df() -> pd.DataFrame:
        cols = [K.Log.node_idx, K.Log.turn_idx, K.Log.turn_val, K.Log.amount, K.Log.level, K.Log.type]
        return pd.DataFrame(columns=cols)

    def _log_dict(self):
        return {
            K.Log.node_idx: self.node_idx,
            K.Log.turn_idx: self.turn_idx,
            K.Log.turn_val: self.turn_lvl,
            K.Log.amount: self.amount,
            K.Log.level: self.level,
            K.Log.type: self.type,
        }


class Line:
    class L(ABC):
        def __init__(self, cfg: Config):
            self.Cfg: Config = cfg

        @staticmethod
        def get(df: pd.DataFrame, col: str, idx: int):
            return df[col].iloc[idx]

        @staticmethod
        def set(df: pd.DataFrame, col: str, idx: int, val):
            col_idx = df.columns.get_loc(col)
            df.iat[idx, col_idx] = val

        @abstractmethod
        def _calc(self, price: float, period: int, prev_val: float):
            pass

        def _set_first(self, df: pd.DataFrame, key: str):
            price = self.get(df, K.Bar.close, -1)
            df[key] = pd.Series(dtype='float64')
            self.set(df, key, -1, price)

        def _set_next(self, df: pd.DataFrame, key: str, period: int):
            price = self.get(df, K.Bar.close, -1)
            prev_val = self.get(df, key, -2)
            next_val = self._calc(price, period, prev_val)
            self.set(df, key, -1, next_val)

    class Ema(L):
        def first(self, df: pd.DataFrame):
            self._set_first(df, K.Ema.fast)
            self._set_first(df, K.Ema.slow)
            return self

        def next(self, df: pd.DataFrame):
            self._set_next(df, K.Ema.fast, self.Cfg.Ema.fast)
            self._set_next(df, K.Ema.slow, self.Cfg.Ema.slow)

        def _calc(self, price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

    class Smma(L):
        def first(self, df: pd.DataFrame):
            self._set_first(df, K.Smma.fast)
            self._set_first(df, K.Smma.slow)
            return self

        def next(self, df: pd.DataFrame):
            self._set_next(df, K.Smma.fast, self.Cfg.Smma.fast)
            self._set_next(df, K.Smma.slow, self.Cfg.Smma.slow)

        def _calc(self, price: float, period: int, prev_val: float):
            value = (prev_val * (period - 1) + price) / period
            return round(value, 4)

    class Macd(L):
        def first(self, df: pd.DataFrame):
            self._set_first(df, K.Macd.fast)
            self._set_first(df, K.Macd.slow)
            df[K.Macd.dif] = 0
            df[K.Macd.dea] = 0
            df[K.Macd.macd] = 0
            return self

        def next(self, df: pd.DataFrame):
            self._set_next(df, K.Macd.fast, self.Cfg.Macd.fast)
            self._set_next(df, K.Macd.slow, self.Cfg.Macd.slow)
            self._next_macd(df)

        def _next_macd(self, df: pd.DataFrame):
            period = self.Cfg.Macd.sign
            fast_ema = self.get(df, K.Macd.fast, -1)
            slow_ema = self.get(df, K.Macd.slow, -1)
            prev_dea = self.get(df, K.Macd.dea, -2)

            dif = round(fast_ema - slow_ema, 4)
            dea = self._calc(dif, period, prev_dea)
            macd = round((dif - dea) * 2, 4)

            self.set(df, K.Macd.dif, -1, dif)
            self.set(df, K.Macd.dea, -1, dea)
            self.set(df, K.Macd.macd, -1, macd)

        def _calc(self, price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

    class Turn(L):
        def __init__(self, cfg: Config):
            super().__init__(cfg)
            self.threshold = 10000

        def new_node(self, df: pd.DataFrame) -> Bar.Node:
            node = Bar.Node()
            node.node_idx = df.index[-1]
            node.node_val = self.get(df, K.Ema.fast, -1)
            return node

        def new_turn(self, node: Bar.Node) -> Bar.Turn:
            turn = Bar.Turn()
            turn.turn_idx = node.node_idx
            turn.turn_val = node.node_val
            turn.turn_lvl = node.node_lvl
            turn.add_quotas = copy.copy(self.Cfg.Rise.add_quotas)
            turn.sub_quotas = copy.copy(self.Cfg.Fall.sub_quotas)
            return turn

        def first(self, df: pd.DataFrame, ext: dict):
            node = self.new_node(df)
            turn = self.new_turn(node)
            turn.turn_lvl = -2
            ext[K.Turn.nodes] = [node]
            ext[K.Turn.turns] = [turn]

            # 最小振幅价格差
            base_price = ext.get(K.Bas.base_price)
            self.threshold = round(base_price * self.Cfg.Turn.least_wave, 4)

            df[K.Turn.turn_lvl] = node.node_lvl
            df[K.Turn.turn_idx] = node.node_idx
            df[K.Turn.turn_val] = node.node_val
            return self

        def next(self, df: pd.DataFrame, ext: dict):
            # 先预设值
            turn_idx = self.get(df, K.Turn.turn_idx, -2)
            turn_val = self.get(df, K.Turn.turn_val, -2)
            self.set(df, K.Turn.turn_lvl, -1, 0)
            self.set(df, K.Turn.turn_idx, -1, turn_idx)
            self.set(df, K.Turn.turn_val, -1, turn_val)

            # 跳过连续相等的节点
            nodes = ext.get(K.Turn.nodes)
            node = self.new_node(df)
            if node.node_val == nodes[-1].node_val:
                return

            # 节点数据不得小于3条
            nodes.append(node)
            if len(nodes) < 3:
                return

            # 计算顶点、拐点
            nodes = ext.get(K.Turn.nodes)
            turns = ext.get(K.Turn.turns)
            self._calc_apex(nodes, turns)
            self._calc_turn(df, nodes, turns)

        def _calc_apex(self, nodes: list[Bar.Node], turns: list[Bar.Turn]):
            """计算顶点"""
            prev_node = nodes[-3]
            midd_node = nodes[-2]
            last_node = nodes[-1]
            last_turn = turns[-1]
            if prev_node.node_val < midd_node.node_val > last_node.node_val:
                midd_node.node_lvl = 1
                self._max_apex(midd_node, last_turn)
            elif prev_node.node_val > midd_node.node_val < last_node.node_val:
                midd_node.node_lvl = -1
                self._min_apex(midd_node, last_turn)

        def _calc_turn(self, df: pd.DataFrame, nodes: list[Bar.Node], turns: list[Bar.Turn]):
            """计算拐点"""
            # 起始拐点
            first_turn = turns[0]
            if first_turn.turn_lvl == -2:
                diff = round(first_turn.turn_val - nodes[-1].node_val, 4)
                if abs(diff) > self.threshold:
                    first_turn.turn_lvl = 1 if diff > 0 else -1
                    self.set(df, K.Turn.turn_lvl, 0, first_turn.turn_lvl)
                return

            # 计算拐点
            turn = self._get_turn(nodes[-1], turns[-1])
            if turn is not None:
                turns.append(turn)
                idx = df.index.get_loc(turn.turn_idx)
                self.set(df, K.Turn.turn_lvl, idx, turn.turn_lvl)
                self.set(df, K.Turn.turn_idx, -1, turn.turn_idx)
                self.set(df, K.Turn.turn_val, -1, turn.turn_val)

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
class Config:
    class _Bas:
        def __init__(self):
            self.cost_limit = 1.50  # 成本上限（比例）
            self.loss_limit = 0.15  # 亏损上限（比例）
            self.gain_limit = 0.05  # 盈利上限（比例）
            self.foot_funds = 3000  # 最小金额
            self.base_funds = 8000  # 基准资金

    class _Ema:
        def __init__(self):
            self.fast = 10  # 快线周期
            self.slow = 30  # 慢线周期

    class _Smma:
        def __init__(self):
            self.fast = 10  # 快线周期
            self.slow = 30  # 慢线周期

    class _Macd:
        def __init__(self):
            self.fast = 13  # 快线周期
            self.slow = 60  # 慢线周期
            self.sign = 5  # 信号线周期

    class _Turn:
        def __init__(self):
            self.least_wave = 0.004  # 最小摆动（比例）

    class _Rise:
        def __init__(self):
            self.add_quotas = [0.300]  # 加仓额度（比例）
            self.thresholds = [0.004]  # 加仓阈值（比例）
            self.macd_limit = 0.0015  # macd下限（比例）

    class _Fall:
        def __init__(self):
            self.sub_quotas = [0.300, 0.400, 0.300]  # 减仓额度（比例）
            self.thresholds = [0.004, 0.007, 0.010]  # 减仓阈值（比例）

    def __init__(self):
        self.Bas: Config._Bas = Config._Bas()
        self.Ema: Config._Ema = Config._Ema()
        self.Smma: Config._Smma = Config._Smma()
        self.Macd: Config._Macd = Config._Macd()
        self.Turn: Config._Turn = Config._Turn()
        self.Rise: Config._Rise = Config._Rise()
        self.Fall: Config._Fall = Config._Fall()


class Broker(ABC):
    def __init__(self, bus: Bus):
        self.Bus = bus
        self.Cfg = bus.Cfg
        self.Ext = bus.Ext

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

    def over_budget(self) -> bool:
        """超过本金上限 or 超过亏损上限"""
        pos = self.last_pos()
        if pos[K.Pos.principal] > self.Cfg.Bas.base_funds * self.Cfg.Bas.cost_limit:
            return True
        if pos[K.Pos.principal] - pos[K.Pos.valuation] >= self.Cfg.Bas.base_funds * self.Cfg.Bas.loss_limit:
            return True
        return False

    def has_no_amount(self) -> bool:
        """没有可用持仓"""
        pos = self.last_pos()
        return pos[K.Pos.avail_amount] <= self.remain_amount()

    def remain_amount(self) -> int:
        """获取保留的股票数量"""
        pos = self.last_pos()
        if pos[K.Pos.total_amount] > pos[K.Pos.avail_amount]:
            return 0
        if pos[K.Pos.valuation] - pos[K.Pos.principal] > self.Cfg.Bas.base_funds * self.Cfg.Bas.gain_limit:
            return 0
        return 100

    def log(self, level: int, amount: float):
        """添加日志"""
        curr_min = self.last_min()
        Log(curr_min).info(amount, level).add_to(self.Bus.Log)

    def last_pos(self) -> dict:
        return self.Bus.Pos.iloc[-1].to_dict()

    def last_day(self) -> dict:
        return self.Bus.Day.iloc[-1].to_dict()

    def last_min(self) -> dict:
        return self.Bus.Min.iloc[-1].to_dict()


class TurnBroker(Broker):
    def is_buy(self) -> bool:
        curr_day = self.last_day()
        curr_min = self.last_min()

        # 前5分钟不买入
        if curr_min[K.Bas.datetime].time() < time(9, 35, 0): return False
        # 超过亏损上限 or 超过本金上限
        if self.over_budget(): return False
        # 日线下跌
        if curr_day[K.Smma.fast] <= curr_day[K.Smma.slow]: return False
        # 分钟线下跌
        if curr_min[K.Ema.fast] <= curr_min[K.Ema.slow]: return False
        # 小于MACD值下限
        if curr_min[K.Macd.macd] < self.Cfg.Rise.macd_limit: return False
        # 涨幅未达到阈值 or 重复操作
        level = self.__rise_level()
        turn = self.Ext.get(K.Turn.turns)[-1]
        if level == -1 or turn.add_quotas[level] == 0: return False
        # 决定买入
        return True

    def is_sell(self) -> bool:
        curr_min = self.last_min()

        # 没有可用持仓
        if self.has_no_amount(): return False
        # 分钟线上涨
        if curr_min[K.Ema.fast] >= curr_min[K.Ema.slow]: return False
        # 跌幅未达到阈值 or 重复操作
        level = self.__fall_level()
        turn = self.Ext.get(K.Turn.turns)[-1]
        if level == -1 or turn.sub_quotas[level] == 0: return False
        # 决定卖出
        return True

    def do_buy(self, func: Callable):
        """执行买入"""
        turn = self.Ext.get(K.Turn.turns)[-1]
        lots = turn.add_quotas
        level = self.__rise_level()

        base_price = self.Ext.get(K.Bas.base_price)
        buy_amount = max(self.Cfg.Bas.base_funds * lots[level], self.Cfg.Bas.foot_funds) / base_price
        amount = round(buy_amount / 100) * 100

        # 执行买入
        func(self.Bus.symbol, amount)
        lots[level] = 0.0
        self.log(level, amount)

    def do_sell(self, func: Callable):
        """执行卖出"""
        turn = self.Ext.get(K.Turn.turns)[-1]
        lots = turn.fall_lots
        level = self.__fall_level()

        # 最小数量：根据基准本金
        base_price = self.Ext.get(K.Bas.base_price)
        min_qty = max(self.Cfg.Bas.base_funds * lots[level], self.Cfg.Bas.foot_funds) / base_price
        # 减仓数量：根据当日初始可用持仓
        base_amount = self.Ext.get(K.Bas.base_amount)
        cur_qty = base_amount * lots[level]
        # 避免低仓位时，还分多次减仓
        sell_qty = max(min_qty, cur_qty)
        # 不得超过当前可用持仓
        sell_amount = min(sell_qty, self.Bus.Pos.avail_amount)
        # 调整到100的倍数，并留下底仓
        amount = round(sell_amount / 100) * 100 - self.Bus.Pos.remain_amount()

        # 执行卖出
        func(self.Bus.symbol, -amount)
        lots[level] = 0.0
        self.log(level, amount)

    def __rise_level(self):
        """当前上涨等级"""
        return self.__calc_level(self.Bus.Cfg.Rise.thresholds)

    def __fall_level(self):
        """当前下跌等级"""
        return self.__calc_level(self.Bus.Cfg.Fall.thresholds)

    def __calc_level(self, thresholds):
        """计算涨跌等级"""
        curr_min = self.last_min()
        base_price = self.Ext.get(K.Bas.base_price)
        diff_value = abs(curr_min[K.Ema.fast] - curr_min[K.Turn.turn_val])
        diff_ratio = round(diff_value / base_price, 4)
        for threshold in reversed(thresholds):
            if diff_ratio > threshold:
                return thresholds.index(threshold)
        return -1


############################################################
class Env:
    def __init__(self, config: str, broker: str):
        self.config = config
        self.broker = broker


class Bus:
    def __init__(self, symbol: str, config: Config):
        self.symbol = symbol  # 股票代码
        self.Cfg = config  # 配置信息
        self.Pos = None  # 仓位数据
        self.Day = None  # 日线数据
        self.Min = None  # 分钟数据
        self.Log = None  # 日志数据
        self.Ext = {}  # 扩展信息


class Market:
    def __init__(self, bus: Bus, broker: Broker):
        self.status = 0  # 状态：0-Initial、1-Started、2-Running
        self.Broker: Broker = broker
        self.Bus: Bus = bus

        self.Ema = None
        self.Smma = None
        self.Macd = None
        self.Turn = None

    def prepare(self, pos, bars):
        # 日线数据
        cfg = self.Bus.Cfg
        day = Bar(bars[0]).new_df()
        self.Smma = Line.Smma(cfg).first(day)
        for bar in bars[1:]:
            Bar(bar).add_to(day)
            self.Smma.next(day)
        self.Bus.Day = day

        # 仓位数据、基准价格、基准持仓
        position = Pos(pos, bars[-1])
        self.Bus.Log = Log.empty_df()
        self.Bus.Pos = position.new_df()
        self.Bus.Ext[K.Bas.base_price] = round(bars[-1].close, 4)
        self.Bus.Ext[K.Bas.base_amount] = position.avail_amount
        self.status = 1

    def running(self, pos, bar):
        cfg = self.Bus.Cfg
        Pos(pos, bar).add_to(self.Bus.Pos)
        if self.status == 2:
            Bar(bar).add_to(self.Bus.Min)
            self.Ema.next(self.Bus.Min)
            self.Macd.next(self.Bus.Min)
            self.Turn.next(self.Bus.Min, self.Bus.Ext)
            return

        if self.status == 1:
            self.Bus.Min = Bar(bar).new_df()
            self.Ema = Line.Ema(cfg).first(self.Bus.Min)
            self.Macd = Line.Macd(cfg).first(self.Bus.Min)
            self.Turn = Line.Turn(cfg).first(self.Bus.Min, self.Bus.Ext)
            self.status = 2
            return

        if self.status == 0:
            self.prepare(pos, [bar])

    def trading(self, func: Callable):
        if self.Broker.is_buy():
            self.Broker.do_buy(func)
            return
        if self.Broker.is_sell():
            self.Broker.do_sell(func)


class Manager:
    markets: dict[str, Market] = {}
    configs: dict[str, Config] = {
        "config": Config(),
    }
    brokers: dict[str, Type[Broker]] = {
        'broker': TurnBroker,
    }

    # 黑名单、白名单
    blacks: list[str] = ['515450.SS', '515100.SS']
    whites: list[str] = []
    stocks: dict[str, Env] = {
        '159857.SZ': Env('config', 'broker'),
    }

    @staticmethod
    def market(symbol: str) -> Market:
        market = Manager.markets.get(symbol)
        if market is not None:
            return market

        env = Manager.stocks.get(symbol, Env('config', 'broker'))
        config = Manager.configs.get(env.config)
        broker = Manager.brokers.get(env.broker)

        bus = Bus(symbol, config)
        market = Market(bus, broker(bus))
        Manager.markets[symbol] = market
        return market

    @staticmethod
    def get_symbols(positions: dict) -> list[str]:
        codes = Manager.whites.copy()
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
        pos = positions.get(symbol)
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in df.iterrows()]
        Manager.market(symbol).prepare(pos, bars)


def handle_data(context, data):
    """每个单位周期执行一次"""
    positions = context.portfolio.positions
    for symbol in g.symbols:
        bar = data[symbol]
        pos = positions.get(symbol)
        Manager.market(symbol).running(pos, bar)
        Manager.market(symbol).trading(order)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
