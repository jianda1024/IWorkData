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
    class Bar:
        datetime = 'datetime'
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
        turn_val = 'turn.turn_val'
        turn_idx = 'turn.turn_idx'
        turn_ema = 'turn.turn_ema'
        nodes = 'turn.nodes'
        turns = 'turn.turns'


############################################################
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
            K.Bar.datetime: self.datetime,
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
            self.node_ema = 0  # 节点均值
            self.apex_val = 0  # 顶点标记

    class Turn:
        def __init__(self):
            self.turn_idx = 0  # 拐点索引
            self.turn_ema = 0  # 拐点均值
            self.turn_val = 0  # 拐点标记

            # 加/减仓额度，用于避免重复交易
            self.add_quotas = []
            self.sub_quotas = []

            # 最大/小顶点，用于判断是否有效的拐点
            self.MaxApex: Bar.Node | None = None
            self.MinApex: Bar.Node | None = None

        def max_apex(self, apex: Bar.Node, threshold: float):
            """到下一拐点前：最大的凸点"""
            if apex.node_ema - self.turn_ema >= threshold:
                if self.MaxApex is None or apex.node_ema > self.MaxApex.node_ema:
                    self.MaxApex = apex

        def min_apex(self, apex: Bar.Node, threshold: float):
            """到下一拐点前：最小的凹点"""
            if self.turn_ema - apex.node_ema >= threshold:
                if self.MinApex is None or apex.node_ema < self.MinApex.node_ema:
                    self.MinApex = apex

        def is_peak(self, node: Bar.Node, threshold):
            """是否波峰"""
            return self.MaxApex and self.MaxApex.node_ema - node.node_ema >= threshold

        def is_valley(self, node: Bar.Node, threshold):
            """是否波谷"""
            return self.MinApex and node.node_ema - self.MinApex.node_ema >= threshold


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
        self.node_idx = info[K.Bar.datetime]
        self.turn_idx = info[K.Turn.turn_idx]
        self.turn_val = info[K.Turn.turn_val]
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
            K.Log.turn_val: self.turn_val,
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
        def new_node(self, df: pd.DataFrame) -> Bar.Node:
            node = Bar.Node()
            node.node_idx = df.index[-1]
            node.node_ema = self.get(df, K.Ema.fast, -1)
            return node

        def new_turn(self, node: Bar.Node) -> Bar.Turn:
            turn = Bar.Turn()
            turn.turn_idx = node.node_idx
            turn.turn_ema = node.node_ema
            turn.turn_val = node.apex_val
            turn.add_quotas = copy.copy(self.Cfg.Rise.add_quotas)
            turn.sub_quotas = copy.copy(self.Cfg.Fall.sub_quotas)
            return turn

        def first(self, df: pd.DataFrame):
            node = self.new_node(df)
            turn = self.new_turn(node)

            df[K.Turn.turn_val] = 0
            df[K.Turn.turn_idx] = df.index[-1]
            df[K.Turn.turn_ema] = node.node_ema
            df[K.Turn.nodes] = pd.Series(dtype=object)
            df[K.Turn.turns] = pd.Series(dtype=object)
            self.set(df, K.Turn.nodes, -1, [node])
            self.set(df, K.Turn.turns, -1, [turn])

        def next(self, df: pd.DataFrame):
            nodes: list[Bar.Node] = self.get(df, K.Turn.nodes, -2)
            turns: list[Bar.Turn] = self.get(df, K.Turn.turns, -2)

            # 先预设值
            turn_idx = self.get(df, K.Turn.turn_idx, -2)
            turn_ema = self.get(df, K.Turn.turn_ema, -2)
            self.set(df, K.Turn.turn_val, -1, 0)
            self.set(df, K.Turn.turn_idx, -1, turn_idx)
            self.set(df, K.Turn.turn_ema, -1, turn_ema)
            self.set(df, K.Turn.nodes, -1, nodes)
            self.set(df, K.Turn.turns, -1, turns)

            # 跳过连续相等的节点
            node = self.new_node(df)
            if node.node_ema == nodes[-1].node_ema:
                return

            # 节点数据不得小于3条
            nodes.append(node)
            if len(nodes) < 3:
                return

            # 计算拐点
            self._calc_turn(df)

        def _calc_turn(self, df: pd.DataFrame):
            """计算拐点"""
            nodes: list[Bar.Node] = self.get(df, K.Turn.nodes, -1)
            turns: list[Bar.Turn] = self.get(df, K.Turn.turns, -1)

            # 最小振幅价格差
            base_price = self.Cfg.Bas.base_price
            least_wave = self.Cfg.Turn.least_wave
            threshold = round(base_price * least_wave, 4)

            # 计算顶点
            prev_node = nodes[-3]
            midd_node = nodes[-2]
            last_node = nodes[-1]
            last_turn = turns[-1]
            if prev_node.node_ema < midd_node.node_ema > last_node.node_ema:
                midd_node.apex_val = 1
                last_turn.max_apex(midd_node, threshold)
            elif prev_node.node_ema > midd_node.node_ema < last_node.node_ema:
                midd_node.apex_val = -1
                last_turn.min_apex(midd_node, threshold)

            # 起始拐点
            first_turn = turns[0]
            if first_turn.turn_val == 0:
                diff = round(first_turn.turn_ema - last_node.node_ema, 4)
                if abs(diff) > threshold:
                    first_turn.turn_val = 1 if diff > 0 else -1
                    self.set(df, K.Turn.turn_val, 0, first_turn.turn_val)
                return

            # 计算拐点
            apex = None
            if last_turn.is_peak(last_node, threshold):
                apex = last_turn.MaxApex
            elif last_turn.is_peak(midd_node, threshold):
                apex = last_turn.MinApex
            if apex is None:
                return
            new_turn = self.new_turn(apex)
            turns.append(new_turn)

            # 更新df
            idx = df.index.get_loc(new_turn.turn_idx)
            self.set(df, K.Turn.turn_val, idx, new_turn.turn_val)
            self.set(df, K.Turn.turn_idx, -1, new_turn.turn_idx)
            self.set(df, K.Turn.turn_ema, -1, new_turn.turn_ema)

        def _calc(self, price: float, period: int, prev_val: float):
            pass


class Broker:
    class B(ABC):
        def __init__(self, bus: StockBus):
            self.Bus: StockBus = bus

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
            cfg = self.Bus.Cfg
            pos = self.Bus.curr_pos()
            if pos[K.Pos.principal] > cfg.Bas.base_funds * cfg.Bas.cost_limit:
                return True
            if pos[K.Pos.principal] - pos[K.Pos.valuation] >= cfg.Bas.base_funds * cfg.Bas.loss_limit:
                return True
            return False

        def has_no_amount(self) -> bool:
            """没有可用持仓"""
            pos = self.Bus.curr_pos()
            return pos[K.Pos.avail_amount] <= self.remain_amount()

        def remain_amount(self) -> int:
            """获取保留的股票数量"""
            cfg = self.Bus.Cfg
            pos = self.Bus.curr_pos()
            if pos[K.Pos.total_amount] > pos[K.Pos.avail_amount]:
                return 0
            if pos[K.Pos.valuation] - pos[K.Pos.principal] > cfg.Bas.base_funds * cfg.Bas.gain_limit:
                return 0
            return 100

        def log(self, level: int, amount: float):
            """添加日志"""
            curr_min = self.Bus.curr_min()
            Log(curr_min).info(amount, level).add_to(self.Bus.Log)

    class Turn(B):
        def is_buy(self) -> bool:
            cfg = self.Bus.Cfg
            curr_day = self.Bus.curr_day()
            curr_min = self.Bus.curr_min()

            # 前5分钟不买入
            if curr_min[K.Bar.datetime].time() < time(9, 35, 0): return False
            # 超过亏损上限 or 超过本金上限
            if self.over_budget(): return False
            # 日线下跌
            if curr_day[K.Smma.fast] <= curr_day[K.Smma.slow]: return False
            # 分钟线下跌
            if curr_min[K.Ema.fast] <= curr_min[K.Ema.slow]: return False
            # 小于MACD值下限
            if curr_min[K.Macd.macd] < cfg.Rise.macd_limit: return False
            # 涨幅未达到阈值 or 重复操作
            level = self.__rise_level()
            turn = curr_min[K.Turn.turns][-1]
            if level == -1 or turn.add_quotas[level] == 0: return False
            # 决定买入
            return True

        def is_sell(self) -> bool:
            curr_min = self.Bus.curr_min()

            # 没有可用持仓
            if self.has_no_amount(): return False
            # 分钟线上涨
            if curr_min[K.Ema.fast] >= curr_min[K.Ema.slow]: return False
            # 跌幅未达到阈值 or 重复操作
            level = self.__fall_level()
            turn = curr_min[K.Turn.turns][-1]
            if level == -1 or turn.sub_quotas[level] == 0: return False
            # 决定卖出
            return True

        def do_buy(self, func: Callable):
            """执行买入"""
            cfg = self.Bus.Cfg
            curr_min = self.Bus.curr_min()
            turn = curr_min[K.Turn.turns][-1]
            lots = turn.add_quotas
            level = self.__rise_level()
            buy_amount = max(cfg.Bas.base_funds * lots[level], cfg.Bas.foot_funds) / cfg.Bas.base_price
            amount = round(buy_amount / 100) * 100

            # 执行买入
            func(self.Bus.symbol, amount)
            lots[level] = 0.0
            self.log(level, amount)

        def do_sell(self, func: Callable):
            """执行卖出"""
            cfg = self.Bus.Cfg
            curr_min = self.Bus.curr_min()
            turn = curr_min[K.Turn.turns][-1]
            lots = turn.fall_lots
            level = self.__fall_level()

            # 最小数量：根据基准本金
            min_qty = max(cfg.Bas.base_funds * lots[level], cfg.Bas.foot_funds) / cfg.Bas.base_price
            # 减仓数量：根据当日初始可用持仓
            cur_qty = cfg.Bas.base_amount * lots[level]
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
            curr_min = self.Bus.curr_min()
            diff_value = abs(curr_min[K.Ema.fast] - curr_min[K.Turn.turn_ema])
            diff_ratio = round(diff_value / self.Bus.Cfg.Bas.base_price, 4)
            for threshold in reversed(thresholds):
                if diff_ratio > threshold:
                    return thresholds.index(threshold)
            return -1


class Config:
    class _Bas:
        def __init__(self):
            self.cost_limit = 1.50  # 成本上限（比例）
            self.loss_limit = 0.15  # 亏损上限（比例）
            self.gain_limit = 0.05  # 盈利上限（比例）
            self.foot_funds = 3000  # 最小金额
            self.base_funds = 8000  # 基准资金
            self.base_price = 0.00  # 基准价格
            self.base_amount = 0.0  # 基准持仓

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


############################################################
class StockBus:
    def __init__(self, symbol: str, broker: str, config: Config):
        self.status = 0  # 状态：0-Initial、1-Started、2-Running
        self.symbol = symbol  # 股票代码
        self.broker = broker  # 经纪人
        self.Cfg = config  # 配置信息
        self.Pos = None  # 仓位数据
        self.Day = None  # 日线数据
        self.Min = None  # 分钟数据
        self.Log = None  # 交易日志

    def curr_pos(self) -> dict:
        return self.Pos.iloc[-1].to_dict()

    def curr_day(self) -> dict:
        return self.Day.iloc[-1].to_dict()

    def curr_min(self) -> dict:
        return self.Min.iloc[-1].to_dict()


class StockMarket:
    def __init__(self, bus: StockBus, bkr: Broker.B):
        self.Bus: StockBus = bus
        self.Bkr: Broker.B = bkr

    def prepare(self, pos, bars):
        # 日线数据
        cfg = self.Bus.Cfg
        day = Bar(bars[0]).new_df()
        Line.Smma(cfg).first(day)
        for bar in bars[1:]:
            Bar(bar).add_to(day)
            Line.Smma(cfg).next(day)
        self.Bus.Day = day

        # 仓位数据、基准价格、基准持仓
        cur_pos = Pos(pos, bars[-1])
        self.Bus.Log = Log.empty_df()
        self.Bus.Pos = cur_pos.new_df()
        self.Bus.Cfg.Bas.base_price = round(bars[-1].close, 4)
        self.Bus.Cfg.Bas.base_amount = cur_pos.avail_amount
        self.Bus.status = 1

    def running(self, pos, bar):
        cfg = self.Bus.Cfg
        Pos(pos, bar).add_to(self.Bus.Pos)
        if self.Bus.status == 2:
            Bar(bar).add_to(self.Bus.Min)
            Line.Ema(cfg).next(self.Bus.Min)
            Line.Macd(cfg).next(self.Bus.Min)
            Line.Turn(cfg).next(self.Bus.Min)
            return

        if self.Bus.status == 1:
            self.Bus.Min = Bar(bar).new_df()
            Line.Ema(cfg).first(self.Bus.Min)
            Line.Macd(cfg).first(self.Bus.Min)
            Line.Turn(cfg).first(self.Bus.Min)
            self.Bus.status = 2
            return

        if self.Bus.status == 0:
            self.prepare(pos, [bar])

    def trading(self, func: Callable):
        if self.Bkr.is_buy():
            self.Bkr.do_buy(func)
            return
        if self.Bkr.is_sell():
            self.Bkr.do_sell(func)


class StockManager:
    _markets: dict[str, StockMarket] = {}
    _brokers: dict[str, Type[Broker.B]] = {
        "Broker.Turn": Broker.Turn,
    }

    @staticmethod
    def market(bus: StockBus) -> StockMarket:
        mkt = StockManager._markets.get(bus.symbol)
        if mkt is not None:
            return mkt
        broker = StockManager._brokers[bus.broker](bus)
        market = StockMarket(bus, broker)
        StockManager._markets[bus.symbol] = market
        return market

    @staticmethod
    def buses() -> list[StockBus]:
        buses: list[StockBus] = []
        config = Config()
        symbols = ['159857.SZ']
        for symbol in symbols:
            bus = StockBus(symbol, 'Broker.Turn', config)
            buses.append(bus)
        return buses


############################################################
def initialize(context):
    """启动时执行一次"""
    buses = StockManager.buses()
    symbols = [bus.symbol for bus in buses]
    g.buses = buses
    g.symbols = symbols
    set_universe(symbols)
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    positions = get_positions()
    history = get_history(60, frequency='1d')
    for bus in g.buses:
        df = history.query(f'code in ["{bus.symbol}"]')
        pos = positions.get(bus.symbol)
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in df.iterrows()]
        StockManager.market(bus).prepare(pos, bars)


def handle_data(context, data):
    """每个单位周期执行一次"""
    positions = context.portfolio.positions
    for bus in g.buses:
        bar = data[bus.symbol]
        pos = positions.get(bus.symbol)
        StockManager.market(bus).running(pos, bar)
        StockManager.market(bus).trading(order)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
