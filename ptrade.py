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
        prev_idx = 'turn.prev_idx'
        prev_bar = 'turn.prev_bar'


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
            self.node_idx = 0
            self.node_ema = 0
            self.apex_val = 0

    class Turn:
        def __init__(self):
            self.node_idx = 0  # 节点时间
            self.node_ema = 0  # 节点EMA
            self.turn_idx = 0  # 拐点时间
            self.turn_ema = 0  # 拐点EMA
            self.turn_val = 0  # 拐点标记
            self.apex_min = None  # 拐点的最小顶点
            self.apex_max = None  # 拐点的最大顶点

        def be_turn(self, apex: Bar.Node):
            self.turn_idx = apex.node_idx
            self.turn_ema = apex.node_ema
            self.turn_val = apex.apex_val
            self.apex_min = None
            self.apex_max = None
            self.be_apex(apex.apex_val)

        def be_apex(self, apex_val=0):
            apex = Bar.Node()
            apex.node_idx = self.node_idx
            apex.node_ema = self.node_ema
            apex.apex_val = apex_val
            if self.apex_min is None or self.apex_min.apex_ema > self.node_ema:
                self.apex_min = apex
            if self.apex_max is None or self.apex_max.apex_ema < self.node_ema:
                self.apex_max = apex
            return self

        def is_peak(self, threshold):
            if self.apex_max.apex_ema - self.node_ema < threshold:
                return False
            if self.apex_max.apex_ema - self.turn_ema < threshold:
                return False
            return True

        def is_valley(self, threshold):
            if self.node_ema - self.apex_min.apex_ema < threshold:
                return False
            if self.turn_ema - self.apex_min.apex_ema < threshold:
                return False
            return True


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
        self.turn_idx = info[K.Turn.prev_idx]
        self.turn_val = info[K.Turn.prev_ema]
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
            df[col].iloc[idx] = val

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
        def first(self, df: pd.DataFrame):
            turn = Bar.Turn()
            turn.node_idx = df.index[-1]
            turn.node_ema = self.get(df, K.Ema.fast, -1)
            turn.turn_idx = turn.node_idx
            turn.turn_ema = turn.node_ema
            turn.turn_val = 0
            turn.be_apex()

            df[K.Turn.turn_val] = 0
            df[K.Turn.prev_idx] = turn.node_idx
            df[K.Turn.prev_bar] = pd.Series(dtype=object)
            self.set(df, K.Turn.prev_bar, -1, turn)

        def next(self, df: pd.DataFrame):
            prev_turn = self.get(df, K.Turn.prev_bar, -2)
            next_turn = copy.copy(prev_turn)
            next_turn.node_idx = df.index[-1]
            next_turn.node_ema = self.get(df, K.Ema.fast, -1)

            # 先预设值
            self.set(df, K.Turn.turn_val, -1, 0)
            self.set(df, K.Turn.prev_idx, -1, next_turn.turn_idx)
            self.set(df, K.Turn.prev_bar, -1, next_turn)

            # 计算顶点、拐点
            self._calc_apex(df)
            self._calc_turn(df)

        def _calc_apex(self, df: pd.DataFrame):
            """计算顶点"""
            if len(df) < 3:
                return

            # 跳过连续相等的节点
            node_1st = self.get(df, K.Turn.prev_bar, -1)
            node_2nd = self.get(df, K.Turn.prev_bar, -2)
            if node_1st.node_ema == node_2nd.node_ema:
                return

            # 倒数第三个节点
            node_3rd = None
            for i in range(-3, -len(df) - 1, -1):
                temp = self.get(df, K.Turn.prev_bar, i)
                if temp.node_ema != node_2nd.node_ema:
                    node_3rd = temp
                    break
            if node_3rd is None:
                return

            # 判断顶点
            is_peak = node_1st.node_ema < node_2nd.node_ema > node_3rd.node_ema
            is_valley = node_1st.node_ema > node_2nd.node_ema < node_3rd.node_ema
            apex_val = 1 if is_peak else -1 if is_valley else 0
            if is_peak or is_valley:
                node_2nd.be_apex(apex_val)
                node_3rd.apex_min = node_2nd.apex_min
                node_3rd.apex_max = node_2nd.apex_max

        def _calc_turn(self, df: pd.DataFrame):
            """计算拐点"""
            turn = self.get(df, K.Turn.prev_bar, -1)

            # 最小振幅价格差
            base_price = self.Cfg.Bas.base_price
            least_wave = self.Cfg.Turn.least_wave
            threshold = round(base_price * least_wave, 4)

            # 首个拐点
            if turn.turn_val == 0:
                turn_val = 0
                if turn.node_ema - turn.turn_ema >= threshold:
                    turn_val = -1
                if turn.turn_ema - turn.node_ema >= threshold:
                    turn_val = 1
                if turn_val != 0:
                    self.set(df, K.Turn.turn_val, 0, turn_val)
                    turn.turn_val = turn_val
                return

            # 判断波峰波谷
            apex = None
            if turn.is_peak(threshold):
                apex = turn.apex_max
            if turn.is_valley(threshold):
                apex = turn.apex_min
            if apex is None:
                return

            # 更新拐点
            turn.be_turn(apex)
            turn_idx = df.index.get_loc(apex.node_idx)
            self.set(df, K.Turn.turn_val, turn_idx, apex.apex_val)
            self.set(df, K.Turn.prev_idx, -1, apex.apex_idx)

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

        def over_limit(self) -> bool:
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

        def log(self, amount: float, level: int):
            """添加日志"""
            curr_min = self.Bus.curr_min()
            Log(curr_min).info(amount, level).add_to(self.Bus.Log)

    class Turn(B):
        def is_buy(self) -> bool:
            cfg = self.Bus.Cfg
            pos = self.Bus.curr_pos()
            curr_day = self.Bus.curr_day()
            curr_min = self.Bus.curr_min()

            if curr_min[K.Bar.datetime].time() < time(9, 35, 0):
                # 前5分钟不买入
                return False
            if self.over_limit():
                # 超过亏损上限 or 超过本金上限
                return False
            if curr_day[K.Smma.fast] <= curr_day[K.Smma.slow]:
                # 日线下跌
                return False
            if curr_min[K.Ema.fast] <= curr_min[K.Ema.slow]:
                # 分钟线下跌
                return False
            if curr_min[K.Macd.macd] < cfg.Rise.macd_limit:
                # 小于MACD值下限
                return False
            # 涨幅未达到阈值 or 重复操作
            level = self.__rise_level()
            if level == -1 or turn.rise_lots[level] == 0:
                return False
            # 决定买入
            return True

        def is_sell(self) -> bool:
            cfg = self.Bus.Cfg
            pos = self.Bus.curr_pos()
            curr_min = self.Bus.curr_min()
            if self.has_no_amount():
                # 没有可用持仓
                return False
            if curr_min[K.Ema.fast] >= curr_min[K.Ema.slow]:
                # 分钟线上涨
                return False
            # 跌幅未达到阈值 or 重复操作
            level = self.__fall_level()
            if level == -1 or turn.fall_lots[level] == 0:
                return False
            # 决定卖出
            return True

        def do_buy(self, func: Callable):
            """执行买入"""
            cfg = self.Bus.Cfg
            _, _, turn = self.Bus.bars()
            level = self.__rise_level()
            lots = turn.rise_lots
            buy_amount = cfg.Bas.base_funds / self.Bus.MinSet.base_price * lots[level]
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
            min_qty = self.Bus.Cfg.Pos.base_principal / today.base_price * lots[level]
            # 减仓数量：根据当日初始可用持仓
            cur_qty = today.base_amount * lots[level]
            # 避免低仓位时，还分多次减仓
            sell_qty = max(min_qty, cur_qty)
            # 不得超过当前可用持仓
            sell_amount = min(sell_qty, self.Bus.Pos.avail_amount)
            # 调整到100的倍数，并留下底仓
            amount = round(sell_amount / 100) * 100 - self.Bus.Pos.remain_amount()

            # 执行卖出
            func(self.Bus.symbol, -amount)
            lots[level] = 0.0

        def __rise_level(self):
            """当前上涨等级"""
            curr_min = self.Bus.curr_min()
            mark = node.Mark
            if mark.rise_level == -2:
                mark.rise_level = self.__calc_level(self.Bus.Cfg.Rise.thresholds)
            return mark.rise_level

        def __fall_level(self):
            """当前下跌等级"""
            _, node, _ = self.Bus.bars()
            mark = node.Mark
            if mark.fall_level == -2:
                mark.fall_level = self.__calc_level(self.Bus.Cfg.Fall.thresholds)
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


class Config:
    class _Bas:
        def __init__(self):
            self.base_funds = 8000  # 基础资金
            self.cost_limit = 1.50  # 成本上限（比例）
            self.loss_limit = 0.15  # 亏损上限（比例）
            self.gain_limit = 0.05  # 盈利上限（比例）
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
