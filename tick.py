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

import pandas as pd


class Var:
    class Base:
        def __init__(self):
            self.buy__time = ('09:40:00', '14:30:00')  # 交易时间（加仓）
            self.sell_time = ('09:40:00', '14:00:00')  # 交易时间（减仓）
            self.back_time = '14:55:00'  # 补仓时间

            self.basic_fund = 10000  # 交易基准金额
            self.start_fund = 3000  # 交易起步金额
            self.least_fund = 1000  # 交易最低金额
            self.cost_limit = 1.50  # 成本上限（比例）
            self.loss_limit = 0.15  # 亏损上限（比例）
            self.gain_limit = 0.15  # 盈利上限（比例）

    class Macd:
        def __init__(self):
            self.fast = 13  # 快线周期
            self.slow = 60  # 慢线周期
            self.sign = 5  # 信号线周期

    class Turn:
        def __init__(self):
            self.least_wave = 0.01  # 最小摆动（比例）
            self.sma_period = 5  # 所参照sma的周期

    class Trad:
        def __init__(self):
            self.rise_bounds = [0.010, 0.015]  # 加仓阈值（比例）
            self.rise_quotas = [0.300, 0.300]  # 加仓额度（比例）
            self.rise_macd = 0.003  # macd限制（比例）

            self.fall_bounds = [0.010, 0.015]  # 减仓阈值（比例）
            self.fall_quotas = [0.500, 0.500]  # 减仓额度（比例）
            self.fall_macd = -0.003  # macd限制（比例）

    class Config:
        def __init__(self):
            self.env = 'Live'  # 测试--Test、线上--Live
            self.base: Var.Base = Var.Base()
            self.macd: Var.Macd = Var.Macd()
            self.turn: Var.Turn = Var.Turn()
            self.trad: Var.Trad = Var.Trad()


class Bar:
    class Ema:
        _attrs = {5: 'ema05', 10: 'ema10', 20: 'ema20', 30: 'ema30', 60: 'ema60'}

        def __init__(self):
            self.ema05 = 0.0
            self.ema10 = 0.0
            self.ema20 = 0.0
            self.ema30 = 0.0
            self.ema60 = 0.0

        def get(self, period: int) -> float:
            return getattr(self, Bar.Ema._attrs.get(period, ''), 0.0)

    class Macd:
        def __init__(self):
            self.fast = 0.0
            self.slow = 0.0
            self.sign = 0.0
            self.diff = 0.0
            self.dea_ = 0.0
            self.macd = 0.0

    class Tick:
        def __init__(self, row: pd.Series):
            self.trade_status = 0.0  # 交易状态TRADE交易中
            self.hsTimeStamp = 0.0  # 时间戳，格式为YYYYMMDDHHMISS
            self.trade_mins = 0.0  # 交易时间，距离开盘已过多少分钟

            self.amount = 0.0  # 持仓量
            self.business_amount = 0.0  # 成交数量
            self.business_amount_in = 0.0  # 内盘成交量
            self.business_amount_out = 0.0  # 外盘成交量
            self.business_balance = 0.0  # 成交金额
            self.business_count = 0.0  # 成交笔数
            self.current_amount = 0.0  # 最近成交量(现手)

            self.up_px = 0.0  # 涨停价格
            self.down_px = 0.0  # 跌停价格
            self.pre_close_px = 0.0  # 昨收价
            self.open_px = 0.0  # 今开盘价
            self.last_px = 0.0  # 最新成交价
            self.high_px = 0.0  # 最高价
            self.low_px = 0.0  # 最低价
            self.avg_px = 0.0  # 均价

            self.turnover_ratio = 0.0  # 换手率
            self.entrust_diff = 0.0  # 委差
            self.entrust_rate = 0.0  # 委比
            self.vol_ratio = 0.0  # 量比

            self.offer_grp = {}  # 卖档位
            self.bid_grp = {}  # 买档位

            data = row.to_dict()
            self.pre_close_px = data.get('preclose_px')
            for key, value in data.items():
                setattr(self, key, value)

    class Node:
        def __init__(self, tick):
            self.ema: Bar.Ema = Bar.Ema()
            self.macd: Bar.Macd = Bar.Macd()
            self.tick: Bar.Tick = Bar.Tick(tick)
            self.time = self.tick.hsTimeStamp


class Line:
    class Ema:
        @staticmethod
        def first(mkt: Market):
            node = mkt.get(-1)
            price = node.tick.last_px
            node.ema.ema05 = price
            node.ema.ema10 = price
            node.ema.ema20 = price
            node.ema.ema30 = price
            node.ema.ema60 = price

        @staticmethod
        def next(mkt: Market):
            node = mkt.get(-1)
            price = node.tick.last_px
            pre_ema = mkt.get(-2).ema
            node.ema.ema05 = Line.Ema.ema(price, 5, pre_ema.ema05)
            node.ema.ema10 = Line.Ema.ema(price, 10, pre_ema.ema10)
            node.ema.ema20 = Line.Ema.ema(price, 20, pre_ema.ema20)
            node.ema.ema30 = Line.Ema.ema(price, 30, pre_ema.ema30)
            node.ema.ema60 = Line.Ema.ema(price, 60, pre_ema.ema60)

        @staticmethod
        def ema(price: float, period: int, prev_val: float):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 5)

    class Macd:
        @staticmethod
        def first(mkt: Market):
            node = mkt.get(-1)
            price = node.tick.last_px
            node.macd.fast = price
            node.macd.slow = price
            node.macd.diff = 0.0
            node.macd.dea_ = 0.0
            node.macd.macd = 0.0

        @staticmethod
        def next(mkt: Market):
            node = mkt.get(-1)
            price = node.tick.last_px
            prev_node = mkt.get(-2)

            fast = Line.Ema.ema(price, mkt.cfg.macd.fast, prev_node.macd.fast)
            slow = Line.Ema.ema(price, mkt.cfg.macd.slow, prev_node.macd.slow)
            diff = round(fast - slow, 5)
            dea_ = Line.Ema.ema(diff, mkt.cfg.macd.sign, prev_node.macd.dea_)
            macd = round((diff - dea_) * 2, 5)

            node.macd.fast = fast
            node.macd.slow = slow
            node.macd.diff = diff
            node.macd.dea_ = dea_
            node.macd.macd = macd


############################################################


class Market:
    def __init__(self, symbol: str, cfg: Var.Config):
        self.bars: list[Bar.Node] = []
        self.symbol = symbol
        self.cfg = cfg
        self.ctx = {}

    def clear(self):
        self.bars = []
        self.ctx = {}

    def get(self, idx: int) -> Bar.Node:
        return self.bars[idx]

    def add(self, node: Bar.Node):
        self.bars.append(node)


############################################################
class Env:
    markets: dict[str, Market] = {}
    symbols: list[str] = [
        '515450.SS',
    ]
    configs: dict[str, Type[Var.Config]] = {
        "515450.SS": Var.Config,
    }

    @staticmethod
    def market(symbol: str) -> Market:
        market = Env.markets.get(symbol)
        if market is None:
            config = Env.configs.get(symbol)()
            market = Env.markets.setdefault(symbol, Market(symbol, config))
        return market


############################################################
def initialize(context):
    """启动时执行一次"""
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    positions = get_positions(Env.symbols)
    for symbol in Env.symbols:
        pos = positions.get(symbol)
        if pos is None:


    codes = list(Env.symbols)


    set_universe(Env.symbols)
    pos = positions.get(symbol)

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


def tick_data(context, data):
    """每个tick执行一次"""
    pass


def handle_data(context, data):
    """每个单位周期执行一次"""
    pass


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
