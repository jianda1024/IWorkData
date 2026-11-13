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

import copy
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd


class Var:
    base_fund = 3000  # 交易基础金额（元）


class Bin:
    class Act:
        def __init__(self):
            self.has_sell = False
            self.sell_price = 0.0
            self.sell_amount = 0.0

            self.has_buy = False
            self.buy_price = 0.0
            self.buy_amount = 0.0

    class Pos:
        def __init__(self, pos):
            self.symbol: str = getattr(pos, 'sid', '')  # 股票代码
            self.total_amount = getattr(pos, 'amount', 0.0)  # 总持仓数量
            self.avail_amount = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
            self.curr_price = getattr(pos, 'last_sale_price', 0.0)  # 最新价格
            self.cost_price = getattr(pos, 'cost_basis', 0.0)  # 成本价格
            self.total_value = round(self.total_amount * self.curr_price, 3)  # 总市值
            self.avail_value = round(self.avail_amount * self.curr_price, 3)  # 可用市值
            self.principal = round(self.total_amount * self.cost_price, 3)  # 本金

    class Bar:
        def __init__(self, bar=None):
            if bar is None: return
            self.datetime = bar.datetime
            self.volume: float = round(bar.volume, 3)  # 交易量
            self.money: float = round(bar._base.money, 3)  # 交易金额
            self.price: float = round(bar.price, 3)  # 最新价
            self.close: float = round(bar.close, 3)  # 收盘价
            self.open: float = round(bar.open, 3)  # 开盘价
            self.high: float = round(bar.high, 3)  # 最高价
            self.low: float = round(bar.low, 3)  # 最低价

        def ohlc(self, bar):
            if bar is None:
                return self
            self.volume += bar.volume
            self.money += bar.money
            self.open = bar.open
            self.high = max(self.high, bar.high)
            self.low = min(self.low, bar.low)
            return self

    class Node:
        def __init__(self, bar: Bin.Bar):
            self.tag = 0
            self.bar = bar
            self.ema = None
            self.macd = None

        def mark(self, tag):
            self.tag = tag
            return self


class Bus:
    def __init__(self):
        self.data: list[Bin.Node] = []

    def __len__(self):
        return len(self.data)

    def add(self, node: Bin.Node):
        self.data.append(node)

    def get(self, idx: int) -> Bin.Node:
        return self.data[idx]

    def extend(self, nodes: list[Bin.Node]):
        self.data.extend(nodes)

    def rollback(self):
        node = self.data.pop()
        if node.tag >= 0:
            self.data.append(node)


############################################################
class Line:
    class Idx:
        class Sma:
            @staticmethod
            def input(row: pd.Series, sma: SimpleNamespace):
                sma.ma05 = row["ma05"]
                sma.ma10 = row["ma10"]
                sma.ma20 = row["ma20"]

            @staticmethod
            def calc_his(df: pd.DataFrame):
                px_close = df["close"]
                df["ma05"] = px_close.rolling(5).mean()
                df["ma10"] = px_close.rolling(10).mean()
                df["ma20"] = px_close.rolling(20).mean()

            @staticmethod
            def calc_live(bus):
                node = bus.get(-1)
                price = node.bar.close
                pre_sma = bus.get(-2).sma
                cur_sma = node.sma = SimpleNamespace()
                cur_sma.ma05 = Line.Util.ema(pre_sma.ma05, price, 5)
                cur_sma.ma10 = Line.Util.ema(pre_sma.ma10, price, 10)
                cur_sma.ma20 = Line.Util.ema(pre_sma.ma20, price, 20)

        class Agg:
            @staticmethod
            def input(row: pd.Series, agg: SimpleNamespace):
                agg.atr = row["agg_atr"]  # 平均真实波幅
                agg.mad = row["agg_mad"]  # 移动累计偏离
                agg.hst = row["agg_hst"]  # 赫斯特指数：H<0.45强回归，0.45≤H≤0.55混沌，H>0.55趋势
                agg.trd = row["agg_trd"]  # MA20趋势：-1向下，0水平，1向上

            @staticmethod
            def calc_his(df: pd.DataFrame):
                px_close, px_high, px_low = df["close"], df["high"], df["low"]
                df["agg_atr"] = df['agg_mad'] = df['agg_hst'] = df["agg_trd"] = 0.0
                index = df.index[-1]

                # ATR
                tr1 = px_high - px_low
                tr2 = (px_high - px_close.shift(1)).abs()
                tr3 = (px_low - px_close.shift(1)).abs()
                atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df.loc[index, 'agg_atr'] = atr.tail(14).mean()

                # MAD
                bias = px_close - df["ma20"]
                df.loc[index, 'agg_mad'] = bias.tail(5).sum()

                # 赫斯特指数
                df.loc[index, 'agg_hst'] = Line.Util.hurst(df)

                # MA20趋势
                window, thresh = 5, 0.005
                sma20 = df['ma20'].dropna().iloc[-window:]
                slope = np.polyfit(np.arange(window), sma20, 1)[0]
                df.loc[index, 'agg_trd'] = 1 if slope > thresh else (-1 if slope < -thresh else 0)

            @staticmethod
            def calc_live(bus):
                node = bus.get(-1)
                price = node.bar.close
                pre_agg = bus.get(-2).agg
                node.agg = copy.copy(pre_agg)
                node.agg.mad = round(pre_agg.mad + price - node.sma.ma20, 4)

        class Obv:
            @staticmethod
            def input(row: pd.Series, obv: SimpleNamespace):
                obv.vol = row["obv_vol"]  # 能量潮
                obv.avg = row["obv_avg"]  # 能量潮均值

            @staticmethod
            def calc_his(df: pd.DataFrame):
                px_volume, px_close = df["volume"], df["close"]
                px_diff = px_close.diff()
                vol_dir = np.where(px_diff > 0, px_volume, np.where(px_diff < 0, -px_volume, 0))
                df['obv_vol'] = vol_dir.cumsum()
                df['obv_avg'] = df['obv_vol'].rolling(window=20).mean()

            @staticmethod
            def calc_live(bus):
                node = bus.get(-1)
                price = node.bar.close
                volume = node.bar.volume
                pre_node = bus.get(-2)
                pre_price = pre_node.bar.close

                cur_obv = node.obv = SimpleNamespace()
                obv_vol = volume if price > pre_price else (-volume if price < pre_price else 0)
                cur_obv.vol = round(pre_node.obv.vol + obv_vol, 0)
                cur_obv.avg = round(pre_node.obv.avg, 0)

        class Vol:
            @staticmethod
            def calc(bus):
                node = bus.get(-1)
                if len(bus) == 1:
                    pre_vol = SimpleNamespace(money=0, volume=0)
                else:
                    pre_vol = bus.get(-2).vol
                bar = node.bar
                vol = node.vol = copy.copy(pre_vol)
                vol.volume += bar.volume  # 成交量总和
                vol.money += bar.money  # 成交额总和
                not_zero = vol.money != 0 and vol.volume != 0
                vol.price = round(vol.money / vol.volume, 4) if not_zero else 0.0  # 成交量加权平均价格

        class Macd:
            @staticmethod
            def input(row: pd.Series, macd: SimpleNamespace):
                macd.fast = row["ma12"]
                macd.slow = row["ma26"]
                macd.dif_ = row["dif_"]
                macd.dea_ = row["dea_"]
                macd.macd = row["macd"]

            @staticmethod
            def calc_his(df: pd.DataFrame):
                px_close = df["close"]
                df["ma12"] = px_close.ewm(span=12, adjust=False).mean()
                df["ma26"] = px_close.ewm(span=26, adjust=False).mean()
                df["dif_"] = df["ma12"] - df["ma26"]
                df["dea_"] = df["dif_"].ewm(span=9, adjust=False).mean()
                df["macd"] = 2 * (df["dif_"] - df["dea_"])

            @staticmethod
            def calc_live(bus):
                node = bus.get(-1)
                price = node.bar.close
                pre_macd = bus.get(-2).macd
                cur_macd = node.macd = SimpleNamespace()
                cur_macd.fast = Line.Util.ema(pre_macd.fast, price, 12)
                cur_macd.slow = Line.Util.ema(pre_macd.slow, price, 26)
                cur_macd.dif_ = round(cur_macd.fast - cur_macd.slow, 4)
                cur_macd.dea_ = Line.Util.ema(pre_macd.dea_, cur_macd.dif_, 9)
                cur_macd.macd = round(2 * (cur_macd.dif_ - cur_macd.dea_), 4)

    class Day:
        keys = ["volume", "price", "close", "open", "high", "low"]

        @staticmethod
        def calc_live(bus: Bus):
            Line.Idx.Sma.calc_live(bus)
            Line.Idx.Agg.calc_live(bus)
            Line.Idx.Obv.calc_live(bus)
            Line.Idx.Macd.calc_live(bus)

        @staticmethod
        def calc_his(data: pd.DataFrame) -> pd.DataFrame:
            df = data[Line.Day.keys].copy()
            Line.Idx.Sma.calc_his(df)
            Line.Idx.Agg.calc_his(df)
            Line.Idx.Obv.calc_his(df)
            Line.Idx.Macd.calc_his(df)
            df = df.round(4)
            return df

        @staticmethod
        def yesterday(df: pd.DataFrame) -> Bin.Node:
            bar = Bin.Bar()
            row = df.iloc[-1]
            bar.datetime = row.name
            for key in Line.Day.keys:
                setattr(bar, key, row[key])

            node = Bin.Node(bar)
            node.agg = SimpleNamespace()
            node.obv = SimpleNamespace()
            node.sma = SimpleNamespace()
            node.macd = SimpleNamespace()
            node.boll = SimpleNamespace()
            Line.Idx.Agg.input(row, node.agg)
            Line.Idx.Obv.input(row, node.obv)
            Line.Idx.Sma.input(row, node.sma)
            Line.Idx.Macd.input(row, node.macd)
            return node

    class Util:
        @staticmethod
        def ema(pre_val: float, price: float, period: int):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * pre_val
            return round(value, 4)

        @staticmethod
        def hurst(df: pd.DataFrame):
            window = 120
            scales = [10, 15, 20, 25, 30]
            prices = df['close'].values
            if len(prices) < window + 1: return 0.5
            yields = np.diff(np.log(prices))[-window:]
            cum_ts = np.cumsum(yields - np.mean(yields))

            fluc = []
            for s in scales:
                ssq = 0
                num = len(cum_ts) // s
                if num < 4: continue
                for i in range(num):
                    seg = cum_ts[i * s:(i + 1) * s]
                    x = np.arange(s)
                    k, b = np.polyfit(x, seg, 1)
                    ssq += np.sum((seg - (k * x + b)) ** 2)
                fluc.append(np.sqrt(ssq / (num * s)))
            hst = np.polyfit(np.log(scales), np.log(fluc), 1)[0]
            return round(float(hst), 3)


class Trader:
    def __init__(self, buy: Callable, sell: Callable):
        self.sell = sell
        self.buy = buy

    def trading(self, market: Market):
        ctx = market.ctx
        if ctx.is_today_finish: return
        if ctx.pos.total_amount == 0.0: return
        if ctx.now.strftime("%H:%M:%S") < "10:00:00": return
        if ctx.pos.total_value < 500 and ctx.should_keep_pos:
            self._do_buy(ctx)
        if ctx.pos.avail_value > 500 and not ctx.should_keep_pos:
            self._do_sell(ctx)

    def _do_buy(self, ctx: Ctx):
        symbol, curr_price = ctx.pos.symbol, ctx.pos.curr_price
        buy_amount = round(Var.base_fund / curr_price / 100) * 100
        self.buy(symbol, buy_amount, limit_price=curr_price + 0.003)
        ctx.act.has_buy = True
        ctx.act.buy_price = curr_price
        ctx.act.buy_amount = buy_amount
        ctx.is_today_finish = True

    def _do_sell(self, ctx: Ctx):
        pos = ctx.pos
        base_amount = round(Var.base_fund * 1.5 / pos.curr_price / 100) * 100
        able_amount = min(pos.avail_amount, base_amount)
        sell_amount = able_amount - (0 if able_amount < pos.total_amount else 100)
        self.sell(pos.symbol, -sell_amount, limit_price=pos.curr_price - 0.003)
        ctx.act.has_sell = True
        ctx.act.sell_price = pos.curr_price
        ctx.act.sell_amount = sell_amount
        ctx.is_today_finish = True


############################################################
class Ctx:
    def __init__(self):
        self.now = None  # 当前时间
        self.pos = None  # 当前持仓
        self.bar = None  # 当前bar(日)
        self.day = None  # 当前节点(日)
        self.fen = None  # 当前节点(分)
        self.act = Bin.Act()  # 操作记录
        self.should_keep_pos = False  # 是否应该持有仓位
        self.is_today_finish = False  # 是否今天交易完成

    def evaluate(self):
        """评估市场行情"""
        vwap = self.fen.vol.price
        agg, obv = self.day.agg, self.day.obv
        sma, macd = self.day.sma, self.day.macd

        # 任意条件满足 → 不持仓
        conditions = [
            vwap < sma.ma20 - 2 * agg.atr,  # 有效破位
            sma.ma05 < sma.ma10 < sma.ma20,  # MA5、MA10、MA20并排空头
            agg.trd == -1 and vwap < sma.ma10,  # MA20 趋势向下 + VWAP小于MA10
            macd.macd / sma.ma20 * 100 < -0.6,  # MACD深度空头
            obv.vol < obv.avg and vwap < sma.ma10, # 量能退潮 + 跌破MA10
            agg.hst > 0.55 and agg.mad < -1.5 * agg.atr,  # 负偏离过大
            agg.hst > 0.55 and vwap < sma.ma20,  # 赫斯特强趋势 + 跌破MA10
        ]
        self.should_keep_pos = not any(conditions)


class Market:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.dayBus = Bus()  # 日线数据
        self.fenBus = Bus()  # 分钟数据
        self.ctx = Ctx()  # 上下文数据
        self.df = None

    def prepare(self, day_his: pd.DataFrame):
        self.df = Line.Day.calc_his(day_his)
        node = Line.Day.yesterday(self.df)
        self.dayBus.add(node)
        return self

    def running(self, pos, bar):
        # 日线数据
        self.dayBus.rollback()
        day_bar = Bin.Bar(bar).ohlc(self.ctx.bar)
        day_node = Bin.Node(day_bar).mark(-1)
        self.dayBus.add(day_node)
        Line.Day.calc_live(self.dayBus)

        # 分钟数据
        fen_node = Bin.Node(Bin.Bar(bar))
        self.fenBus.add(fen_node)
        Line.Idx.Vol.calc(self.fenBus)

        # 上下文数据
        self.ctx.now = bar.datetime
        self.ctx.pos = Bin.Pos(pos)
        self.ctx.bar = day_bar
        self.ctx.day = day_node
        self.ctx.fen = fen_node
        self.ctx.evaluate()


############################################################
class Env:
    trader: Trader = None
    markets: dict[str, Market] = {}

    @staticmethod
    def launch(buy: Callable, sell: Callable):
        Env.trader = Trader(buy=buy, sell=sell)


def initialize(context):
    """启动时执行一次"""
    Env.launch(order, order)
    if not is_trade():
        set_commission(commission_ratio=0.00005, min_commission=0.5, type="ETF")
        # pos = {'sid': "588790.SS", 'amount': "100", 'enable_amount': "100", 'cost_basis': "1.0"}
        pos = {'sid': "159995.SZ", 'amount': "100", 'enable_amount': "100", 'cost_basis': "1.0"}
        set_yesterday_position([pos])


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    positions = context.portfolio.positions
    pos_codes = list(positions.keys())
    set_universe(pos_codes)
    day_history = get_history(240, frequency='1d', security_list=pos_codes, fq='dypre')
    for code in pos_codes:
        day_his = day_history.query(f'code in ["{code}"]')
        Env.markets[code] = Market(code).prepare(day_his)


def handle_data(context, data):
    """每个单位周期执行一次"""
    trader = Env.trader
    positions = context.portfolio.positions
    for symbol, market in Env.markets.items():
        bar = data[symbol]
        pos = positions.get(symbol)
        market.running(pos, bar)
        trader.trading(market)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    Env.markets.clear()
    pass
