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
from typing import Callable, Any

import numpy as np
import pandas as pd


class Var:
    base_fund = 5000  # 交易基础金额（元）
    ma20_flat_thresh = 0.005  # MA20斜率阈值

    fen = {
        "macd_fast": 13,
        "macd_slow": 60,
        "macd_sign": 5,
    }


class Bin:
    class Act:
        def __init__(self):
            self.has_buy = False  # 是否已经买入
            self.has_sell = False  # 是否已经卖出
            self.buy_price = 0.0  # 买入时的价格
            self.sell_price = 0.0  # 卖出时的价格
            self.buy_amount = 0.0  # 买入时的数量
            self.sell_amount = 0.0  # 卖出时的数量
            self.buy_back_price = 0.0  # 反T接回价格

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
            self.money: float = round(bar.money, 3)  # 交易金额
            self.price: float = round(bar.price, 3)  # 最新价
            self.close: float = round(bar.close, 3)  # 收盘价
            self.open: float = round(bar.open, 3)  # 开盘价
            self.high: float = round(bar.high, 3)  # 最高价
            self.low: float = round(bar.low, 3)  # 最低价

        def agg(self, bar: Bin.Bar):
            if bar is None:
                return self
            self.volume += bar.volume
            self.money += bar.money
            self.open = bar.open
            self.high = max(self.high, bar.high)
            self.low = min(self.low, bar.low)
            return self


class Box:
    class Atr:
        def __init__(self):
            self.cur_tr: float = 0.0  # 瞬时真实波幅
            self.avg_tr: float = 0.0  # 平均真实波幅

    class Ema:
        def __init__(self):
            self.ma05: float = 0.0
            self.ma10: float = 0.0
            self.ma20: float = 0.0

    class Obv:
        def __init__(self):
            self.obv_vol: float = 0.0  # 能量潮
            self.sum_vol: float = 0.0  # 成交量合计
            self.obv_ratio: float = 0.0  # 能量潮比值

    class Orb:
        def __init__(self):
            self.orb_low: float = 0.0  # 开盘区间内的最低价
            self.orb_high: float = 0.0  # 开盘区间内的最高价

    class Vwa:
        def __init__(self):
            self.sum_volume: float = 0.0  # 成交量总和
            self.sum_money: float = 0.0  # 成交额总和
            self.avg_price: float = 0.0  # 成交量加权平均价格

            self.sum_bar: int = 0  # bar总和
            self.sum_yan: int = 0  # 阳线总和
            self.sum_yin: int = 0  # 阴线总和
            self.sum_yan_vol: float = 0.0  # 阳线成交量总和
            self.sum_yin_vol: float = 0.0  # 阴线成交量总和

    class Macd:
        def __init__(self):
            self.fast: float = 0.0
            self.slow: float = 0.0
            self.dif_: float = 0.0
            self.dea_: float = 0.0
            self.macd: float = 0.0

    class Boll:
        def __init__(self):
            self.atr: float = 0.0  # 平均真实波幅
            self.ma20: float = 0.0  # 中轨
            self.ma20_slope: float = 0.0  # 中轨斜率
            self.ma20_trend: float = 0.0  # 中轨趋势：-1向下，0水平，1向上
            self.upper: float = 0.0  # 上轨
            self.lower: float = 0.0  # 下轨
            self.band_wid: float = 0.0  # 布林带宽度
            self.band_avg: float = 0.0  # 布林带均宽
            self.band_tag: float = 0.0  # 布林带状态：-1收敛，0正常，1发散

    class Node:
        def __init__(self, bar: Bin.Bar):
            self.bar = bar
            self.ema = None
            self.macd = None
            self.flag: int = 0

        def mark(self, flag):
            self.flag = flag
            return self


class Line:
    class Atr:
        @staticmethod
        def calc(bus):
            node = bus.last()
            node.atr = atr = Box.Atr()
            high = node.bar.high
            low = node.bar.low
            if len(bus) == 1:
                atr.cur_tr = high - low
                atr.avg_tr = high - low
            else:
                prev_node = bus.get(-2)
                prev_close = prev_node.bar.close
                atr.cur_tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                atr.avg_tr = Line.Ema.ema(prev_node.atr.avg_tr, atr.cur_tr, 14)

    class Ema:
        @staticmethod
        def calc(bus: Bus):
            node = bus.last()
            price = node.bar.close
            node.ema = Box.Ema()
            if len(bus) == 1:
                Line.Ema.first(node, price)
            else:
                Line.Ema.next(bus, node, price)

        @staticmethod
        def first(node: Box.Node, price: float):
            curr_ema = node.ema
            curr_ema.ma05 = price
            curr_ema.ma10 = price
            curr_ema.ma20 = price

        @staticmethod
        def next(bus: Bus, node: Box.Node, price: float):
            curr_ema = node.ema
            prev_ema = bus.get(-2).ema
            curr_ema.ma05 = Line.Ema.ema(prev_ema.ma05, price, 5)
            curr_ema.ma10 = Line.Ema.ema(prev_ema.ma10, price, 10)
            curr_ema.ma20 = Line.Ema.ema(prev_ema.ma20, price, 20)

        @staticmethod
        def ema(prev_val: float, price: float, period: int):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

    class Obv:
        @staticmethod
        def calc(bus):
            if len(bus) == 1: return
            node = bus.last()
            close = node.bar.close
            volume = node.bar.volume
            pre_node = bus.get(-2)
            pre_close = pre_node.bar.close
            obv_vol = volume if close > pre_close else (-volume if close < pre_close else 0)

            node.obv = Box.Obv()
            node.obv.obv_vol = pre_node.obv.obv_vol + obv_vol
            node.obv.sum_vol = pre_node.obv.sum_vol + volume
            node.obv.obv_ratio = round(node.obv.obv_vol / node.obv.sum_vol, 3)

    class Orb:
        @staticmethod
        def calc(bus):
            node = bus.last()
            price = node.bar.close
            node.orb = orb = Box.Orb()
            if len(bus) == 1:
                orb.orb_low = price
                orb.orb_high = price
            else:
                prev_orb = bus.get(-2).orb
                orb.orb_low = min(prev_orb.orb_low, price)
                orb.orb_high = max(prev_orb.orb_high, price)

    class Vwa:
        @staticmethod
        def calc(bus):
            node = bus.last()
            curr_bar = node.bar
            prev_vwa = bus.get(-2).vwa if len(bus) > 1 else Box.Vwa()
            node.vwa = vwa = copy.copy(prev_vwa)

            vwa.sum_volume += curr_bar.volume
            vwa.sum_money += curr_bar.money
            vwa.avg_price = round(vwa.sum_money / vwa.sum_volume / 100, 4)
            vwa.sum_bar += 1
            if curr_bar.close > curr_bar.open:
                vwa.sum_yan += 1
                vwa.sum_yan_vol += curr_bar.volume
            if curr_bar.close < curr_bar.open:
                vwa.sum_yin += 1
                vwa.sum_yin_vol += curr_bar.volume

    class Macd:
        @staticmethod
        def calc(bus: Bus):
            node = bus.last()
            price = node.bar.close
            node.macd = Box.Macd()
            if len(bus) == 1:
                Line.Macd._first(node, price)
            else:
                Line.Macd._next(bus, node, price)

        @staticmethod
        def _first(node: Box.Node, price: float):
            curr_macd = node.macd
            curr_macd.fast = price
            curr_macd.slow = price
            curr_macd.dif_ = 0.0
            curr_macd.dea_ = 0.0
            curr_macd.macd = 0.0

        @staticmethod
        def _next(bus: Bus, node: Box.Node, price: float):
            curr_macd = node.macd
            prev_node = bus.get(-2)
            prev_macd = prev_node.macd
            prev_ma10 = prev_node.ema.ma10
            curr_macd.fast = Line.Ema.ema(prev_macd.fast, price, bus.conf.get('macd_fast', 12))
            curr_macd.slow = Line.Ema.ema(prev_macd.slow, price, bus.conf.get('macd_slow', 26))
            curr_macd.dif_ = round(curr_macd.fast - curr_macd.slow, 4)
            curr_macd.dea_ = Line.Ema.ema(prev_macd.dea_, curr_macd.dif_, bus.conf.get('macd_sign', 9))
            curr_macd.macd = round((curr_macd.dif_ - curr_macd.dea_) * 2 / prev_ma10 * 100, 4)

    class Boll:
        @staticmethod
        def calc(bus):
            if len(bus) == 1: return
            node = bus.last()
            node.boll = copy.copy(bus.get(-2).boll)


############################################################
class Step:
    class Start:
        @staticmethod
        def eval(ctx: Ctx):
            if Step.Start.is_sell(ctx):
                return "Sell", "BuyBack"
            if Step.Start.is_buy(ctx):
                return "Buy", "End"
            return "Wait", "Wait"

        @staticmethod
        def is_sell(ctx: Ctx) -> bool:
            if ctx.curr_time > "10:00:00": return False  # 超过时间点
            if ctx.pos.avail_value < 500: return False  # 没有可用持仓

            vwa = ctx.fen.vwa
            ema = ctx.fen.ema
            macd = ctx.fen.macd
            avg_price = round(vwa.avg_price - 1.5 * ctx.fen.atr.avg_tr, 3)
            if ctx.pos.curr_price >= avg_price: return False  # 价格未跌破均价线
            if ema.ma05 >= ema.ma10: return False
            if ema.ma10 >= ema.ma20: return False
            if macd.macd >= 0: return False
            if macd.dif_ >= 0: return False

            orb = ctx.orb
            if ema.ma05 < orb.orb_low: return True  # 跌破ORL
            if vwa.sum_yin / vwa.sum_bar > 0.8: return True  # 阴线数量
            if vwa.sum_yin_vol / vwa.sum_volume > 0.8: return True  # 阴线成交量

            # 较开盘价，跌破阈值
            if ctx.curr_time < "09:40:00":
                if ctx.rela_price_pct < -0.6:
                    return True
            # 其他
            return False

        @staticmethod
        def is_buy(ctx: Ctx) -> bool:
            vwa = ctx.fen.vwa
            ema = ctx.fen.ema
            macd = ctx.fen.macd
            avg_price = round(vwa.avg_price + 1.0 * ctx.fen.atr.avg_tr, 3)
            if not ctx.should_keep_pos: return False  # 不应该持仓
            if ctx.pos.total_value > 500: return False  # 已经有持仓
            if ctx.pos.curr_price <= avg_price: return False  # 价格未突破均价线
            if ema.ma05 <= ema.ma10: return False
            if ema.ma10 <= ema.ma20: return False
            if macd.dif_ <= 0: return False
            if macd.macd <= 0: return False
            return False

    class BuyBack:
        @staticmethod
        def eval(ctx: Ctx):
            if Step.BuyBack.is_buy_back(ctx):
                return "Buy", "End"
            return "Wait", "Wait"

        @staticmethod
        def is_buy_back(ctx: Ctx) -> bool:
            Rule.sync_back_price(ctx)
            ema = ctx.fen.ema
            macd = ctx.fen.macd
            if not ctx.should_keep_pos: return False  # 不应该持仓
            if ema.ma05 <= ema.ma10: return False
            if ema.ma10 <= ema.ma20: return False
            if macd.dif_ <= 0: return False
            if macd.macd <= 0: return False
            if ctx.act.buy_back_price > ctx.pos.curr_price: return False  # 风控价大于当前价格
            return True


class Rule:
    @staticmethod
    def calc_his_days(df: pd.DataFrame) -> list[Box.Node]:
        df = df.copy()
        px_volume = df["volume"]
        px_close = df["close"]
        px_high = df["high"]
        px_low = df["low"]

        # ATR
        tr1 = px_high - px_low
        tr2 = (px_high - px_close.shift(1)).abs()
        tr3 = (px_low - px_close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
        df["atr"] = tr.rolling(14).mean()

        # OBV
        dir_arr = px_close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        obv_vol = (dir_arr * px_volume)
        df["obv_vol"] = obv_vol.rolling(10).sum()
        df["sum_vol"] = px_volume.rolling(10).sum()
        df["obv_ratio"] = df["obv_vol"] / df["sum_vol"]

        # SMA
        thresh = Var.ma20_flat_thresh
        df["sma05"] = px_close.rolling(5).mean()
        df["sma10"] = px_close.rolling(10).mean()
        df["sma20"] = px_close.rolling(20).mean()
        df['sma20_slope'] = df['sma20'].rolling(5).apply(Rule.calc_lr_slope, raw=True)
        df["sma20_trend"] = np.select([df.sma20_slope > thresh, df.sma20_slope < -thresh], [1, -1], 0)

        # MACD
        df["ema12"] = px_close.ewm(span=12, adjust=False).mean()
        df["ema26"] = px_close.ewm(span=26, adjust=False).mean()
        df["dif_"] = df["ema12"] - df["ema26"]
        df["dea_"] = df["dif_"].ewm(span=9, adjust=False).mean()
        df["macd"] = (df["dif_"] - df["dea_"]) * 2 / df["sma10"].shift(1) * 100

        # BOLL
        std = px_close.rolling(20).std()
        df['boll_up'] = df["sma20"] + 2 * std
        df['boll_dn'] = df["sma20"] - 2 * std
        df['band_wid'] = (df['boll_up'] - df['boll_dn']) / df["sma20"]
        df['band_avg'] = df['band_wid'].rolling(20).mean()
        df['band_tag'] = np.select([df.band_wid < df.band_avg * 0.75, df.band_wid > df.band_avg * 1.2], [-1, 1], 0)

        df = df.round(4)
        return [Rule.to_node(index, row) for index, row in df.tail(10).iterrows()]

    @staticmethod
    def calc_lr_slope(vals):
        # 计算线性回归斜率
        return np.polyfit([0, 1, 2, 3, 4], vals, 1)[0]

    @staticmethod
    def to_node(index, row: pd.Series) -> Box.Node:
        # Bar
        bar = Bin.Bar()
        bar.datetime = index
        for k in ["volume", "money", "price", "close", "open", "high", "low"]:
            setattr(bar, k, row[k])

        # OBV
        node = Box.Node(bar)
        node.obv = Box.Obv()
        node.obv.obv_vol = row["obv_vol"]
        node.obv.sum_vol = row["sum_vol"]
        node.obv.obv_ratio = row["obv_ratio"]

        # SMA
        node.ema = Box.Ema()
        node.ema.ma05 = row["sma05"]
        node.ema.ma10 = row["sma10"]
        node.ema.ma20 = row["sma20"]

        # MACD
        node.macd = Box.Macd()
        node.macd.fast = row["ema12"]
        node.macd.slow = row["ema26"]
        node.macd.dif_ = row["dif_"]
        node.macd.dea_ = row["dea_"]
        node.macd.macd = row["macd"]

        # BOLL
        node.boll = Box.Boll()
        node.boll.atr = row["atr"]
        node.boll.ma20 = row["sma20"]
        node.boll.ma20_slope = row["sma20_slope"]
        node.boll.ma20_trend = row["sma20_trend"]
        node.boll.upper = row["boll_up"]
        node.boll.lower = row["boll_dn"]
        node.boll.band_wid = row["band_wid"]
        node.boll.band_avg = row["band_avg"]
        node.boll.band_tag = row["band_tag"]
        return node

    @staticmethod
    def should_keep_pos(ctx: Ctx) -> bool:
        """是否应该持仓"""
        day = ctx.day
        if day.obv.obv_ratio < -0.22:
            return False
        if day.ema.ma05 < day.ema.ma10 < day.ema.ma20:
            return False
        if day.macd.macd < -0.6:
            return False
        if day.boll.ma20_trend == -1 and day.bar.close < day.ema.ma10:
            return False
        if day.boll.band_tag == 1 and day.macd.macd < 0:
            return False
        if day.bar.close < day.ema.ma20 - 1.5 * day.boll.atr:
            return False
        return True

    @staticmethod
    def init_back_price(ctx: Ctx):
        atr = ctx.fen.atr.avg_tr
        sell_price = ctx.act.sell_price
        back_price = sell_price + 0.005 * ctx.base_price
        ctx.act.buy_back_price = round(back_price + 1.6 * atr, 3)

    @staticmethod
    def sync_back_price(ctx: Ctx):
        levels = [
            (8.0, 0.2),
            (6.0, 0.4),
            (4.5, 0.6),
            (3.0, 0.8),
            (2.0, 1.0),
            (1.0, 1.2),
            (0.5, 1.4),
            (0.0, 1.6),
        ]

        atr = ctx.fen.atr.avg_tr
        old_price = ctx.act.buy_back_price
        new_price = ctx.act.buy_back_price
        sell_price = ctx.act.sell_price
        pct = round((sell_price - ctx.curr_price) / ctx.base_price * 100, 3)
        for thresh, coef in levels:
            if pct >= thresh:
                new_price = sell_price - (thresh - 0.5) * ctx.base_price * 0.01 + coef * atr
                break
        ctx.act.buy_back_price = round(min(new_price, old_price), 3)


class Trader:
    def __init__(self, buy: Callable, sell: Callable):
        self.step = Step.Start
        self.sell = sell
        self.buy = buy

    def trading(self, market: Market):
        ctx = market.ctx
        if ctx.is_today_finish:
            return
        if ctx.curr_time <= "09:35:00":
            return
        if ctx.curr_time <= "14:40:00":
            self._main_trading(ctx)
            return
        if ctx.curr_time >= "14:45:00":
            self._tail_trading(ctx)
        # self._tail_trading(ctx)

    def _main_trading(self, ctx: Ctx):
        action, next_step = self.step.eval(ctx)
        if action == 'Wait':
            return
        if action == 'Buy':
            self._do_buy(ctx)
        if action == 'Sell':
            self._do_sell(ctx)
        if next_step == 'BuyBack':
            self.step = Step.BuyBack
            Rule.init_back_price(ctx)
            return
        if next_step == 'End':
            ctx.is_today_finish = True
            return

    def _tail_trading(self, ctx: Ctx):
        if ctx.pos.avail_value > 500 and not ctx.should_keep_pos:
            self._do_sell(ctx)
            return
        if ctx.pos.total_value < 500 and ctx.should_keep_pos:
            self._do_buy(ctx)
            return

    def _do_buy(self, ctx: Ctx):
        pos = ctx.pos
        curr_price = pos.curr_price
        buy_amount = round(Var.base_fund / curr_price / 100) * 100
        self.buy(pos.symbol, buy_amount, limit_price=curr_price + 0.003)
        ctx.act.has_buy = True
        ctx.act.buy_price = curr_price
        ctx.act.buy_amount = buy_amount

    def _do_sell(self, ctx: Ctx):
        pos = ctx.pos
        curr_price = pos.curr_price
        base_amount = round(Var.base_fund * 1.5 / curr_price / 100) * 100
        able_amount = min(pos.avail_amount, base_amount)
        sell_amount = able_amount - (0 if able_amount < pos.total_amount else 100)
        self.sell(pos.symbol, -sell_amount, limit_price=curr_price - 0.003)
        ctx.act.has_sell = True
        ctx.act.sell_price = curr_price
        ctx.act.sell_amount = sell_amount


############################################################
class Bus:
    def __init__(self):
        self.data: list[Box.Node] = []
        self.conf: dict[str, Any] = {}

    def __len__(self):
        return len(self.data)

    def add(self, node: Box.Node):
        self.data.append(node)

    def get(self, idx: int) -> Box.Node:
        return self.data[idx]

    def last(self) -> Box.Node:
        return self.data[-1]

    def extend(self, nodes: list[Box.Node]):
        self.data.extend(nodes)

    def rollback(self):
        node = self.data.pop()
        if node.flag >= 0:
            self.data.append(node)


class Ctx:
    def __init__(self):
        self.curr_time = ''  # 当前时间
        self.base_price = 0.0  # 昨收盘价
        self.open_price = 0.0  # 开盘价格
        self.curr_price = 0.0  # 最新价格
        self.rela_price_pct = 0.0  # 相对价格（%）

        self.pos = None  # 当前持仓
        self.day = None  # 当前节点(日)
        self.fen = None  # 当前节点(分)
        self.act = Bin.Act()  # 操作记录
        self.orb = Box.Orb()  # 开盘ORB
        self.agg_bar = None  # 今日聚合bar
        self.should_keep_pos = False  # 是否应该持有仓位
        self.is_today_finish = False  # 是否今天交易完成

    def refresh(self):
        self.curr_price = self.pos.curr_price
        if self.open_price == 0.0: self.open_price = self.day.bar.open
        self.rela_price_pct = round((self.curr_price - self.open_price) / self.base_price * 100, 2)
        self.should_keep_pos = Rule.should_keep_pos(self)


class Market:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.dayBus = Bus()  # 日线数据
        self.fenBus = Bus()  # 分钟数据
        self.ctx = Ctx()  # 上下文数据

    def prep(self, day_his: pd.DataFrame):
        self.dayBus.extend(Rule.calc_his_days(day_his))
        self.ctx.base_price = self.dayBus.last().bar.close
        self.fenBus.conf.update(Var.fen)
        return self

    def running(self, pos, bar):
        curr_time = bar.datetime.strftime("%H:%M:%S")

        # 日线数据
        self.dayBus.rollback()
        self.ctx.agg_bar = Bin.Bar(bar).agg(self.ctx.agg_bar)
        day_node = Box.Node(self.ctx.agg_bar).mark(-1)
        self.dayBus.add(day_node)
        Line.Ema.calc(self.dayBus)
        Line.Obv.calc(self.dayBus)
        Line.Macd.calc(self.dayBus)
        Line.Boll.calc(self.dayBus)

        # 分钟数据
        fen_node = Box.Node(Bin.Bar(bar))
        self.fenBus.add(fen_node)
        Line.Atr.calc(self.fenBus)
        Line.Ema.calc(self.fenBus)
        Line.Vwa.calc(self.fenBus)
        Line.Macd.calc(self.fenBus)
        if curr_time <= "10:50:00":
            Line.Orb.calc(self.fenBus)

        # 上下文数据
        self.ctx.curr_time = curr_time
        self.ctx.pos = Bin.Pos(pos)
        self.ctx.day = day_node
        self.ctx.fen = fen_node
        self.ctx.refresh()


############################################################
class Env:
    tickers: dict[str, tuple[Market, Trader]] = {}

    @staticmethod
    def to_dict(obj):
        if hasattr(obj, '__dict__'):
            return {k: Env.to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [Env.to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: Env.to_dict(v) for k, v in obj.items()}
        else:
            return obj

    @staticmethod
    def print(obj):
        print(Env.to_dict(obj))

    @staticmethod
    def marking(context):
        for market, _ in Env.tickers.values():
            ctx = market.ctx
            ctx.orb = copy.copy(ctx.fen.orb)


def initialize(context):
    """启动时执行一次"""
    run_daily(context, Env.marking, time='09:36')
    run_daily(context, Env.marking, time='09:41')
    run_daily(context, Env.marking, time='09:46')
    run_daily(context, Env.marking, time='09:51')

    if is_trade(): return
    set_commission(commission_ratio=0.00005, min_commission=0.5, type="ETF")
    # pos = {'sid': "515650.SS", 'amount': "100", 'enable_amount': "100", 'cost_basis': "1.0" }
    pos = {'sid': "159995.SZ", 'amount': "100", 'enable_amount': "100", 'cost_basis': "1.0"}
    set_yesterday_position([pos])


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    Env.tickers.clear()
    positions = context.portfolio.positions
    pos_codes = list(positions.keys())
    set_universe(pos_codes)
    day_history = get_history(240, frequency='1d', security_list=pos_codes)
    for symbol in pos_codes:
        day_df = day_history.query(f'code in ["{symbol}"]')
        market = Market(symbol).prep(day_df)
        trader = Trader(order, order)
        Env.tickers[symbol] = (market, trader)


def handle_data(context, data):
    """每个单位周期执行一次"""
    positions = context.portfolio.positions
    for symbol, (market, trader) in Env.tickers.items():
        bar = data[symbol]
        pos = positions.get(symbol)
        market.running(pos, bar)
        trader.trading(market)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    Env.tickers.clear()
    pass
