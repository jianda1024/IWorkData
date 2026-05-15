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

from typing import Callable, Any

import pandas as pd


class Var:
    base_fund = 3000  # 交易基础金额（元）
    open_trade_time = "09:35:00"  # 开盘静默时间（09:30 ~ 09:35）
    head_trade_time = "09:40:00"  # 开盘静默时间（09:35 ~ 09:40）
    main_trade_time = "14:30:00"  # 盘中交易时间（09:35 ~ 14:30）
    tail_trade_time = "14:55:00"  # 尾盘平仓时间（14:55 ~ 15:00）

    fen = {
        "macd_fast": 13,
        "macd_slow": 60,
        "macd_sign": 5,
    }

    rise_safe_lvls = {
        9.0: 1.080,
        8.0: 1.070,
        7.0: 1.060,
        6.0: 1.050,
        5.0: 1.040,
        4.0: 1.032,
        3.5: 1.025,
        3.0: 1.020,
        2.5: 1.015,
        2.0: 1.010,
        1.5: 1.005,
        1.0: 1.000,
    }
    fall_safe_lvls = {
        9.0: 0.920,
        8.0: 0.930,
        7.0: 0.940,
        6.0: 0.950,
        5.0: 0.960,
        4.0: 0.968,
        3.5: 0.975,
        3.0: 0.980,
        2.5: 0.985,
        2.0: 0.990,
        1.5: 0.995,
        1.0: 1.000,
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
            self.safe_rise_price = 0.0  # 正T风控价格
            self.safe_fall_price = 1000  # 反T风控价格

    class Agg:
        def __init__(self):
            self.avg_vol = 0.0  # 前20日，每天前X分钟，成交量和的平均值
            self.sum_vol = 0.0  # 今日，前X分钟，成交量之和
            self.orb_low = 100  # 开盘区间内的最低价
            self.orb_high = 0.0  # 开盘区间内的最高价
            self.orb_mid_low = 0.0  # ORL+(ORH-ORL)*0.3
            self.orb_mid_high = 0.0  # ORL+(ORH-ORL)*0.7

    class Pos:
        def __init__(self, pos):
            self.symbol: str = getattr(pos, 'sid', '')  # 股票代码
            self.total_amount: float = getattr(pos, 'amount', 0.0)  # 总持仓数量
            self.avail_amount: float = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
            self.curr_price: float = getattr(pos, 'last_sale_price', 0.0)  # 最新价格
            self.cost_price: float = getattr(pos, 'cost_basis', 0.0)  # 成本价格
            self.valuation: float = round(self.total_amount * self.curr_price, 3)  # 市值
            self.principal: float = round(self.total_amount * self.cost_price, 3)  # 本金

    class Bar:
        def __init__(self, bar=None):
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

    class Obv:
        def __init__(self):
            self.obv: float = 0.0
            self.obv_ma20: float = 0.0

    class Ema:
        def __init__(self):
            self.ma05: float = 0.0
            self.ma10: float = 0.0
            self.ma20: float = 0.0
            self.ma30: float = 0.0

    class Macd:
        def __init__(self):
            self.fast: float = 0.0
            self.slow: float = 0.0
            self.dif_: float = 0.0
            self.dea_: float = 0.0
            self.macd: float = 0.0

    class Boll:
        def __init__(self):
            self.atr: float = 0.0
            self.mid: float = 0.0
            self.upper: float = 0.0
            self.lower: float = 0.0
            self.width: float = 0.0

    class Node:
        def __init__(self, bar: Bin.Bar):
            self.time: str = bar.datetime.strftime("%H:%M:%S")
            self.bar = bar
            self.ema = None
            self.macd = None
            self.flag: int = 0

        def mark(self, flag):
            self.flag = flag
            return self


class Line:
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
            node.vwa = vwa = Box.Vwa()
            if len(bus) == 1:
                prev_money = prev_volume = 0
            else:
                prev_vwa = bus.get(-2).vwa
                prev_money, prev_volume = prev_vwa.sum_money, prev_vwa.sum_volume
            bar = node.bar
            vwa.sum_volume = prev_volume + bar.volume
            vwa.sum_money = prev_money + bar.money
            vwa.avg_price = round(vwa.sum_money / vwa.sum_volume / 100, 4)
            vwa.sum_bar += 1
            if bar.close >= bar.open:
                vwa.sum_yan += 1
                vwa.sum_yan_vol += bar.volume
            else:
                vwa.sum_yin += 1
                vwa.sum_yin_vol += bar.volume

    class Obv:
        @staticmethod
        def calc(bus):
            if len(bus) == 1: return
            node = bus.last()
            pre = bus.get(-2)
            close = node.bar.close
            volume = node.bar.volume
            pre_close = pre.bar.close
            vol = volume if close > pre_close else (-volume if close < pre_close else 0)

            node.obv = Box.Obv()
            node.obv.obv = pre.obv.obv + vol
            node.obv.obv_ma20 = pre.obv.obv_ma20

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
            curr_ema.ema05 = price
            curr_ema.ema10 = price
            curr_ema.ema20 = price
            curr_ema.ema30 = price

        @staticmethod
        def next(bus: Bus, node: Box.Node, price: float):
            curr_ema = node.ema
            prev_ema = bus.get(-2).ema
            curr_ema.ema05 = Line.Ema.ema(prev_ema.ema05, price, 5)
            curr_ema.ema10 = Line.Ema.ema(prev_ema.ema10, price, 10)
            curr_ema.ema20 = Line.Ema.ema(prev_ema.ema20, price, 20)
            curr_ema.ema30 = Line.Ema.ema(prev_ema.ema30, price, 30)

        @staticmethod
        def ema(prev_val: float, price: float, period: int):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * prev_val
            return round(value, 4)

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
            curr_macd.ema_ = price
            curr_macd.dif_ = 0.0
            curr_macd.dea_ = 0.0
            curr_macd.macd = 0.0

        @staticmethod
        def _next(bus: Bus, node: Box.Node, price: float):
            curr_macd = node.macd
            prev_node = bus.get(-2)
            prev_macd = prev_node.macd
            base_price = bus.conf.get('base_price', node.ema.ema20)
            curr_macd.fast = Line.Ema.ema(prev_macd.fast, price, bus.conf.get('macd_fast', 12))
            curr_macd.slow = Line.Ema.ema(prev_macd.slow, price, bus.conf.get('macd_slow', 26))
            curr_macd.dif_ = round((curr_macd.fast - curr_macd.slow) / base_price * 100, 5)
            curr_macd.dea_ = Line.Ema.ema(prev_macd.dea_, curr_macd.dif_, bus.conf.get('macd_sign', 9))
            curr_macd.macd = round((curr_macd.dif_ - curr_macd.dea_) * 2, 4)

    class Boll:
        @staticmethod
        def calc(bus):
            if len(bus) == 1: return
            atr = bus.get(-2).boll.atr
            node = bus.last()
            ema20 = node.ema.ema20

            node.boll = Box.Boll()
            node.boll.atr = atr
            node.boll.mid = ema20
            node.boll.upper = round(ema20 + 2 * atr, 4)
            node.boll.lower = round(ema20 - 2 * atr, 4)
            node.boll.width = round(4 * atr / ema20, 4)

    class Index:
        @staticmethod
        def calc_avg_vol(df: pd.DataFrame, num: int) -> float:
            """最后20日，开盘前num分钟平均成交量"""
            return (
                df.assign(date=df.index.date)
                .groupby('date').head(num)
                .groupby('date')['volume'].sum()
                .tail(20).mean().round()
            )

        @staticmethod
        def calc_days(df: pd.DataFrame) -> list[Box.Node]:
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
            df["obv"] = (dir_arr * px_volume).cumsum()
            df["obv_ma20"] = df["obv"].rolling(20).mean()

            # SMA
            df["ma05"] = px_close.rolling(5).mean()
            df["ma10"] = px_close.rolling(10).mean()
            df["ma20"] = px_close.rolling(20).mean()

            # MACD
            ema12 = px_close.ewm(span=12, adjust=False).mean()
            ema26 = px_close.ewm(span=26, adjust=False).mean()
            df["diff"] = ema12 - ema26
            df["dea"] = df["diff"].ewm(span=9, adjust=False).mean()
            df["macd"] = (df["diff"] - df["dea"]) * 2

            # BOLL
            df["mid"] = df["ma20"]
            df["upper"] = df["ma20"] + 2 * df["atr"]
            df["lower"] = df["ma20"] - 2 * df["atr"]
            df["width"] = (df["upper"] - df["lower"]) / df["ma20"]
            df = df.round(4)
            return [Line.Index.row_to_node(index, row) for index, row in df.tail(10).iterrows()]

        @staticmethod
        def row_to_node(index, row: pd.Series) -> Box.Node:
            # Bar
            bar = Bin.Bar()
            bar.datetime = index
            for k in ["volume", "money", "price", "close", "open", "high", "low"]:
                setattr(bar, k, row[k])

            # OBV
            node = Box.Node(bar)
            node.obv = Box.Obv()
            node.obv.obv = row["obv"]
            node.obv.obv_ma20 = row["obv_ma20"]

            # SMA
            node.ema = Box.Ema()
            node.ema.ma05 = row["ma05"]
            node.ema.ma10 = row["ma10"]
            node.ema.ma20 = row["ma20"]

            # BOLL
            node.boll = Box.Boll()
            node.boll.atr = row["atr"]
            node.boll.mid = row["mid"]
            node.boll.upper = row["upper"]
            node.boll.lower = row["lower"]
            node.boll.width = row["width"]
            return node


class Step:
    class Start:
        @staticmethod
        def eval(ctx: Ctx):
            if Step.Start.branch_buy(ctx):
                return "Buy", "BuyStep"
            if Step.Start.branch_sell(ctx):
                return "Sell", "SellStep"
            if ctx.node.time <= Var.head_trade_time:
                return "Hold", "Wait"
            if ctx.rise_score - ctx.fall_score >= 15 and ctx.rise_score >= 70:
                return "Buy", "BuyStep"
            if ctx.fall_score - ctx.rise_score >= 15 and ctx.fall_score >= 70:
                return "Sell", "SellStep"
            return "Hold", "Wait"

        @staticmethod
        def branch_buy(ctx: Ctx) -> bool:
            vwa = ctx.node.vwa
            if not ctx.is_allow_buy: return False
            if ctx.curr_price <= ctx.agg.orb_high: return False
            if ctx.curr_price <= vwa.avg_price: return False
            if ctx.agg.sum_vol <= 0.8 * ctx.agg.avg_vol: return False
            if ctx.comp_price_pct < 0.4: return False
            if vwa.sum_yan < 1.2 * vwa.sum_yin: return False
            if vwa.sum_yan_vol < 1.2 * vwa.sum_yin_vol: return False
            return True

        @staticmethod
        def branch_sell(ctx: Ctx) -> bool:
            vwa = ctx.node.vwa
            if not ctx.is_allow_sell: return False
            if ctx.curr_price >= ctx.agg.orb_low: return False
            if ctx.curr_price >= vwa.avg_price: return False
            if ctx.agg.sum_vol <= 0.8 * ctx.agg.avg_vol: return False
            if ctx.comp_price_pct > -0.4: return False
            if vwa.sum_yin < 1.2 * vwa.sum_yan: return False
            if vwa.sum_yin_vol < 1.2 * vwa.sum_yan_vol: return False
            return True

    class Sell:
        @staticmethod
        def eval(ctx: Ctx):
            if Step.Sell.branch_01(ctx): return "Buy", "EndStep"
            return "Hold", "Wait"

        @staticmethod
        def branch_01(ctx: Ctx) -> bool:
            if not ctx.is_allow_buy: return False
            if ctx.node.ema.ema05 < ctx.act.safe_fall_price: return False
            if ctx.ema_state == 'fall': return False
            if ctx.macd_state == 'fall': return False
            return True

    class Buy:
        @staticmethod
        def eval(ctx: Ctx):
            if Step.Buy.branch_01(ctx): return "Sell", "EndStep"
            return "Hold", "Wait"

        @staticmethod
        def branch_01(ctx: Ctx) -> bool:
            if not ctx.is_allow_sell: return False
            if ctx.node.ema.ema05 > ctx.act.safe_rise_price: return False
            if ctx.ema_state == 'rise': return False
            if ctx.macd_state == 'rise': return False
            return True

    class End:
        @staticmethod
        def eval():
            return "Hold", "Wait"


class Util:
    @staticmethod
    def to_dict(obj):
        """递归将对象转换为字典"""
        if hasattr(obj, '__dict__'):
            return {k: Util.to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [Util.to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: Util.to_dict(v) for k, v in obj.items()}
        else:
            return obj

    @staticmethod
    def print(obj):
        print(Util.to_dict(obj))


############################################################
class Rule:
    @staticmethod
    def is_allow_buy(market: Market) -> bool:
        """是否允许买入"""
        ctx = market.ctx
        day = market.dayBus.last()
        if ctx.act.has_buy: return False
        if day.ema.ema05 > day.ema.ema10 > day.ema.ema20:
            return True
        if day.ema.ema05 < day.ema.ema10 < day.ema.ema20:
            return False
        if day.macd.dif_ > 0:
            if day.macd.macd > 0: return True
            if day.macd.macd < -0.5: return False
            if (ctx.node.ema.ema20 - day.ema.ema20) / day.ema.ema20 * 100 > -0.5: return True
        if day.macd.dif_ <= 0:
            if day.macd.macd <= 0: return False
            if ctx.node.ema.ema20 - day.ema.ema20 > 0: return True
            if day.macd.macd > 1: return True
        return False

    @staticmethod
    def is_allow_sell(market: Market) -> bool:
        """是否允许卖出"""
        ctx = market.ctx
        if ctx.act.has_sell: return False
        return ctx.pos.avail_amount * ctx.pos.curr_price > 500

    @staticmethod
    def ema_state(ctx: Ctx) -> str:
        ema = ctx.node.ema
        if ema.ema05 < ema.ema10 < ema.ema20: return 'fall'
        if ema.ema05 > ema.ema10 > ema.ema20: return 'rise'
        return 'flat'

    @staticmethod
    def macd_state(ctx: Ctx) -> str:
        macd = ctx.node.macd
        if macd.dif_ < 0 and macd.dea_ < 0: return 'fall'
        if macd.dif_ > 0 and macd.dea_ > 0: return 'rise'
        return 'flat'

    @staticmethod
    def rise_score(ctx: Ctx) -> int:
        rise_score = 0
        if not ctx.is_allow_buy: return rise_score

        agg = ctx.agg
        vwa = ctx.node.vwa
        curr_price = ctx.node.ema.ema05
        if curr_price > agg.orb_high: rise_score += 35  # 突破ORH
        if agg.orb_mid_high < curr_price <= agg.orb_high: rise_score += 20  # ORH区间

        if curr_price > vwa.avg_price: rise_score += 25  # 在均价线上
        if agg.sum_vol > 1.2 * agg.avg_vol: rise_score += 20  # 强放量
        if 0.8 * agg.avg_vol < agg.sum_vol <= 1.2 * agg.avg_vol: rise_score += 10  # 温和量

        if vwa.sum_yan > 1.2 * vwa.sum_yin: rise_score += 10  # 阳线数量
        if vwa.sum_yan_vol > 1.2 * vwa.sum_yin_vol: rise_score += 10  # 阳线成交量

        if ctx.ema_state == 'rise': rise_score += 3  # MA
        if ctx.macd_state == 'rise': rise_score += 2  # MACD
        return rise_score

    @staticmethod
    def fall_score(ctx: Ctx) -> int:
        fall_score = 0
        if not ctx.is_allow_sell: return fall_score

        agg = ctx.agg
        vwa = ctx.node.vwa
        curr_price = ctx.node.ema.ema05
        if curr_price < agg.orb_low: fall_score += 35  # 突破ORL
        if agg.orb_low <= curr_price < agg.orb_mid_low: fall_score += 20  # ORL区间

        if curr_price < vwa.avg_price: fall_score += 25  # 在均价线下
        if agg.sum_vol > 1.2 * agg.avg_vol: fall_score += 20  # 强放量
        if 1.2 * agg.avg_vol >= agg.sum_vol > 0.8 * agg.avg_vol: fall_score += 10  # 温和量

        if vwa.sum_yin > 1.2 * vwa.sum_yan: fall_score += 10  # 阴线数量
        if vwa.sum_yin_vol > 1.2 * vwa.sum_yan_vol: fall_score += 10  # 阴线成交量

        if ctx.ema_state == 'fall': fall_score += 3  # MA
        if ctx.macd_state == 'fall': fall_score += 2  # MACD
        return fall_score

    @staticmethod
    def safe_fall_price(ctx: Ctx):
        act = ctx.act
        if act.has_buy: return
        if not act.has_sell: return
        if act.sell_price == 0.0: return
        rate = 1.005
        pct = round((act.sell_price - ctx.curr_price) / act.sell_price * 100, 2)
        for threshold, ratio in Var.fall_safe_lvls.items():
            if pct >= threshold:
                rate = ratio
                break
        act.safe_fall_price = min(act.safe_fall_price, round(act.sell_price * rate, 3))

    @staticmethod
    def safe_rise_price(ctx: Ctx):
        act = ctx.act
        if act.has_sell: return
        if not act.has_buy: return
        if act.buy_price == 0.0: return
        rate = 0.995
        pct = round((ctx.curr_price - act.buy_price) / act.buy_price * 100, 2)
        for threshold, ratio in Var.rise_safe_lvls.items():
            if pct >= threshold:
                rate = ratio
                break
        act.safe_rise_price = max(act.safe_rise_price, round(act.buy_price * rate, 3))


class Trader:
    def __init__(self, buy: Callable, sell: Callable):
        self.step = Step.Start
        self.sell = sell
        self.buy = buy

    def main_trading(self, ctx: Ctx):
        if ctx.act.has_buy and ctx.act.has_sell: return
        action, next_step = self.step.eval(ctx)
        if action == 'Buy':
            self._do_buy(ctx)
        if action == 'Sell':
            self._do_sell(ctx)
        if next_step != 'Wait':
            self._to_next(next_step)

    def tail_trading(self, ctx: Ctx):
        if ctx.act.has_buy and ctx.is_allow_sell:
            self._do_sell(ctx)
            return
        if ctx.act.has_sell and ctx.is_allow_buy:
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
        # Env.print(ctx)

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
        # Env.print(ctx)

    def _to_next(self, next_step: str):
        if next_step == 'SellStep':
            self.step = Step.Sell()
            return
        if next_step == 'BuyStep':
            self.step = Step.Buy()
            return
        if next_step == 'EndStep':
            self.step = Step.End()
            return


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
        self.base_price = 0.0  # 昨收盘价
        self.open_price = 0.0  # 开盘价格
        self.curr_price = 0.0  # 最新价格
        self.open_price_pct = 0.0  # 开盘价格（%）
        self.curr_price_pct = 0.0  # 最新价格（%）
        self.comp_price_pct = 0.0  # 相对价格（%）

        self.pos = None  # 当前持仓
        self.day = None  # 当前节点(日)
        self.fen = None  # 当前节点(分)
        self.act = Bin.Act()  # 操作记录
        self.agg = Bin.Agg()  # 聚合数据
        self.his_avg_vols = []  # 最后20日，开盘前X分钟平均成交量
        self.is_allow_buy = False  # 是否允许买入
        self.is_allow_sell = False  # 是否允许卖出

    def refresh(self):
        self.curr_price = self.pos.curr_price
        if self.open_price == 0.0: self.open_price = self.day.bar.open
        self.open_price_pct = round((self.open_price - self.base_price) / self.base_price * 100, 2)
        self.curr_price_pct = round((self.curr_price - self.base_price) / self.base_price * 100, 2)
        self.comp_price_pct = round((self.curr_price - self.open_price) / self.base_price * 100, 2)
        Rule.safe_rise_price(self)
        Rule.safe_fall_price(self)

    def collect(self):
        low = self.fen.orb.orb_low
        high = self.fen.orb.orb_high
        self.agg.sum_vol = self.fen.vwa.sum_volume
        self.agg.avg_vol = self.his_avg_vols.pop()
        self.agg.orb_low = low
        self.agg.orb_high = high
        self.agg.orb_mid_low = round(low + (high - low) * 0.3, 3)
        self.agg.orb_mid_high = round(low + (high - low) * 0.7, 3)


class Market:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.dayBus = Bus()  # 日线数据
        self.fenBus = Bus()  # 分钟数据
        self.ctx = Ctx()  # 上下文数据

    def prep(self, day_his: pd.DataFrame, fen_his: pd.DataFrame):
        self.dayBus.extend(Line.Index.calc_days(day_his))
        self.fenBus.conf.update(Var.fen)
        self.fenBus.conf['base_price'] = self.ctx.base_price
        self.ctx.base_price = self.dayBus.last().bar.close
        self.ctx.his_avg_vols = [Line.Index.calc_avg_vol(fen_his, i) for i in [4, 3, 2, 1]]
        return self

    def running(self, pos, bar):
        # 日线数据
        self.dayBus.rollback()
        last_bar = self.dayBus.last().bar
        curr_bar = Bin.Bar(bar).agg(last_bar)
        day = Box.Node(curr_bar).mark(-1)
        self.dayBus.add(day)
        Line.Obv.calc(self.dayBus)
        Line.Ema.calc(self.dayBus)
        Line.Macd.calc(self.dayBus)
        Line.Boll.calc(self.dayBus)

        # 分钟数据
        fen = Box.Node(Bin.Bar(bar))
        self.fenBus.add(fen)
        Line.Vwa.calc(self.fenBus)
        Line.Ema.calc(self.fenBus)
        Line.Macd.calc(self.fenBus)
        if fen.time <= "10:50:00":
            Line.Orb.calc(self.fenBus)

        # 上下文数据
        self.ctx.pos = Bin.Pos(pos)
        self.ctx.day = day
        self.ctx.fen = fen
        self.ctx.refresh()
        self.ctx.is_allow_buy = Rule.is_allow_buy(self)
        self.ctx.is_allow_sell = Rule.is_allow_sell(self)

    def trading(self):
        if not self.ctx.is_allow_buy and not self.ctx.is_allow_sell:
            return
        curr_time = self.ctx.node.time
        if curr_time <= Var.open_trade_time:
            return
        if curr_time <= Var.main_trade_time:
            self.trader.main_trading(self.ctx)
            return
        if curr_time >= Var.tail_trade_time:
            self.trader.tail_trading(self.ctx)


############################################################
class Env:
    markets: dict[str, Market] = {}
    _trader: Trader = None

    @staticmethod
    def trader() -> Trader:
        if Env._trader is None:
            Env._trader = Trader(order, order)
        return Env._trader

    @staticmethod
    def set_agg(context):
        for market in Env.markets.values():
            market.ctx.collect()


############################################################
def initialize(context):
    """启动时执行一次"""
    run_daily(context, Env.set_agg, time='09:36')
    run_daily(context, Env.set_agg, time='09:41')
    run_daily(context, Env.set_agg, time='09:46')
    run_daily(context, Env.set_agg, time='09:51')

    if is_trade(): return
    set_commission(commission_ratio=0.00005, min_commission=0.5, type="ETF")
    pos = {'sid': "515650.SS", 'amount': "100", 'enable_amount': "100", 'cost_basis': "1.0"}
    set_yesterday_position([pos])


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    Env.markets.clear()
    positions = context.portfolio.positions
    pos_codes = list(positions.keys())
    set_universe(pos_codes)
    day_history = get_history(60, frequency='1d', security_list=pos_codes)
    fen_history = get_history(960, frequency='5m', security_list=pos_codes)
    for code in pos_codes:
        day_df = day_history.query(f'code in ["{code}"]')
        fen_df = fen_history.query(f'code in ["{code}"]')
        market = Market(code).prep(day_df, fen_df)
        Env.markets[code] = market


def handle_data(context, data):
    """每个单位周期执行一次"""
    positions = context.portfolio.positions
    for symbol, market in Env.markets.items():
        bar = data[symbol]
        pos = positions.get(symbol)
        market.running(pos, bar)
        market.trading()


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    Env.markets.clear()
    pass
