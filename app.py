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
from typing import Callable, Any

import numpy as np
import pandas as pd


class Var:
    base_fund = 3000  # 交易基础金额（元）
    ma20_flat_thresh = 0.005  # MA20斜率阈值

    fen = {
        "macd_fast": 13,
        "macd_slow": 60,
        "macd_sign": 5,
    }


class Bin:
    class Act:
        def __init__(self):
            self.has_sell = False  # 是否已经卖出
            self.sell_price = 0.0  # 卖出时的价格
            self.sell_amount = 0.0  # 卖出时的数量
            self.limit_back_price = 0.0  # 反T接回限制价格
            self.limit_sell_price = 0.0  # 冲高卖出限制价格

            self.has_buy = False
            self.buy_price = 0.0
            self.buy_amount = 0.0

    class Agg:
        def __init__(self):
            self.sum_vol = 0.0  # 今日，前X分钟，成交量之和
            self.avg_vol = 0.0  # 前20日，每天前X分钟，成交量和的平均值
            self.orb_low = 100  # 开盘区间内的最低价
            self.orb_high = 0.0  # 开盘区间内的最高价
            self.avg_vols = None  # 历史成交量的平均值

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


class Bus:
    def __init__(self):
        self.data: list[Node] = []
        self.conf: dict[str, Any] = {}

    def __len__(self):
        return len(self.data)

    def add(self, node: Node):
        self.data.append(node)

    def get(self, idx: int) -> Node:
        return self.data[idx]

    def last(self) -> Node:
        return self.data[-1]

    def extend(self, nodes: list[Node]):
        self.data.extend(nodes)

    def rollback(self):
        node = self.data.pop()
        if node.flag >= 0:
            self.data.append(node)


class Node:
    def __init__(self, bar: Bin.Bar):
        self.bar = bar
        self.ema = None
        self.macd = None
        self.flag = 0

    def mark(self, flag):
        self.flag = flag
        return self


############################################################
class Line:
    class Atr:
        @staticmethod
        def calc(bus):
            node = bus.last()
            high = node.bar.high
            low = node.bar.low
            if len(bus) == 1:
                node.atr = high - low
            else:
                pre_node = bus.get(-2)
                pre_close = pre_node.bar.close
                cur_value = max(high - low, abs(high - pre_close), abs(low - pre_close))
                node.atr = Line.Ema.mean(pre_node.atr, cur_value, 14)

    class Orb:
        @staticmethod
        def calc(bus, ctx):
            if ctx.curr_time > "09:50:00":
                return
            node = bus.last()
            price = node.bar.close
            node.orb = orb = SimpleNamespace()
            if len(bus) == 1:
                orb.orb_low = price  # 开盘区间内的最低价
                orb.orb_high = price  # 开盘区间内的最高价
            else:
                pre_orb = bus.get(-2).orb
                orb.orb_low = min(pre_orb.orb_low, price)
                orb.orb_high = max(pre_orb.orb_high, price)
            if ctx.curr_time in {"09:35:00", "09:40:00", "09:45:00", "09:50:00"}:
                ctx.agg.orb_low = orb.orb_low
                ctx.agg.orb_high = orb.orb_high

    class Ema:
        @staticmethod
        def calc(bus: Bus):
            node = bus.last()
            price = node.bar.close
            cur_ema = node.ema = SimpleNamespace()
            if len(bus) == 1:
                cur_ema.ma10 = cur_ema.ma20 = cur_ema.ma30 = price
            else:
                pre_ema = bus.get(-2).ema
                cur_ema.ma10 = Line.Ema.mean(pre_ema.ma10, price, 10)
                cur_ema.ma20 = Line.Ema.mean(pre_ema.ma20, price, 20)
                cur_ema.ma30 = Line.Ema.mean(pre_ema.ma30, price, 30)

        @staticmethod
        def mean(pre_val: float, price: float, period: int):
            alpha = 2 / (period + 1)
            value = alpha * price + (1 - alpha) * pre_val
            return round(value, 4)

    class Vol:
        @staticmethod
        def calc(bus):
            node = bus.last()
            if len(bus) == 1:
                pre_vol = SimpleNamespace(money=0, volume=0)
            else:
                pre_vol = bus.get(-2).vol
            bar = node.bar
            vol = node.vol = copy.copy(pre_vol)
            vol.volume += bar.volume  # 成交量总和
            vol.money += bar.money  # 成交额总和
            vol.price = round(vol.money / vol.volume, 4)  # 成交量加权平均价格

    class Macd:
        @staticmethod
        def calc(bus: Bus):
            node = bus.last()
            price = node.bar.close
            if len(bus) == 1:
                Line.Macd._first(node, price)
            else:
                Line.Macd._next(bus, node, price)

        @staticmethod
        def _first(node: Node, price: float):
            cur_macd = node.macd = SimpleNamespace()
            cur_macd.fast = cur_macd.slow = price
            cur_macd.dif_ = cur_macd.dea_ = cur_macd.macd = 0.0

        @staticmethod
        def _next(bus: Bus, node: Node, price: float):
            cur_macd = node.macd = SimpleNamespace()
            pre_node = bus.get(-2)
            pre_macd = pre_node.macd
            cur_macd.fast = Line.Ema.mean(pre_macd.fast, price, bus.conf.get('macd_fast', 12))
            cur_macd.slow = Line.Ema.mean(pre_macd.slow, price, bus.conf.get('macd_slow', 26))
            cur_macd.dif_ = round(cur_macd.fast - cur_macd.slow, 4)
            cur_macd.dea_ = Line.Ema.mean(pre_macd.dea_, cur_macd.dif_, bus.conf.get('macd_sign', 9))
            cur_macd.macd = round(2 * (cur_macd.dif_ - cur_macd.dea_), 4)

    class Days:
        @staticmethod
        def calc_live(bus):
            """即时日线指标"""
            pre_node = bus.get(-2)
            cur_node = bus.get(-1)
            price = cur_node.bar.close
            volume = cur_node.bar.volume

            # SMA
            pre_ema = pre_node.ema
            cur_ema = cur_node.ema = SimpleNamespace()
            cur_ema.ma05 = Line.Ema.mean(pre_ema.ma05, price, 5)
            cur_ema.ma10 = Line.Ema.mean(pre_ema.ma10, price, 10)
            cur_ema.ma20 = Line.Ema.mean(pre_ema.ma20, price, 20)

            # MACD
            pre_macd = pre_node.macd
            cur_macd = cur_node.macd = SimpleNamespace()
            cur_macd.fast = Line.Ema.mean(pre_macd.fast, price, 12)
            cur_macd.slow = Line.Ema.mean(pre_macd.slow, price, 26)
            cur_macd.dif_ = round(cur_macd.fast - cur_macd.slow, 4)
            cur_macd.dea_ = Line.Ema.mean(pre_macd.dea_, cur_macd.dif_, 9)
            cur_macd.macd = round(2 * (cur_macd.dif_ - cur_macd.dea_), 4)

            # OBV
            pre_close = pre_node.bar.close
            obv_vol = volume if price > pre_close else (-volume if price < pre_close else 0)
            cur_obv = cur_node.obv = SimpleNamespace()
            cur_obv.obv = pre_node.obv.obv + obv_vol
            cur_obv.obv_ma = Line.Ema.mean(pre_node.obv.obv_ma, cur_obv.obv, 20)

            # BOLL
            cur_node.boll = copy.copy(pre_node.boll)

        @staticmethod
        def calc_hist(df: pd.DataFrame) -> list[Node]:
            """历史日线指标"""
            px_volume, px_close, px_high, px_low = df["volume"], df["close"], df["high"], df["low"]

            # SMA
            thresh = Var.ma20_flat_thresh
            df["ma05"] = px_close.rolling(5).mean()
            df["ma10"] = px_close.rolling(10).mean()
            df["ma20"] = px_close.rolling(20).mean()
            df['ma20_slope'] = df['ma20'].rolling(5).apply(Line.Days._line_slope, raw=True)
            df["ma20_trend"] = np.select([df.ma20_slope > thresh, df.ma20_slope < -thresh], [1, -1], 0)

            # MACD
            df["ma12"] = px_close.ewm(span=12, adjust=False).mean()
            df["ma26"] = px_close.ewm(span=26, adjust=False).mean()
            df["dif_"] = df["ma12"] - df["ma26"]
            df["dea_"] = df["dif_"].ewm(span=9, adjust=False).mean()
            df["macd"] = 2 * (df["dif_"] - df["dea_"])

            # OBV
            px_diff = px_close.diff()
            vol_dir = np.where(px_diff > 0, px_volume, np.where(px_diff < 0, -px_volume, 0))
            df['obv'] = vol_dir.cumsum()
            df['obv_ma'] = df['obv'].rolling(window=20).mean()

            # ATR
            tr1 = px_high - px_low
            tr2 = (px_high - px_close.shift(1)).abs()
            tr3 = (px_low - px_close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
            df["atr"] = tr.rolling(14).mean()

            # BOLL
            std = px_close.rolling(20).std()
            df['boll_up'] = df["ma20"] + 2 * std
            df['boll_dn'] = df["ma20"] - 2 * std
            df['band_wid'] = (df['boll_up'] - df['boll_dn']) / df["ma20"]
            df['band_avg'] = df['band_wid'].rolling(20).mean()
            df['band_tag'] = np.select([df.band_wid < df.band_avg * 0.75, df.band_wid > df.band_avg * 1.2], [-1, 1], 0)

            df = df.round(4)
            return [Line.Days._to_node(index, row) for index, row in df.tail(200).iterrows()]

        @staticmethod
        def _line_slope(vals):
            # 线性回归斜率
            return np.polyfit([0, 1, 2, 3, 4], vals, 1)[0]

        @staticmethod
        def _to_node(index, row: pd.Series) -> Node:
            # Bar
            bar = Bin.Bar()
            bar.datetime = index
            for k in ["volume", "price", "close", "open", "high", "low"]:
                setattr(bar, k, row[k])

            # SMA
            node = Node(bar)
            node.ema = SimpleNamespace()
            node.ema.ma05 = row["ma05"]
            node.ema.ma10 = row["ma10"]
            node.ema.ma20 = row["ma20"]

            # MACD
            node.macd = SimpleNamespace()
            node.macd.fast = row["ma12"]
            node.macd.slow = row["ma26"]
            node.macd.dif_ = row["dif_"]
            node.macd.dea_ = row["dea_"]
            node.macd.macd = row["macd"]

            # OBV
            node.obv = SimpleNamespace()
            node.obv.obv = row["obv"]  # 能量潮
            node.obv.obv_ma = row["obv_ma"]  # 成交量合计

            # BOLL
            node.boll = SimpleNamespace()
            node.boll.atr = row["atr"]  # 平均真实波幅
            node.boll.upper = row["boll_up"]  # 上轨
            node.boll.lower = row["boll_dn"]  # 下轨
            node.boll.ma20_slope = row["ma20_slope"]  # 中轨斜率
            node.boll.ma20_trend = row["ma20_trend"]  # 中轨趋势：-1向下，0水平，1向上
            node.boll.band_wid = row["band_wid"]  # 布林带宽度
            node.boll.band_avg = row["band_avg"]  # 布林带均宽
            node.boll.band_tag = row["band_tag"]  # 布林带状态：-1收敛，0正常，1发散
            return node


class Step:
    class Start:
        @staticmethod
        def eval(ctx: Ctx):
            if ctx.pos.total_value < 500: return Flow.Next_Buy
            if ctx.pos.avail_value > 500: return Flow.Next_Sell
            return Flow.Wait

    class Buy:
        @staticmethod
        def eval(ctx: Ctx):
            """买入"""
            atr, ema, vol = ctx.fen.atr, ctx.fen.ema, ctx.fen.vol
            if ctx.curr_time < "09:45:00": return Flow.Wait
            if Rule.not_allow_buy(ctx): return Flow.Wait
            if Rule.overall_trend() < 2: return Flow.Wait  # 市场行情走弱
            if vol.price < min(ctx.zero_price, ctx.open_price): return Flow.Wait  # 均价在开盘价或零轴下方
            if ema.ma10 < vol.price + atr: return Flow.Wait  # 未突破均价线
            if ema.ma10 > ctx.agg.orb_high + atr: return Flow.Buy  # 突破ORH
            if ctx.wave_price_pct > 0.5: return Flow.Buy  # 较开盘价
            return Flow.Wait

    class Sell:
        @staticmethod
        def eval(ctx: Ctx):
            """卖出"""
            if ctx.curr_price_pct > 2: return Flow.Next_HighSell

            pass

    class HighSell:
        @staticmethod
        def eval(ctx: Ctx):
            """高位卖出"""
            levels = [
                (9.0, 0.4), (8.0, 0.8), (7.0, 1.2), (6.0, 1.6),
                (5.0, 2.0), (4.0, 2.0), (3.0, 1.5), (2.0, 1.0),
            ]
            act = ctx.act
            sell_px = act.limit_sell_price
            new_sell_px = sell_px if sell_px != 0.0 else ctx.zero_price
            for rise, buffer in levels:
                if ctx.curr_price_pct < rise: continue
                new_sell_px = ctx.zero_price * (100 + rise - buffer) * 0.01
                break
            act.limit_sell_price = round(max(sell_px, new_sell_px), 3)

            # 是否卖出
            atr, ema, vol = ctx.fen.atr, ctx.fen.ema, ctx.fen.vol
            if Rule.not_allow_sell(ctx): return Flow.Wait
            if ema.ma10 < vol.price - atr: return Flow.Sell  # 跌破均价线
            if ema.ma10 < act.limit_sell_price - atr: return Flow.Sell
            return Flow.Wait

    class Step02:
        @staticmethod
        def eval(ctx: Ctx):
            """判断是否卖出"""
            if Rule.not_allow_sell(ctx): return Flow.Wait
            if Rule.is_open_sideways(ctx): return Flow.Wait
            if ctx.curr_price_pct > 2: return Flow.Step02.sell_00(ctx)
            if ctx.curr_time <= "09:45:00": return Flow.Step02.sell_01(ctx)
            if ctx.curr_time <= "11:00:00": return Flow.Step02.sell_02(ctx)
            if ctx.curr_time >= "11:01:00": return Flow.Step02.sell_03(ctx)
            return Flow.Wait

        @staticmethod
        def sell_01(ctx: Ctx):
            agg, atr, ema, vol = ctx.agg, ctx.fen.atr, ctx.fen.ema, ctx.fen.vol
            if Rule.overall_trend() > 1: return Flow.Wait
            if agg.sum_vol < 1.2 * agg.avg_vol:  return Flow.Wait  # 未放量
            if ema.ma10 > vol.price - atr: return Flow.Wait  # 未跌破均价线
            if ema.ma10 < agg.orb_low - atr: return Flow.Sell  # 跌破ORL
            if ctx.wave_price_pct < -0.5: return Flow.Sell  # 较开盘价，跌破阈值
            return Flow.Wait

        @staticmethod
        def sell_02(ctx: Ctx):
            pct = ctx.open_price_pct
            fall_pct, coef = (
                (1.0, 2.0) if pct >= 5 else
                (0.8, 1.8) if pct >= 2 else
                (0.6, 1.6) if pct >= 1 else
                (0.4, 1.4)
            )
            atr, ema, vol = ctx.fen.atr, ctx.fen.ema, ctx.fen.vol
            if ctx.curr_price > vol.price - coef * atr: return Flow.Wait  # 未跌破均价线
            if ctx.curr_price < ctx.open_price * (100 - fall_pct) * 0.01 - coef * atr: return Flow.Sell  # 跌破阈值
            if ctx.curr_price < ctx.agg.orb_low - coef * atr: return Flow.Sell  # 跌破ORL
            if Rule.overall_trend() < -1: return Flow.Sell  # 大盘下跌
            return Flow.Wait

        @staticmethod
        def sell_03(ctx: Ctx):
            atr, vol = ctx.fen.atr, ctx.fen.vol
            if ctx.curr_price < ctx.open_price * 0.995 - 1.5 * atr: return Flow.Sell  # 跌破阈值
            if ctx.curr_price < ctx.agg.orb_low - 1.5 * atr: return Flow.Sell  # 跌破ORL
            if ctx.curr_price < vol.price - atr: return Flow.Sell  # 跌破均价线
            return Flow.Wait

    class Back:
        @staticmethod
        def eval(ctx: Ctx):
            Rule.sync_back_price(ctx)
            atr, ema, vol = ctx.fen.atr, ctx.fen.ema, ctx.fen.vol
            if Rule.not_allow_buy(ctx): return Flow.Wait  # 不应该买
            if ema.ma20 > ctx.act.limit_back_price: return Flow.Buy  # 突破风控价
            price = ema.ma20 if ctx.curr_time <= "11:00:00" else ctx.curr_price
            if price > vol.price + atr: return Flow.Buy  # 突破均线
            return Flow.Wait


class Flow:
    Buy = ("Buy", "End")
    Wait = ("Wait", "Wait")
    Sell = ("Sell", Step.Back)

    Next_Buy = ("Next", Step.Buy)
    Next_Sell = ("Next", Step.Sell)
    Next_HighSell = ("Next", Step.HighSell)


class Rule:
    @staticmethod
    def should_keep_pos(ctx: Ctx) -> bool:
        """是否应该持仓"""
        day = ctx.day
        bar, obv, ema, macd, boll = day.bar, day.obv, day.ema, day.macd, day.boll
        if boll.ma20_trend == -1 and bar.close < ema.ma10: return False
        if boll.band_tag == 1 and macd.macd < 0: return False
        if bar.close < ema.ma20 - 2 * boll.atr: return False
        if ema.ma05 < ema.ma10 < ema.ma20: return False
        if obv.obv - obv.obv_ma < 0: return False
        if macd.macd / ema.ma20 * 100 < -0.6: return False
        return True

    @staticmethod
    def not_allow_buy(ctx: Ctx) -> bool:
        """不允许买"""
        pos, ema, macd = ctx.pos, ctx.fen.ema, ctx.fen.macd
        if not ctx.should_keep_pos: return True  # 不应该持仓
        if pos.total_amount == 0.0: return True  # 被手动清仓的，不自动开仓
        if pos.total_value > 500: return True  # 已经有持仓
        if ema.ma10 < ema.ma30: return True  # 分时MA看空
        if macd.dif_ < 0: return True  # 分时MACD看空
        if macd.macd < 0: return True  # 分时MACD看空
        return False

    @staticmethod
    def not_allow_sell(ctx: Ctx) -> bool:
        """不允许卖"""
        ema, macd = ctx.fen.ema, ctx.fen.macd
        if ctx.pos.avail_value < 500: return True  # 没有可用持仓
        if ema.ma10 > ema.ma30: return True  # 分时MA看多
        if macd.dif_ > 0: return True  # 分时MACD看多
        if macd.macd > 0: return True  # 分时MACD看多
        return False

    @staticmethod
    def is_open_sideways(ctx: Ctx) -> bool:
        """是否平开平走行情"""
        if ctx.curr_time > "10:00:00": return False
        if abs(ctx.open_price_pct) > 0.4: return False
        vwap_zero_pct = (ctx.fen.vol.price - ctx.zero_price) / ctx.zero_price * 100
        return abs(vwap_zero_pct) < 0.5

    @staticmethod
    def sync_back_price(ctx: Ctx):
        """同步反T接回价格"""
        levels = [
            (9.0, 0.0, 0.0),
            (8.0, 0.5, 1.5),
            (7.0, 0.5, 1.5),
            (6.0, 1.0, 2.0),
            (5.0, 1.0, 2.0),
            (4.0, 1.0, 2.0),
            (3.0, 0.5, 1.5),
            (2.0, 0.5, 1.5),
        ]
        act, atr = ctx.act, ctx.fen.atr
        sell_price, buck_price = act.sell_price, act.limit_back_price
        if buck_price == 0:
            act.limit_back_price = round(sell_price + 1.5 * atr, 3)
            return

        new_price = buck_price
        price_pct = round((sell_price - ctx.curr_price) / ctx.zero_price * 100, 3)
        for thresh, pct, coef in levels:
            if price_pct < thresh: continue
            new_price = sell_price - (thresh - pct) * ctx.zero_price * 0.01 + coef * atr
            break
        act.limit_back_price = round(min(new_price, buck_price), 3)

    @staticmethod
    def overall_trend() -> int:
        """大盘趋势"""
        state = 0
        for mkt in Env.indexes.values():
            if not mkt.fenBus: continue
            agg, fen = mkt.ctx.agg, mkt.ctx.fen
            atr, ema, avg = fen.atr, fen.ema.ma10, fen.vol.price
            if ema < avg and ema < agg.orb_low - atr:
                state -= 1
            elif ema > avg and ema > agg.orb_high + atr:
                state += 1
        return state

    @staticmethod
    def his_avg_vols(df: pd.DataFrame, num: int) -> float:
        """最后20日，开盘前num分钟平均成交量list"""
        return (
            df.assign(date=df.index.date)
            .groupby('date').head(num)
            .groupby('date')['volume'].sum()
            .tail(20).mean().round()
        )


class Trader:
    def __init__(self, buy: Callable, sell: Callable):
        self.step = Step.Start
        self.sell = sell
        self.buy = buy

    def trading(self, market: Market):
        ctx = market.ctx
        if ctx.is_today_finish: return
        if ctx.curr_time <= "14:00:00":
            self._main_trading(ctx)
        elif ctx.curr_time >= "14:45:00":
            self._tail_trading(ctx)

    def _main_trading(self, ctx: Ctx):
        action, next_step = self.step.eval(ctx)
        if action == 'Buy': self._do_buy(ctx)
        if action == 'Sell': self._do_sell(ctx)
        if action == 'Wait': return
        if next_step == 'End': return
        if next_step == 'Wait': return
        self.step = Flow.step(next_step)

    def _tail_trading(self, ctx: Ctx):
        if ctx.pos.total_amount == 0.0: return
        if ctx.pos.avail_value > 500 and not ctx.should_keep_pos:
            self._do_sell(ctx)
        if ctx.pos.total_value < 500 and ctx.should_keep_pos:
            self._do_buy(ctx)

    def _do_buy(self, ctx: Ctx):
        symbol, curr_price = ctx.pos.symbol, ctx.pos.curr_price
        buy_amount = round(Var.base_fund / curr_price / 100) * 100
        self.buy(symbol, buy_amount, limit_price=curr_price + 0.003)
        ctx.is_today_finish = True
        ctx.act.has_buy = True
        ctx.act.buy_price = curr_price
        ctx.act.buy_amount = buy_amount

    def _do_sell(self, ctx: Ctx):
        pos = ctx.pos
        base_amount = round(Var.base_fund * 1.5 / pos.curr_price / 100) * 100
        able_amount = min(pos.avail_amount, base_amount)
        sell_amount = able_amount - (0 if able_amount < pos.total_amount else 100)
        self.sell(pos.symbol, -sell_amount, limit_price=pos.curr_price - 0.003)
        ctx.act.has_sell = True
        ctx.act.sell_price = pos.curr_price
        ctx.act.sell_amount = sell_amount


############################################################
class Ctx:
    def __init__(self):
        self.curr_time = ''  # 当前时间
        self.curr_price = 0.0  # 最新价格
        self.zero_price = 0.0  # 昨收盘价
        self.open_price = 0.0  # 今开盘价
        self.open_price_pct = 0.0  # 开盘价幅度（%）
        self.curr_price_pct = 0.0  # 当前价幅度（%）
        self.wave_price_pct = 0.0  # 当前价相对开盘价的幅度（%）

        self.should_keep_pos = False  # 是否应该持有仓位
        self.is_today_finish = False  # 是否今天交易完成
        self.act = Bin.Act()  # 操作记录
        self.agg = Bin.Agg()  # 聚合数据
        self.pos = None  # 当前持仓
        self.bar = None  # 当前bar(日)
        self.day = None  # 当前节点(日)
        self.fen = None  # 当前节点(分)

    def refresh(self):
        self.curr_price = self.pos.curr_price
        if self.open_price == 0.0:
            self.open_price = self.bar.open
            self.open_price_pct = round((self.open_price - self.zero_price) / self.zero_price * 100, 2)
        self.curr_price_pct = round((self.curr_price - self.zero_price) / self.zero_price * 100, 2)
        self.wave_price_pct = round((self.curr_price - self.open_price) / self.zero_price * 100, 2)
        self.should_keep_pos = Rule.should_keep_pos(self)


class Market:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.dayBus = Bus()  # 日线数据
        self.fenBus = Bus()  # 分钟数据
        self.ctx = Ctx()  # 上下文数据

    def prep(self, day_his: pd.DataFrame, fen_his: pd.DataFrame):
        day_df = day_his[["volume", "price", "close", "open", "high", "low"]].copy()
        self.ctx.agg.avg_vols = [Rule.his_avg_vols(fen_his, i) for i in [3, 2, 1]]
        self.dayBus.extend(Line.Days.calc_hist(day_df))
        self.ctx.zero_price = self.dayBus.last().bar.close
        self.fenBus.conf.update(Var.fen)
        return self

    def running(self, pos, bar):
        self.ctx.curr_time = bar.datetime.strftime("%H:%M:%S")

        # 日线数据
        self.dayBus.rollback()
        day_bar = Bin.Bar(bar).ohlc(self.ctx.bar)
        day_node = Node(day_bar).mark(-1)
        self.dayBus.add(day_node)
        Line.Days.calc_live(self.dayBus)

        # 分钟数据
        fen_node = Node(Bin.Bar(bar))
        self.fenBus.add(fen_node)
        Line.Atr.calc(self.fenBus)
        Line.Ema.calc(self.fenBus)
        Line.Vol.calc(self.fenBus)
        Line.Macd.calc(self.fenBus)
        Line.Orb.calc(self.fenBus, self.ctx)

        # 上下文数据
        self.ctx.pos = Bin.Pos(pos)
        self.ctx.bar = day_bar
        self.ctx.day = day_node
        self.ctx.fen = fen_node
        self.ctx.refresh()

    def running_index(self, bar):
        fen_node = Node(Bin.Bar(bar))
        self.fenBus.add(fen_node)
        Line.Ema.calc(self.fenBus)
        Line.Vol.calc(self.fenBus)
        Line.Macd.calc(self.fenBus)
        Line.Orb.calc(self.fenBus, self.ctx)
        self.ctx.fen = fen_node


############################################################
class Env:
    indexes: dict[str, Market] = {}
    tickers: dict[str, tuple[Market, Trader]] = {}

    @staticmethod
    def restart(context):
        Env.tickers.clear()
        Env.indexes = {
            "510300.SS": Market("510300.SS"),  # 沪深300
            "510500.SS": Market("510500.SS"),  # 中证500
            "512100.SS": Market("512100.SS")  # 中证1000
        }

    @staticmethod
    def marking(context):
        for market, _ in Env.tickers.values():
            agg = market.ctx.agg
            vol = market.ctx.fen.vol
            agg.sum_vol = vol.volume
            agg.avg_vol = agg.avg_vols.pop()

    @staticmethod
    def clear_open_orders(symbol):
        status = {'0', '1', '2', '7'}
        orders = get_open_orders(symbol)
        for order in orders:
            if order.status in status:
                cancel_order(order.id)


def initialize(context):
    """启动时执行一次"""
    run_daily(context, Env.marking, time='09:36')
    run_daily(context, Env.marking, time='09:41')
    run_daily(context, Env.marking, time='09:46')

    if is_trade(): return
    set_commission(commission_ratio=0.00005, min_commission=0.5, type="ETF")
    # pos = {'sid': "588790.SS", 'amount': "100", 'enable_amount': "100", 'cost_basis': "1.0"}
    pos = {'sid': "159995.SZ", 'amount': "100", 'enable_amount': "100", 'cost_basis': "1.0"}
    set_yesterday_position([pos])


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    Env.restart(context)
    positions = context.portfolio.positions
    pos_codes = list(positions.keys())
    all_codes = pos_codes + list(Env.indexes.keys())
    set_universe(all_codes)
    day_history = get_history(120, frequency='1d', security_list=pos_codes)
    fen_history = get_history(960, frequency='5m', security_list=pos_codes)
    for symbol in pos_codes:
        day_df = day_history.query(f'code in ["{symbol}"]')
        fen_df = fen_history.query(f'code in ["{symbol}"]')
        market = Market(symbol).prep(day_df, fen_df)
        trader = Trader(order, order)
        Env.tickers[symbol] = (market, trader)


def handle_data(context, data):
    """每个单位周期执行一次"""
    curr_time = context.blotter.current_dt.strftime("%H:%M:%S")
    positions = context.portfolio.positions
    if curr_time <= "10:00:00":
        for symbol, market in Env.indexes.items():
            bar = data[symbol]
            market.running_index(bar)
    for symbol, (market, trader) in Env.tickers.items():
        bar = data[symbol]
        pos = positions.get(symbol)
        market.running(pos, bar)
        trader.trading(market)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    Env.tickers.clear()
    pass
