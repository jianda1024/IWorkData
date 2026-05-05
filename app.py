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

from collections import deque
from datetime import time
from types import SimpleNamespace
from typing import Callable


class Var:
    base_fund = 3000  # 交易基础金额（元）
    open_data_time = time(9, 35, 0)  # 开盘急涨急跌，收集数据时间（09:30~09:35）
    open_eval_time = time(9, 36, 0)  # 开盘急涨急跌，判断评估时间（09:39）
    head_data_time = time(9, 50, 0)  # 开盘定调收集数据时间（09:30~09:50）
    main_eval_time = time(14, 30, 0)  # 盘中操作时间（09:50~14:30）
    tail_eval_time = time(14, 55, 0)  # 尾盘平仓时间：14:55 平掉所有T仓，回归底仓

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
        def __init__(self, bar):
            self.datetime = bar.datetime
            self.volume: float = round(bar.volume, 3)  # 交易量
            self.money: float = round(bar.money, 3)  # 交易金额
            self.price: float = round(bar.price, 3)  # 最新价
            self.close: float = round(bar.close, 3)  # 收盘价
            self.open: float = round(bar.open, 3)  # 开盘价
            self.high: float = round(bar.high, 3)  # 最高价
            self.low: float = round(bar.low, 3)  # 最低价


class Box:
    class Vma:
        def __init__(self):
            """成交量加权平均价格"""
            self.volume: float = 0.0
            self.money: float = 0.0
            self.value: float = 0.0

    class Ema:
        def __init__(self):
            self.ema05: float = 0.0
            self.ema10: float = 0.0
            self.ema20: float = 0.0
            self.ema30: float = 0.0

    class Macd:
        def __init__(self):
            self.fast: float = 0.0
            self.slow: float = 0.0
            self.dif_: float = 0.0
            self.dea_: float = 0.0
            self.macd: float = 0.0


class Bus:
    def __init__(self, maxlen=None, conf=None):
        if conf is None: conf = {}
        self.data: deque[Node] = deque(maxlen=maxlen)
        self.conf = conf

    def __len__(self):
        return len(self.data)

    def add(self, node: Node):
        self.data.append(node)

    def get(self, idx: int) -> Node:
        return self.data[idx]

    def last(self) -> Node:
        return self.data[-1]

    def rollback(self):
        node = self.data.pop()
        if node.flag >= 0:
            self.data.append(node)


class Node:
    def __init__(self, bar):
        self.time: str = bar.datetime.strftime("%H:%M:%S")
        self.flag: int = 0

        self.bar = Bin.Bar(bar)
        self.vma = None
        self.ema = None
        self.macd = None

    def mark(self, flag):
        self.flag = flag
        return self


############################################################
class Line:
    class Vma:
        @staticmethod
        def calc(bus: Bus):
            vma = Box.Vma()
            node = bus.last()
            node.vma = vma
            if len(bus) == 1:
                vma.money = node.bar.money
                vma.volume = node.bar.volume
            else:
                prev_vma = bus.get(-2).vma
                vma.money = prev_vma.money + node.bar.money
                vma.volume = prev_vma.volume + node.bar.volume
            vma.value = round(vma.money / vma.volume / 100, 4)

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
        def first(node: Node, price: float):
            curr_ema = node.ema
            curr_ema.ema05 = price
            curr_ema.ema10 = price
            curr_ema.ema20 = price
            curr_ema.ema30 = price

        @staticmethod
        def next(bus: Bus, node: Node, price: float):
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
        def _first(node: Node, price: float):
            curr_macd = node.macd
            curr_macd.fast = price
            curr_macd.slow = price
            curr_macd.ema_ = price
            curr_macd.dif_ = 0.0
            curr_macd.dea_ = 0.0
            curr_macd.macd = 0.0

        @staticmethod
        def _next(bus: Bus, node: Node, price: float):
            curr_macd = node.macd
            prev_node = bus.get(-2)
            prev_macd = prev_node.macd
            base_price = bus.conf.get('base_price', node.ema.ema05)
            curr_macd.fast = Line.Ema.ema(prev_macd.fast, price, bus.conf.get('macd_fast', 12))
            curr_macd.slow = Line.Ema.ema(prev_macd.slow, price, bus.conf.get('macd_slow', 26))
            curr_macd.dif_ = round((curr_macd.fast - curr_macd.slow) / base_price * 100, 5)
            curr_macd.dea_ = Line.Ema.ema(prev_macd.dea_, curr_macd.dif_, bus.conf.get('macd_sign', 9))
            curr_macd.macd = round((curr_macd.dif_ - curr_macd.dea_) * 2, 4)


class Flow:
    class Step:
        @staticmethod
        def eval(ctx: Ctx):
            cur_time = ctx.node.bar.datetime.time()
            if cur_time <= Var.head_data_time:
                return Flow.HeadStep.eval(ctx)
            return Flow.MainStep.eval(ctx)

    class HeadStep(Step):
        @staticmethod
        def eval(ctx: Ctx):
            if ctx.sum_volume < 1.3 * ctx.sum_vol_05: return None, None
            if Flow.HeadStep._check_sell(ctx): return "sell", "SellStep"
            if Flow.HeadStep._check_buy(ctx): return "buy", "BuyStep"
            return None, None

        @staticmethod
        def _check_sell(ctx: Ctx) -> bool:
            if not ctx.is_allow_sell: return False
            if ctx.wave_pct >= -0.5: return False
            if ctx.ema_trend != 'fall': return False
            if ctx.macd_trend != 'fall': return False
            if ctx.node.ema.ema05 >= ctx.node.vma.value: return False
            return True

        @staticmethod
        def _check_buy(ctx: Ctx) -> bool:
            if not ctx.is_allow_buy: return False
            if ctx.wave_pct <= 0.5: return False
            if ctx.ema_trend != 'rise': return False
            if ctx.macd_trend != 'rise': return False
            if ctx.node.ema.ema05 <= ctx.node.vma.value: return False
            return True

    class MainStep(Step):
        @staticmethod
        def eval(ctx: Ctx):
            if ctx.sum_volume < 1.25 * ctx.sum_vol_20:
                if Flow.MainStep._check_lower_sell_01(ctx): return "sell", "SellStep"
                if Flow.MainStep._check_lower_sell_02(ctx): return "sell", "SellStep"
                if Flow.MainStep._check_lower_buy(ctx): return "buy", "BuyStep"
                return None, None
            if ctx.sum_volume >= 1.25 * ctx.sum_vol_20:
                if Flow.MainStep._check_upper_sell(ctx): return "sell", "SellStep"
                if Flow.MainStep._check_upper_buy(ctx): return "buy", "BuyStep"
            return None, None

        @staticmethod
        def _check_lower_sell_01(ctx: Ctx) -> bool:
            if not ctx.is_allow_sell: return False
            if ctx.curr_pct <= 1: return False
            if ctx.node.ema.ema05 >= ctx.orb_low: return False
            if ctx.node.ema.ema05 >= ctx.node.vma.value: return False
            if ctx.ema_trend != 'fall': return False
            if ctx.macd_trend != 'fall': return False
            return True

        @staticmethod
        def _check_lower_sell_02(ctx: Ctx) -> bool:
            if not ctx.is_allow_sell: return False
            if ctx.node.ema.ema05 >= ctx.node.vma.value: return False
            if ctx.node.ema.ema05 >= ctx.orb_low: return False
            if ctx.wave_pct >= -0.5: return False
            if ctx.ema_trend != 'fall': return False
            if ctx.macd_trend != 'fall': return False
            return True

        @staticmethod
        def _check_lower_buy(ctx: Ctx) -> bool:
            if not ctx.is_allow_buy: return False
            if ctx.node.ema.ema05 <= ctx.node.vma.value: return False
            if ctx.node.ema.ema05 <= ctx.orb_high: return False
            if ctx.wave_pct <= 0.5: return False
            if ctx.ema_trend != 'rise': return False
            if ctx.macd_trend != 'rise': return False
            return True

        @staticmethod
        def _check_upper_sell(ctx: Ctx) -> bool:
            if not ctx.is_allow_sell: return False
            if ctx.node.ema.ema05 >= ctx.node.vma.value: return False
            if ctx.node.ema.ema05 >= ctx.orb_low: return False
            if ctx.ema_trend != 'fall': return False
            if ctx.macd_trend != 'fall': return False
            return True

        @staticmethod
        def _check_upper_buy(ctx: Ctx) -> bool:
            if not ctx.is_allow_buy: return False
            if ctx.node.ema.ema05 <= ctx.orb_high: return False
            if ctx.node.ema.ema05 <= ctx.node.vma.value: return False
            if ctx.ema_trend != 'rise': return False
            if ctx.macd_trend != 'rise': return False
            return True

    class SellStep(Step):
        @staticmethod
        def eval(ctx: Ctx):
            if Flow.SellStep._check_buy_back_01(ctx): return "buy", "EndStep"
            if Flow.SellStep._check_buy_back_02(ctx): return "buy", "EndStep"
            return None, None

        @staticmethod
        def _check_buy_back_01(ctx: Ctx) -> bool:
            if not ctx.is_allow_buy: return False
            if ctx.ema_trend == 'fall': return False
            if ctx.macd_trend == 'fall': return False
            if ctx.node.ema.ema05 <= ctx.orb_high: return False
            if ctx.node.ema.ema05 <= ctx.node.vma.value: return False
            return True

        @staticmethod
        def _check_buy_back_02(ctx: Ctx) -> bool:
            if not ctx.is_allow_buy: return False
            if ctx.ema_trend == 'fall': return False
            if ctx.macd_trend == 'fall': return False
            if ctx.node.ema.ema05 <= ctx.keep_fall_price: return False
            return True

    class BuyStep(Step):
        @staticmethod
        def eval(ctx: Ctx):
            if Flow.BuyStep._check_sell_out_01(ctx): return "sell", "EndStep"
            if Flow.BuyStep._check_sell_out_02(ctx): return "sell", "EndStep"
            return None, None

        @staticmethod
        def _check_sell_out_01(ctx: Ctx) -> bool:
            if not ctx.is_allow_sell: return False
            if ctx.ema_trend == 'rise': return False
            if ctx.macd_trend == 'rise': return False
            if ctx.node.ema.ema05 >= ctx.orb_low: return False
            if ctx.node.ema.ema05 >= ctx.node.vma.value: return False
            return True

        @staticmethod
        def _check_sell_out_02(ctx: Ctx) -> bool:
            if not ctx.is_allow_sell: return False
            if ctx.ema_trend == 'rise': return False
            if ctx.macd_trend == 'rise': return False
            if ctx.node.ema.ema05 >= ctx.keep_rise_price: return False
            return True

    class EndStep(Step):
        @staticmethod
        def eval(ctx: Ctx):
            return None, None


class Rule:
    @staticmethod
    def ema_trend(ema: Box.Ema) -> str:
        """EMA趋势"""
        if ema.ema05 < ema.ema10 < ema.ema20 < ema.ema30: return 'fall'
        if ema.ema05 > ema.ema10 > ema.ema20 > ema.ema30: return 'rise'
        return 'flat'

    @staticmethod
    def macd_trend(macd: Box.Macd):
        """MACD趋势"""
        if macd.dif_ < 0 and macd.dea_ < 0: return 'fall'
        if macd.dif_ > 0 and macd.dea_ > 0: return 'rise'
        return 'flat'

    @staticmethod
    def is_allow_buy(market: Market) -> bool:
        """允许买入"""
        if market.ctx.has_buy: return False
        return market.dayBus.last().macd.macd > 0

    @staticmethod
    def is_allow_sell(market: Market) -> bool:
        """允许卖出"""
        if market.ctx.has_sell: return False
        return market.pos.avail_amount * market.pos.curr_price > 500

    @staticmethod
    def rise_risk_control(ctx: Ctx):
        """正T风险控制"""
        if ctx.has_sell: return
        if not ctx.has_buy: return
        if ctx.buy_price == 0.0: return
        rate = 0.995
        pct = round((ctx.curr_price - ctx.buy_price) / ctx.buy_price * 100, 2)
        for threshold, ratio in Var.rise_safe_lvls.items():
            if pct >= threshold:
                rate = ratio
                break
        ctx.keep_rise_price = max(ctx.keep_rise_price, round(ctx.buy_price * rate, 3))

    @staticmethod
    def fall_risk_control(ctx: Ctx):
        """反T风险控制"""
        if ctx.has_buy: return
        if not ctx.has_sell: return
        if ctx.sell_price == 0.0: return
        rate = 1.005
        pct = round((ctx.sell_price - ctx.curr_price) / ctx.sell_price * 100, 2)
        for threshold, ratio in Var.fall_safe_lvls.items():
            if pct >= threshold:
                rate = ratio
                break
        ctx.keep_fall_price = min(ctx.keep_fall_price, round(ctx.sell_price * rate, 3))


class Trader:
    def __init__(self, buy: Callable, sell: Callable):
        self.step = Flow.Step()
        self.sell = sell
        self.buy = buy

    def main_trading(self, market: Market):
        ctx = market.ctx
        if ctx.has_buy and ctx.has_sell: return
        action, step = self.step.eval(ctx)
        self._to_next(step)
        if action == 'buy':
            self._do_buy(market)
        if action == 'sell':
            self._do_sell(market)

    def tail_trading(self, market: Market):
        ctx = market.ctx
        if ctx.has_buy and ctx.is_allow_sell:
            self._do_sell(market)
            return
        if ctx.has_sell and ctx.is_allow_buy:
            self._do_buy(market)
            return

    def _to_next(self, next_step: str):
        if next_step == 'SellStep':
            self.step = Flow.SellStep()
            return
        if next_step == 'BuyStep':
            self.step = Flow.BuyStep()
            return
        if next_step == 'EndStep':
            self.step = Flow.EndStep()
            return

    def _do_buy(self, market: Market):
        pos = market.pos
        curr_price = pos.curr_price
        buy_amount = round(Var.base_fund / curr_price / 100) * 100
        self.buy(market.symbol, buy_amount, limit_price=curr_price + 0.003)
        market.ctx.has_buy = True
        market.ctx.buy_price = curr_price
        market.ctx.buy_amount = buy_amount

    def _do_sell(self, market: Market):
        pos = market.pos
        curr_price = pos.curr_price
        base_amount = round(Var.base_fund * 1.5 / curr_price / 100) * 100
        able_amount = min(pos.avail_amount, base_amount)
        sell_amount = able_amount - (0 if able_amount < pos.total_amount else 100)
        self.sell(market.symbol, -sell_amount, limit_price=curr_price - 0.003)
        market.ctx.has_sell = True
        market.ctx.sell_price = curr_price
        market.ctx.sell_amount = sell_amount


############################################################
class Ctx:
    def __init__(self):
        self.node = None  # 当前节点
        self.curr_time = None # 当前时间
        self.base_price = 0.0  # 昨日收盘价
        self.open_price = 0.0  # 开盘价格
        self.curr_price = 0.0  # 最新价格
        self.open_pct = 0.0  # 开盘价格（%）
        self.curr_pct = 0.0  # 最新价格（%）
        self.wave_pct = 0.0  # 相对开盘价的波动价格（%）

        self.ema_trend = ''  # EMA趋势
        self.macd_trend = ''  # MACD趋势
        self.is_allow_buy = False  # 是否允许买入
        self.is_allow_sell = False  # 是否允许卖出
        self.keep_rise_price = -1000  # 正T风控价格
        self.keep_fall_price = 1000  # 反T风控价格

        self.orb_low = 0.0  # 开盘区间内的最低价
        self.orb_high = 0.0  # 开盘区间内的最高价
        self.sum_volume = 0.0  # 开盘区间内的成交量之和
        self.sum_vol_05 = 0.0  # 前5日每日前8分钟的成交量之和
        self.sum_vol_20 = 0.0  # 前5日每日前20分钟的成交量之和

        self.has_buy = False  # 是否已经买入
        self.has_sell = False  # 是否已经卖出
        self.buy_price = 0.0  # 买入时的价格
        self.sell_price = 0.0  # 卖出时的价格
        self.buy_amount = 0.0  # 买入时的数量
        self.sell_amount = 0.0  # 卖出时的数量


class Market:
    def __init__(self, symbol: str):
        self.dayBus = Bus(maxlen=120)  # 日线数据
        self.fenBus = Bus(maxlen=240, conf=Var.fen)  # 分钟数据
        self.symbol = symbol  # 股票代码
        self.ctx = Ctx()  # 上下文数据
        self.pos = None  # 当前持仓

    def prep(self, bars: list):
        if not bars:
            return self
        for bar in bars:
            self.dayBus.add(Node(bar))
            Line.Ema.calc(self.dayBus)
            Line.Macd.calc(self.dayBus)
        self.ctx.base_price = self.dayBus.last().bar.close
        self.fenBus.conf['base_price'] = self.ctx.base_price
        return self

    def next(self, pos, bar):
        # 分钟数据
        node = Node(bar)
        self.fenBus.add(node)
        Line.Vma.calc(self.fenBus)
        Line.Ema.calc(self.fenBus)
        Line.Macd.calc(self.fenBus)

        # 日线数据
        self.dayBus.rollback()
        self.dayBus.add(Node(bar).mark(-1))
        Line.Ema.calc(self.dayBus)
        Line.Macd.calc(self.dayBus)

        # 更新上下文数据
        self.pos = Bin.Pos(pos)
        self.update_ctx()

    def update_ctx(self):
        node = self.fenBus.last()
        base_price = self.ctx.base_price
        curr_price = self.pos.curr_price
        self.ctx.node = node
        if len(self.fenBus) == 1:
            self.ctx.open_price = node.bar.open
            self.ctx.open_pct = round((self.ctx.open_price - base_price) / base_price * 100, 2)
            self.ctx.orb_high = curr_price
            self.ctx.orb_low = curr_price
        if node.bar.datetime.time() <= Var.head_data_time:
            self.ctx.orb_low = min(self.ctx.orb_low, curr_price)
            self.ctx.orb_high = max(self.ctx.orb_high, curr_price)
            self.ctx.sum_volume = self.ctx.sum_volume + node.bar.volume
        self.ctx.curr_time = node.bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.ctx.curr_price = curr_price
        self.ctx.curr_pct = round((curr_price - base_price) / base_price * 100, 2)
        self.ctx.wave_pct = round((curr_price - self.ctx.open_price) / base_price * 100, 2)
        self.ctx.ema_trend = Rule.ema_trend(node.ema)
        self.ctx.macd_trend = Rule.macd_trend(node.macd)
        self.ctx.is_allow_buy = Rule.is_allow_buy(self)
        self.ctx.is_allow_sell = Rule.is_allow_sell(self)
        Rule.rise_risk_control(self.ctx)
        Rule.fall_risk_control(self.ctx)


############################################################
class Env:
    indexes: list[str] = ['000001.SS', '000852.SS']
    symbols: list[str] = []
    markets: dict[str, Market] = {}
    traders: dict[str, Trader] = {}

    @staticmethod
    def market(symbol: str):
        return Env.markets.get(symbol)

    @staticmethod
    def trader(symbol: str):
        trader = Env.traders.get(symbol)
        if trader is None:
            trader = Trader(order, order)
            Env.traders[symbol] = trader
        return trader

    @staticmethod
    def launch(context):
        Env.clear()
        positions = context.portfolio.positions
        pos_codes = list(positions.keys())
        Env.symbols = pos_codes.copy()
        set_universe(pos_codes)
        history_day = get_history(60, frequency='1d', security_list=pos_codes)
        for code in pos_codes:
            bars = Env.parse_bars(history_day, code)
            Env.markets[code] = Market(code).prep(bars)

        history_fen = get_history(960, frequency='5m', security_list=pos_codes)
        for code in pos_codes:
            ctx = Env.markets[code].ctx
            data = history_fen.query(f'code in ["{code}"]')
            ctx.sum_vol_05 = (
                data.assign(date=data.index.date)
                .groupby('date').head(1)
                .groupby('date')['volume'].sum()
                .tail(17).mean()
            )
            ctx.sum_vol_20 = (
                data.assign(date=data.index.date)
                .groupby('date').head(4)
                .groupby('date')['volume'].sum()
                .tail(17).mean()
            )

    @staticmethod
    def parse_bars(history, symbol):
        data = history.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in data.iterrows()]
        return bars

    @staticmethod
    def clear():
        Env.symbols.clear()
        Env.markets.clear()
        Env.traders.clear()


############################################################
def initialize(context):
    """启动时执行一次"""
    if is_trade(): return
    set_commission(commission_ratio=0.00005, min_commission=0.5, type="ETF")

    # 设置底仓
    pos = {}
    pos['sid'] = "588760.SS"
    pos['sid'] = "159995.SZ"
    pos['amount'] = "100"
    pos['enable_amount'] = "100"
    pos['cost_basis'] = "1.0"
    set_yesterday_position([pos])
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    Env.launch(context)
    pass


def handle_data(context, data):
    """每个单位周期执行一次"""
    cur_time = context.blotter.current_dt.time()
    positions = context.portfolio.positions
    for symbol in Env.symbols:
        bar = data[symbol]
        pos = positions.get(symbol)
        market = Env.market(symbol)
        trader = Env.trader(symbol)
        market.next(pos, bar)
        if cur_time <= Var.open_data_time:
            continue
        if cur_time <= Var.main_eval_time:
            trader.main_trading(market)
            continue
        if cur_time > Var.tail_eval_time:
            trader.tail_trading(market)


def tick_data(context, data):
    """每个tick执行一次"""
    pass


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    Env.clear()
    pass
