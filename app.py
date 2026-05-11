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

from types import SimpleNamespace
from typing import Callable, Any

import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression


class Var:
    base_fund = 3000  # 交易基础金额（元）
    tail_time = "14:40:00"

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

        def agg(self, bar: Bin.Bar):
            if bar is None:
                return self
            self.volume += bar.volume
            self.money += bar.money
            self.open = bar.open
            self.high = max(self.high, bar.high)
            self.low = min(self.low, bar.low)
            return self

        def to_dict(self) -> dict:
            return {
                "volume": self.volume,
                "money": self.money,
                "price": self.price,
                "close": self.close,
                "open": self.open,
                "high": self.high,
                "low": self.low,
            }

    class Node:
        def __init__(self):
            pass


class Box:
    class Adx:
        def __init__(self):
            self.di_bull: float = 0.0
            self.di_bear: float = 0.0
            self.adx: float = 0.0
            self.atr: float = 0.0

    class Sma:
        def __init__(self):
            self.ma05: float = 0.0
            self.ma10: float = 0.0
            self.ma20: float = 0.0

    class Macd:
        def __init__(self):
            self.dif: float = 0.0
            self.dea: float = 0.0
            self.macd: float = 0.0

    class Boll:
        def __init__(self):
            self.mid = 0.0  # 中轨
            self.upper = 0.0  # 上轨
            self.lower = 0.0  # 下轨

    class Rsrs:
        def __init__(self):
            self.beta = 0.0  # 原始斜率
            self.zscore = 0.0  # 标准化后的强度


class Line:
    class Adx:
        @staticmethod
        def calculate(df: DataFrame, node: Bin.Node):
            # 前一日收盘价
            prev_close = df['close'].shift(1)

            # 计算 TR（真实波幅）
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - prev_close).abs()
            tr3 = (df['low'] - prev_close).abs()
            tr = np.maximum.reduce([tr1, tr2, tr3])

            # 计算 +DM / -DM（方向动量）
            up_move = df['high'] - df['high'].shift(1)
            dn_move = df['low'].shift(1) - df['low']
            dm_bull = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
            dm_bear = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)

            # 平滑处理
            period = 14
            tr_smooth = Line.Adx.smooth(tr, period)
            dm_bull_smooth = Line.Adx.smooth(dm_bull, period)
            dm_bear_smooth = Line.Adx.smooth(dm_bear, period)

            # 计算 +DI / -DI
            di_bull = (dm_bull_smooth / tr_smooth) * 100
            di_bear = (dm_bear_smooth / tr_smooth) * 100

            # 计算 DX → ADX
            dx = abs(di_bull - di_bear) / (di_bull + di_bear) * 100
            adx = Line.Adx.smooth(dx, period)

            node.adx = Box.Adx()
            node.adx.di_bull = di_bull.iloc[-1]
            node.adx.di_bear = di_bear.iloc[-1]
            node.adx.adx = adx.iloc[-1]
            node.adx.atr = tr_smooth.iloc[-1]

        @staticmethod
        def smooth(series: Series, window):
            return series.rolling(window=window, min_periods=1).mean()

    class Sma:
        @staticmethod
        def calculate(df: DataFrame, node: Bin.Node):
            ma05 = Line.Sma.smooth(df['close'], 5)
            ma10 = Line.Sma.smooth(df['close'], 10)
            ma20 = Line.Sma.smooth(df['close'], 20)
            node.sma = Box.Sma()
            node.sma.ma05 = ma05.iloc[-1]
            node.sma.ma10 = ma10.iloc[-1]
            node.sma.ma20 = ma20.iloc[-1]

        @staticmethod
        def smooth(series: pd.Series, window: int) -> pd.Series:
            return series.rolling(window=window, min_periods=1).mean()

    class Macd:
        @staticmethod
        def calculate(df: DataFrame, node: Bin.Node):
            period_fast = 12
            period_slow = 26
            period_sign = 9

            close = df["close"]
            ema_fast = Line.Macd.smooth(close, period_fast)
            ema_slow = Line.Macd.smooth(close, period_slow)
            dif = ema_fast - ema_slow
            dea = Line.Macd.smooth(dif, period_sign)
            macd = (dif - dea) * 2

            node.macd = Box.Macd()
            node.macd.dif = dif.iloc[-1]
            node.macd.dea = dea.iloc[-1]
            node.macd.macd = macd.iloc[-1]

        @staticmethod
        def smooth(series: pd.Series, span: int) -> pd.Series:
            return series.ewm(span=span, adjust=False).mean()

    class Boll:
        @staticmethod
        def calculate(df: DataFrame, node: Bin.Node):
            period_boll = 20  # 中轨 SMA 周期
            multiplier = 2  # 带宽倍数（默认 2倍 ATR）

            low = df["low"]
            high = df["high"]
            close = df["close"]
            prev_close = close.shift(1)

            # 1. 计算 TR、ATR
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.Series(np.maximum.reduce([tr1, tr2, tr3]), index=df.index)
            atr = Line.Sma.smooth(tr, period_boll)

            # 2. 中轨
            boll_mid = Line.Sma.smooth(close, period_boll)

            # 3. 上下轨
            boll_upper = boll_mid + multiplier * atr
            boll_lower = boll_mid - multiplier * atr

            # 赋值到 node
            node.boll = Box.Boll()
            node.boll.mid = boll_mid.iloc[-1]
            node.boll.upper = boll_upper.iloc[-1]
            node.boll.lower = boll_lower.iloc[-1]

    class Rsrs:
        @staticmethod
        def calculate(df: DataFrame, node: Bin.Node):
            beta_window = 18  # 回归周期
            zscore_window = 600  # 标准化周期

            # 1. 计算 beta
            low = df["low"]
            high = df["high"]
            beta = Line.Rsrs._calc_beta(high, low, beta_window)

            # 2. 标准化 zscore
            mean = beta.rolling(window=zscore_window, min_periods=1).mean()
            std = beta.rolling(window=zscore_window, min_periods=1).std()
            zscore = (beta - mean) / std

            # 3. 赋值给 node
            node.rsrs = Box.Rsrs()
            node.rsrs.beta = beta.iloc[-1]
            node.rsrs.zscore = zscore.iloc[-1]

        @staticmethod
        def _calc_beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
            """ 滚动窗口计算 RSRS 斜率 """
            beta_list = []
            for i in range(len(high)):
                if i < window - 1:
                    beta_list.append(np.nan)
                    continue

                h = high.iloc[i - window + 1: i + 1].values
                l = low.iloc[i - window + 1: i + 1].values
                beta = Line.Rsrs._regress(l, h)
                beta_list.append(beta)

            return pd.Series(beta_list, index=high.index)

        @staticmethod
        def _regress(x: np.ndarray, y: np.ndarray) -> float:
            """线性回归，返回斜率 beta"""
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)
            return float(model.coef_[0])


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
class Ctx:
    def __init__(self):
        self.act = Bin.Act()  # 操作记录
        self.bar = None  # 当前bar
        self.pos = None  # 当前持仓
        self.node = None  # 当前节点
        self.is_allow_buy = False  # 是否允许买入
        self.is_allow_sell = False  # 是否允许卖出


class Market:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 股票代码
        self.time = None  # 当前时间
        self.data = None  # 日线数据
        self.ctx = Ctx()  # 上下文数据

    def prep(self, df: pd.DataFrame):
        self.data = df.copy()
        return self

    def running(self, pos, bar):
        self.time = bar.datetime.strftime("%Y-%m-%d %H:%M:%S")
        self.ctx.bar = Bin.Bar(bar).agg(self.ctx.bar)
        self.ctx.pos = Bin.Pos(pos)
        self.ctx.node = Bin.Node()

        df = self.data.copy()
        curr_bar = self.ctx.bar
        df.loc[curr_bar.datetime] = curr_bar.to_dict()
        Line.Adx.calculate(df, self.ctx.node)
        Line.Sma.calculate(df, self.ctx.node)
        Line.Macd.calculate(df, self.ctx.node)
        Line.Boll.calculate(df, self.ctx.node)
        Line.Rsrs.calculate(df, self.ctx.node)

    def trading(self):
        if not self.ctx.is_allow_buy and not self.ctx.is_allow_sell:
            return
        curr_time = self.ctx.node.time
        if curr_time <= Var.wait_time:
            return
        if curr_time <= Var.main_trade_time:
            self.trader.main_trading(self.ctx)
            return
        if curr_time == "14:55:00":
            self.trader.tail_trading(self.ctx)


############################################################
class Env:
    markets: dict[str, Market] = {}

    @staticmethod
    def launch(context):
        Env.markets.clear()
        positions = context.portfolio.positions
        pos_codes = list(positions.keys())
        set_universe(pos_codes)
        histories = get_history(60, frequency='1d', security_list=pos_codes)
        for code in pos_codes:
            df = histories.query(f'code in ["{code}"]')
            Env.markets[code] = Market(code).prep(df)

    @staticmethod
    def avg_vol(data, num):
        return (
            data.assign(date=data.index.date)
            .groupby('date').head(num)
            .groupby('date')['volume'].sum()
            .tail(20).mean()
        )

    @staticmethod
    def parse_bars(history, symbol):
        data = history.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in data.iterrows()]
        return bars

    @staticmethod
    def to_dict(obj):
        """递归将对象转换为字典"""
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


############################################################
def initialize(context):
    """启动时执行一次"""
    if is_trade(): return
    set_commission(commission_ratio=0.00005, min_commission=0.5, type="ETF")

    # 设置底仓
    pos = {}
    pos['sid'] = "515650.SS"
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
    positions = context.portfolio.positions
    for symbol, market in Env.markets.items():
        bar = data[symbol]
        pos = positions.get(symbol)
        market.running(pos, bar)
        market.trading()


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
