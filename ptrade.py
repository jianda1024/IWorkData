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

from abc import ABC, abstractmethod
from datetime import time
from types import SimpleNamespace
from typing import Self, Callable


class Config:
    class _Pos:
        def __init__(self):
            self.base_principal = 8000  # 基础资金
            self.cost_limit = 1.50  # 成本上限（比例）
            self.loss_limit = 0.15  # 亏损上限（比例）
            self.gain_limit = 0.05  # 盈利上限（比例）

    class _Sma:
        def __init__(self):
            self.fast = 10  # 快线周期
            self.slow = 30  # 慢线周期

    class _Macd:
        def __init__(self):
            self.fast = 12  # 快线周期
            self.slow = 26  # 慢线周期
            self.sign = 9  # 信号线周期

    class _Rise:
        def __init__(self):
            self.add_quotas = [0.300]  # 加仓额度（比例）
            self.thresholds = [0.003]  # 加仓阈值（比例）
            self.early_decline = 0.01  # 前期跌幅（比例）

    class _Fall:
        def __init__(self):
            self.sub_quotas = [0.300, 0.400, 0.300]  # 减仓额度（比例）
            self.thresholds = [0.003, 0.006, 0.009]  # 减仓阈值（比例）

    class _Wave:
        def __init__(self):
            self.min_swing = 0.003  # 最小摆动（比例）

    def __init__(self):
        self.Pos: Config._Pos = Config._Pos()
        self.Sma: Config._Sma = Config._Sma()
        self.Macd: Config._Macd = Config._Macd()
        self.Rise: Config._Rise = Config._Rise()
        self.Fall: Config._Fall = Config._Fall()
        self.Wave: Config._Wave = Config._Wave()

    def shift(self, level: str = 'default'):
        match level:
            case 'default':
                self.Sma.fast = 10
                self.Sma.slow = 30
            case 'day_bar':
                self.Sma.fast = 5
                self.Sma.slow = 10


class NodeBar:
    class _Bar:
        def __init__(self, bar):
            self.datetime = bar.datetime  # 时间
            self.volume = round(bar.volume, 2)  # 交易量
            self.money = round(bar.money, 2)  # 交易金额
            self.price = round(bar.price, 4)  # 最新价
            self.close = round(bar.close, 4)  # 收盘价
            self.open = round(bar.open, 4)  # 开盘价
            self.high = round(bar.high, 4)  # 最高价
            self.low = round(bar.low, 4)  # 最低价

    class _Sma:
        def __init__(self):
            self.fast = 0.0  # SMA快线
            self.slow = 0.0  # SMA慢线

        def first(self, bar: NodeBar._Bar):
            self.fast = round(bar.close, 4)
            self.slow = round(bar.close, 4)
            return self

        def next(self, bar: NodeBar._Bar, pre_sma: Self):
            fast = (pre_sma.fast * (CFG.Sma.fast - 1) + bar.close) / CFG.Sma.fast
            slow = (pre_sma.slow * (CFG.Sma.slow - 1) + bar.close) / CFG.Sma.slow
            self.fast = round(fast, 4)
            self.slow = round(slow, 4)
            return self

        def is_rise(self):
            """均线上涨"""
            return self.fast >= self.slow

        def is_fall(self):
            """均线下跌"""
            return self.fast <= self.slow

    class _Macd:
        def __init__(self):
            self.ema_fast = 0.0
            self.ema_slow = 0.0
            self.dif = 0.0
            self.dea = 0.0
            self.macd = 0.0

        def first(self, bar: NodeBar._Bar):
            self.ema_fast = round(bar.close, 4)
            self.ema_slow = round(bar.close, 4)
            return self

        def next(self, bar: NodeBar._Bar, pre_macd: Self):
            self.ema_fast = self._ema(bar.close, CFG.Macd.fast, pre_macd.ema_fast)
            self.ema_slow = self._ema(bar.close, CFG.Macd.slow, pre_macd.ema_slow)
            self.dif = round(self.ema_fast - self.ema_slow, 4)
            self.dea = self._ema(self.dif, CFG.Macd.sign, pre_macd.dea)
            self.macd = round((self.dif - self.dea) * 2, 4)
            return self

        @staticmethod
        def _ema(price, period, pre_ema):
            alpha = 2 / (period + 1)
            ema = alpha * price + (1 - alpha) * pre_ema
            return round(ema, 4)

    class _Mark:
        def __init__(self):
            # 涨跌等级，标记在节点上，用于避免重复计算
            self.rise_level = -2
            self.fall_level = -2

    def __init__(self, bar):
        self.Bar: NodeBar._Bar = NodeBar._Bar(bar)
        self.Sma: NodeBar._Sma = NodeBar._Sma()
        self.Macd: NodeBar._Macd = NodeBar._Macd()
        self.Mark: NodeBar._Mark = NodeBar._Mark()

        self.datetime = self.Bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.price = self.Bar.price
        self.turn_sma = 0.0
        self.turn_val = 0

    def first(self) -> Self:
        """初始节点，NodeBar"""
        self.Sma.first(self.Bar)
        self.Macd.first(self.Bar)
        self.turn_sma = self.Sma.fast
        self.turn_val = -2
        return self

    def next(self, pre_node: Self) -> Self:
        """下一个节点，NodeBar"""
        self.Sma.next(self.Bar, pre_node.Sma)
        self.Macd.next(self.Bar, pre_node.Macd)
        self.turn_sma = self.Sma.fast
        return self

    def turn(self):
        """转换为拐点，TurnBar"""
        return TurnBar(self)


class TurnBar:
    class _Mark:
        def __init__(self):
            # 涨跌批次，标记在拐点上，用于避免重复交易
            self.rise_lots = CFG.Rise.add_quotas.copy()
            self.fall_lots = CFG.Fall.sub_quotas.copy()

    def __init__(self, node: NodeBar):
        self.Mark: TurnBar._Mark = TurnBar._Mark()
        self.datetime = node.datetime
        self.turn_sma = node.turn_sma
        self.turn_val = node.turn_val
        self._max_node: NodeBar | None = None
        self._min_node: NodeBar | None = None

    def max_node(self, node: NodeBar, threshold):
        """到下一拐点前：最大的凸点"""
        if node.turn_sma - self.turn_sma > threshold:
            if self._max_node is None or node.turn_sma > self._max_node.turn_sma:
                self._max_node = node

    def min_node(self, node: NodeBar, threshold):
        """到下一拐点前：最小的凹点"""
        if self.turn_sma - node.turn_sma > threshold:
            if self._min_node is None or node.turn_sma < self._min_node.turn_sma:
                self._min_node = node

    def next_turn(self, node: NodeBar, threshold):
        """下一拐点"""
        if self.turn_val == -2:
            diff = round(self.turn_sma - node.turn_sma, 4)
            if abs(diff) > threshold:
                self.turn_val = 1 if diff > 0 else -1
            return None
        if self._max_node and self._max_node.turn_sma - node.turn_sma > threshold:
            return self._max_node.turn()
        if self._min_node and node.turn_sma - self._min_node.turn_sma > threshold:
            return self._min_node.turn()
        return None


class Handler(ABC):
    @abstractmethod
    def hub(self) -> DataHub:
        """数据集"""
        pass

    @abstractmethod
    def bar(self) -> tuple[NodeBar, NodeBar, TurnBar]:
        """最新日线节点、最新分钟线节点、最新分钟线拐点"""
        pass

    def trade(self, func: Callable):
        """执行交易"""
        if self.__is_buy():
            self.__do_buy(func)
            return
        if self.__is_sell():
            self.__do_sell(func)
            return

    def __is_buy(self):
        """是否买入"""
        yest, node, turn = self.bar()
        # 非波谷
        if turn.turn_val != -1:
            return False
        # 前10分钟不买入
        if node.Bar.datetime.time() < time(9, 40, 0):
            return False
        # 超过亏损上限 or 超过本金上限
        if self.hub().CurrPos.over_limit():
            return False
        # 日线下跌 or 分钟线下跌
        if yest.Sma.is_fall() or node.Sma.is_fall():
            return False
        # 涨幅未达到阈值 or 重复操作
        level = self.__rise_level()
        if level == -1 or turn.Mark.rise_lots[level] == 0:
            return False
        # 只是小幅下跌
        # if self.__is_light_fall():
            return False

        # 决定买入
        return True

    def __is_sell(self):
        """是否卖出"""
        yest, node, turn = self.bar()
        # 非波峰
        if turn.turn_val != 1:
            return False
        # 没有可用持仓
        if self.hub().CurrPos.has_no_amount():
            return False
        # 分钟线上涨
        if node.Sma.is_rise():
            return False
        # 跌幅未达到阈值 or 重复操作
        level = self.__fall_level()
        if level == -1 or turn.Mark.fall_lots[level] == 0:
            return False
        # 决定卖出
        return True

    def __do_buy(self, func: Callable):
        """执行买入"""
        _, _, turn = self.bar()
        level = self.__rise_level()
        lots = turn.Mark.rise_lots
        buy_amount = CFG.Pos.base_principal / self.hub().base_price * lots[level]
        amount = round(buy_amount / 100) * 100

        # 执行买入
        func(self.hub().code, amount)
        lots[level] = 0.0
        # del self.hub().HereBus.turns[:-1]

    def __do_sell(self, func: Callable):
        """执行卖出"""
        _, _, turn = self.bar()
        level = self.__fall_level()
        lots = turn.Mark.fall_lots

        # 最小数量：根据基准本金
        min_qty = CFG.Pos.base_principal / self.hub().base_price * lots[level]
        # 减仓数量：根据当日初始可用持仓
        cur_qty = self.hub().base_amount * lots[level]
        # 避免低仓位时，还分多次减仓
        sell_qty = max(min_qty, cur_qty)
        # 不得超过当前可用持仓
        sell_amount = min(sell_qty, self.hub().CurrPos.avail_amount)
        # 调整到100的倍数，并留下底仓
        amount = round(sell_amount / 100) * 100 - self.hub().CurrPos.remain_amount()

        # 执行卖出
        func(self.hub().code, -amount)
        lots[level] = 0.0

    def __rise_level(self):
        """当前上涨等级"""
        _, node, _ = self.bar()
        mark = node.Mark
        if mark.rise_level == -2:
            mark.rise_level = self.__calc_level(CFG.Rise.thresholds)
        return mark.rise_level

    def __fall_level(self):
        """当前下跌等级"""
        _, node, _ = self.bar()
        mark = node.Mark
        if mark.fall_level == -2:
            mark.fall_level = self.__calc_level(CFG.Fall.thresholds)
        return mark.fall_level

    def __calc_level(self, thresholds):
        """计算涨跌等级"""
        _, node, turn = self.bar()
        diff_value = abs(node.turn_sma - turn.turn_sma)
        diff_ratio = round(diff_value / self.hub().base_price, 4)
        for threshold in reversed(thresholds):
            if diff_ratio > threshold:
                return thresholds.index(threshold)
        return -1

    def __is_light_fall(self):
        """是否只是小幅下跌"""
        turns = self.hub().HereBus.turns
        if len(turns) < 3:
            return False
        max_sma = max(turn.turn_sma for turn in turns)
        if max_sma - turns[-1].turn_sma < self.hub().base_price * CFG.Rise.early_decline:
            return True
        return False


class DataHub(Handler):
    class _AwayBus:
        def __init__(self):
            self.nodes: list[NodeBar] = []

        def prepare(self, bars):
            CFG.shift("day_bar")
            self.nodes.clear()
            self.nodes.append(NodeBar(bars[0]).first())
            for bar in bars[1:]:
                node = NodeBar(bar).next(self.nodes[-1])
                self.nodes.append(node)
            CFG.shift()

    class _HereBus:
        def __init__(self):
            self.nodes: list[NodeBar] = []
            self.turns: list[TurnBar] = []
            self.base_price = 0.0

        def prepare(self, base_price):
            self.base_price = base_price
            self.nodes.clear()
            self.turns.clear()

        def first(self, bar):
            node = NodeBar(bar).first()
            turn = node.turn()
            self.nodes.append(node)
            self.turns.append(turn)

        def next(self, bar):
            node = NodeBar(bar).next(self.nodes[-1])
            if node.turn_sma == self.nodes[-1].turn_sma:
                return
            self.nodes.append(node)
            if len(self.nodes) < 3:
                return

            # 最小振幅价格差
            threshold = round(self.base_price * CFG.Wave.min_swing, 4)

            # 计算凹凸点
            prev = self.nodes[-3]
            node = self.nodes[-2]
            post = self.nodes[-1]
            if prev.turn_sma < node.turn_sma > post.turn_sma:
                node.turn_val = 1
                self.turns[-1].max_node(node, threshold)
            elif prev.turn_sma > node.turn_sma < post.turn_sma:
                node.turn_val = -1
                self.turns[-1].min_node(node, threshold)

            # 计算拐点
            turn = self.turns[-1].next_turn(post, threshold)
            self.turns.append(turn) if turn else None

    class _CurrPos:
        def __init__(self):
            self.avail_amount = 0.0  # 可用持仓数量
            self.total_amount = 0.0  # 总持仓数量
            self.cost_price = 0.0  # 成本价格
            self.last_price = 0.0  # 最新价格
            self.valuation = 0.0  # 市值
            self.principal = 0.0  # 本金

        def update(self, pos, price_last=0.0):
            self.avail_amount = getattr(pos, 'enable_amount', 0.0)
            self.total_amount = getattr(pos, 'amount', 0.0)
            self.cost_price = getattr(pos, 'cost_basis', 0.0)
            self.last_price = getattr(pos, 'last_sale_price', price_last)
            self.valuation = round(self.total_amount * self.last_price, 2)
            self.principal = round(self.total_amount * self.cost_price, 2)

        def over_limit(self) -> bool:
            """超过亏损上限 or 超过本金上限"""
            if self.principal - self.valuation >= CFG.Pos.base_principal * CFG.Pos.loss_limit:
                return True
            return self.principal > CFG.Pos.base_principal * CFG.Pos.cost_limit

        def has_no_amount(self) -> bool:
            """没有可用持仓"""
            return self.avail_amount <= self.remain_amount()

        def remain_amount(self) -> int:
            """获取保留的股票数量"""
            if self.total_amount > self.avail_amount:
                return 0
            if self.valuation - self.principal > CFG.Pos.base_principal * CFG.Pos.gain_limit:
                return 0
            return 100

    def __init__(self, code):
        self.AwayBus: DataHub._AwayBus = DataHub._AwayBus()  # 历史数据
        self.HereBus: DataHub._HereBus = DataHub._HereBus()  # 当天数据
        self.CurrPos: DataHub._CurrPos = DataHub._CurrPos()  # 当前仓位

        self.code = code  # 股票代码
        self.base_price = 0.0  # 基准价格
        self.base_amount = 0.0  # 基准持仓
        self._status = 0  # 状态：0初始、1启动、2就绪

    def prepare(self, bars):
        self.base_price = round(bars[-1].close, 4)
        self.AwayBus.prepare(bars)
        self.HereBus.prepare(self.base_price)
        self._status = 1
        return self

    def running(self, bar, pos):
        self.CurrPos.update(pos)
        if self._status == 2:
            self.HereBus.next(bar)
            return self
        if self._status == 1:
            self.base_amount = self.CurrPos.avail_amount
            self.HereBus.first(bar)
            self._status = 2
            return self
        if self._status == 0:
            self.prepare([bar])
        return self

    def hub(self) -> DataHub:
        return self

    def bar(self) -> tuple[NodeBar, NodeBar, TurnBar]:
        return self.AwayBus.nodes[-1], self.HereBus.nodes[-1], self.HereBus.turns[-1]


############################################################
class StockMarket:
    def __init__(self):
        self.hubs: dict[str, DataHub] = {}

    def prepare(self, code, bars):
        self.hubs[code] = DataHub(code).prepare(bars)

    def running(self, code, bar, pos):
        hub = self.hubs.get(code).running(bar, pos)
        hub.trade(order)

    def delete(self, symbol):
        del self.hubs[symbol]

    def clear(self):
        self.hubs.clear()


"""
########################################################################################################################
import logging
from ptrade import GlobalVars

log = logging.getLogger()
g = GlobalVars()


def set_universe(symbol):
    g.symbol = symbol.replace('SS', 'SH')
    pass


def order(symbol, amount):
    pass


def get_history(**kwargs):
    pass
def get_positions():
    pass


########################################################################################################################
"""

# 全局配置
CFG = Config()


def initialize(context):
    """启动时执行一次"""
    g.symbols = ['159857.SZ']
    set_universe(g.symbols)
    g.market = StockMarket()
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    his_data = get_history(60, frequency='1d')
    for symbol in g.symbols:
        df = his_data.query(f'code in ["{symbol}"]')
        bars = [SimpleNamespace(datetime=idx, **row.to_dict()) for idx, row in df.iterrows()]
        g.market.prepare(symbol, bars)


def handle_data(context, data):
    """每个单位周期执行一次"""
    positions = context.portfolio.positions
    for symbol in g.symbols:
        pos = positions.get(symbol)
        bar = data[symbol]
        g.market.running(symbol, bar, pos)


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
