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
from typing import Self, Callable


########################################################################################################################

class Vary:
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class PosConfig(Vary):
    def __init__(self, **kwargs):
        self.base_capital = 10000  # 基础额度
        self.invest_limit = 3  # 投资上限（比例）
        self.profit_target = 0.03  # 盈利目标（比例）
        self.update(**kwargs)


class SmaConfig(Vary):
    def __init__(self, **kwargs):
        self.fast = 10  # 快线周期
        self.slow = 30  # 慢线周期
        self.update(**kwargs)


class MACDConfig(Vary):
    def __init__(self, **kwargs):
        self.fast = 12  # 快线周期
        self.slow = 26  # 慢线周期
        self.sign = 9  # 信号线周期
        self.update(**kwargs)


class TradeConfig(Vary):
    def __init__(self, **kwargs):
        self.amp_lower_limit = 0.004  # 振幅下限（比例）
        self.rise_add_quotas = [0.250, 0.250]  # 加仓额度（比例）
        self.rise_thresholds = [0.004, 0.006]  # 加仓阈值（比例）
        self.fall_sub_quotas = [0.200, 0.300, 0.300, 0.200]  # 减仓额度（比例）
        self.fall_thresholds = [0.004, 0.006, 0.009, 0.012]  # 减仓阈值（比例）
        self.update(**kwargs)


class StockConfig(Vary):
    def __init__(self, **kwargs):
        self.pos = PosConfig()
        self.sma = SmaConfig()
        self.macd = MACDConfig()
        self.trade = TradeConfig()
        self.update(**kwargs)

    def shift(self, level: str = 'default'):
        match level:
            case 'default':
                self.sma = SmaConfig()
            case 'day_bar':
                self.sma = SmaConfig(fast=10, slow=60)


########################################################################################################################

class Pos:
    def __init__(self, pos, price_last: float = 0.0):
        self.avail_amount = getattr(pos, 'enable_amount', 0.0)  # 可用持仓数量
        self.total_amount = getattr(pos, 'amount', 0.0)  # 总持仓数量
        self.cost_price = getattr(pos, 'cost_basis', 0.0)  # 成本价格
        self.last_price = getattr(pos, 'last_sale_price', price_last)  # 最新价格
        self.valuation = self.total_amount * self.last_price  # 持仓市值


class Bar:
    def __init__(self, bar):
        self.datetime = bar.datetime  # 时间
        self.price = round(bar.price, 4)  # 最新价
        self.close = round(bar.close, 4)  # 收盘价
        self.open = round(bar.open, 4)  # 开盘价
        self.high = round(bar.high, 4)  # 最高价
        self.low = round(bar.low, 4)  # 最低价
        self.money = round(bar.money, 2)  # 交易金额
        self.volume = round(bar.volume)  # 交易量


class SMABar:
    def __init__(self):
        self.fast = 0.0  # SMA快线
        self.slow = 0.0  # SMA慢线

    def init(self, bar: Bar):
        self.fast = round(bar.close, 4)
        self.slow = round(bar.close, 4)
        return self

    def next(self, bar: Bar, pre_sma: Self):
        fast = (pre_sma.fast * (CFG.sma.fast - 1) + bar.close) / CFG.sma.fast
        slow = (pre_sma.slow * (CFG.sma.slow - 1) + bar.close) / CFG.sma.slow
        self.fast = round(fast, 4)
        self.slow = round(slow, 4)
        return self


class MACDBar:
    def __init__(self):
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.dif = 0.0
        self.dea = 0.0
        self.macd = 0.0

    def init(self, bar: Bar):
        self.ema_fast = round(bar.close, 4)
        self.ema_slow = round(bar.close, 4)
        return self

    def next(self, bar: Bar, pre_macd: Self):
        self.ema_fast = self._ema(bar.close, CFG.macd.fast, pre_macd.ema_fast)
        self.ema_slow = self._ema(bar.close, CFG.macd.slow, pre_macd.ema_slow)
        self.dif = round(self.ema_fast - self.ema_slow, 4)
        self.dea = self._ema(self.dif, CFG.macd.sign, pre_macd.dea)
        self.macd = round((self.dif - self.dea) * 2, 4)
        return self

    @staticmethod
    def _ema(price, period, pre_ema):
        alpha = 2 / (period + 1)
        ema = alpha * price + (1 - alpha) * pre_ema
        return round(ema, 4)


class NodeBar:
    def __init__(self, bar: Bar):
        self.datetime = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.price = round(bar.price, 4)
        self.value = 0.0
        self.turning = 0

        self._bar = bar
        self._sma = None
        self._macd = None

    def init(self) -> Self:
        """初始节点，NodeBar"""
        self._sma = SMABar().init(self._bar)
        self._macd = MACDBar().init(self._bar)
        self.value = self._sma.fast
        self.turning = -2
        return self

    def next(self, pre_node: Self) -> Self:
        """下一个节点，NodeBar"""
        self._sma = SMABar().next(self._bar, pre_node.sma())
        self._macd = MACDBar().next(self._bar, pre_node.macd())
        self.value = self._sma.fast
        return self

    def turn(self):
        """转换为拐点，TurnBar"""
        return TurnBar(self)

    def bar(self) -> Bar:
        return self._bar

    def sma(self) -> SMABar:
        return self._sma

    def macd(self) -> MACDBar:
        return self._macd


class TurnBar:
    def __init__(self, node: NodeBar):
        self.datetime = node.datetime
        self.turning = node.turning
        self.price = node.price
        self.value = node.value

        self._rise_lots = CFG.trade.rise_add_quotas.copy()
        self._fall_lots = CFG.trade.fall_sub_quotas.copy()
        self._max_node = None
        self._min_node = None

    def max_node(self, node: NodeBar, threshold: float):
        """到下一拐点前：最大的凸点"""
        if node.value - self.value > threshold:
            if self._max_node is None or node.value > self._max_node.value:
                self._max_node = node

    def min_node(self, node: NodeBar, threshold: float):
        """到下一拐点前：最小的凹点"""
        if self.value - node.value > threshold:
            if self._min_node is None or node.value < self._min_node.value:
                self._min_node = node

    def next_turn(self, node: NodeBar, threshold: float):
        """下一拐点"""
        if self.turning == -2:
            diff = round(self.value - node.value, 4)
            if abs(diff) > threshold:
                self.turning = 1 if diff > 0 else -1
            return None
        if self._max_node and self._max_node.value - node.value > threshold:
            return self._max_node.turn()
        if self._min_node and node.value - self._min_node.value > threshold:
            return self._min_node.turn()
        return None

    def rise_lots(self):
        return self._rise_lots

    def fall_lots(self):
        return self._fall_lots


class HistBus:
    def __init__(self):
        self.nodes = []

    def prep(self, bars: list[Bar]):
        if not bars:
            return self
        self.nodes.clear()
        self.nodes.append(NodeBar(bars[0]).init())
        for bar in bars[1:]:
            node = NodeBar(bar).next(self.nodes[-1])
            self.nodes.append(node)
        return self


class PresBus:
    def __init__(self):
        self.base_price = 0.0  # 基准价格
        self.nodes = []  # 节点
        self.turns = []  # 拐点

    def prep(self, bar):
        self.base_price = round(bar.close, 4)
        self.nodes.clear()
        self.turns.clear()
        return self

    def init(self, bar):
        node = NodeBar(bar).init()
        turn = node.turn()
        self.nodes.append(node)
        self.turns.append(turn)

    def next(self, bar):
        node = NodeBar(bar).next(self.nodes[-1])
        if node.value == self.nodes[-1].value:
            return
        self.nodes.append(node)
        if len(self.nodes) < 3:
            return

        # 最小振幅价格差
        threshold = round(self.base_price * CFG.trade.amp_lower_limit, 4)

        # 计算凹凸点
        prev = self.nodes[-3]
        node = self.nodes[-2]
        post = self.nodes[-1]
        if prev.value < node.value > post.value:
            node.turning = 1
            self.turns[-1].max_node(node, threshold)
        elif prev.value > node.value < post.value:
            node.turning = -1
            self.turns[-1].min_node(node, threshold)

        # 计算拐点
        turn = self.turns[-1].next_turn(post, threshold)
        self.turns.append(turn) if turn else None

        # if self.nodes[-1].datetime > '2025-10-27 13:09:00':
        #     pass


class StockMarket:
    def __init__(self, symbol):
        self.symbol = symbol  # 股票代码
        self.base_pos = None  # 基准仓位
        self.curr_pos = None  # 当前仓位
        self.curr_bus = None  # 当前行情
        self.hist_bus = None  # 历史行情

    def prep(self, bar, bars):
        self.curr_bus = PresBus().prep(bar)
        self.hist_bus = HistBus().prep(bars)

    def next(self, bar, pos):
        self.curr_pos = Pos(pos)
        if not self.curr_bus.nodes:
            self.base_pos = Pos(pos)
            self.curr_bus.init(bar)
        else:
            self.curr_bus.next(bar)


########################################################################################################################

class ExecLog:
    def __init__(self, symbol, amount):
        self.symbol = symbol  # 股票代码
        if amount > 0:
            self.trade_type = 'buy'
            self.trade_amount = amount
        else:
            self.trade_type = 'sell'
            self.trade_amount = -amount

        # 节点信息
        self.bar_datetime = ''
        self.bar_price = 0.0
        self.bar_value = 0.0

        # 映射信息
        self.map_datetime = ''
        self.map_price = 0.0
        self.map_value = 0.0

    def node(self, node: NodeBar):
        self.bar_datetime = node.datetime
        self.bar_price = node.price
        self.bar_value = node.value
        return self

    def turn(self, turn: TurnBar):
        self.map_datetime = turn.datetime
        self.map_price = turn.price
        self.map_value = turn.value
        return self


class Checker:
    def __init__(self, mkt: StockMarket):
        self._base_price = mkt.curr_bus.base_price  # 基准价格
        self._yest_node = mkt.hist_bus.nodes[-1]  # 昨日节点(日线)
        self._curr_node = mkt.curr_bus.nodes[-1]  # 当前节点(分钟)
        self._curr_turn = mkt.curr_bus.turns[-1]  # 当前拐点(分钟)
        self._curr_pos = mkt.curr_pos  # 当前持仓
        self._rise_lvl = -2  # 当前上涨等级
        self._fall_lvl = -2  # 当前下跌等级

    def is_buy(self):
        """是否买入"""
        # 如果仓位超过上限
        if self._curr_pos.valuation >= CFG.pos.base_capital * CFG.pos.invest_limit:
            return False
        # 如果日线下跌
        if self._yest_node.sma().fast <= self._yest_node.sma().low:
            return False
        # 如果分钟线下跌
        if self._curr_node.sma().fast <= self._curr_node.sma().low:
            return False
        if self._curr_node.sma().fast <= self._curr_turn.sma().fast:
            return False
        # 如果涨幅未达到阈值
        if self.rise_lvl() == -1:
            return False
        # 如果是重复操作
        if self._curr_turn.rise_lots[self.rise_lvl()] == 0:
            return False
        # 决定买入
        return True

    def is_sell(self):
        """是否卖出"""
        # 如果没有可用持仓
        if self._curr_pos.avail_amount <= 0:
            return False
        # 如果日线上涨
        if self._yest_node.sma().fast >= self._yest_node.sma().low:
            return False
        # 如果分钟线上涨
        if self._curr_node.sma().fast >= self._curr_node.sma().low:
            return False
        if self._curr_node.sma().fast >= self._curr_turn.sma().fast:
            return False
        # 如果跌幅未达到阈值
        if self.fall_lvl() == -1:
            return False
        # 如果是重复操作
        if self._curr_turn.fall_lots[self.rise_lvl()] == 0:
            return False
        # 决定卖出
        return True

    def rise_lvl(self):
        """当前上涨等级"""
        if self._rise_lvl == -2:
            self._rise_lvl = self.__level(CFG.trade.rise_thresholds)
        return self._rise_lvl

    def fall_lvl(self):
        """当前下跌等级"""
        if self._fall_lvl == -2:
            self._fall_lvl = self.__level(CFG.trade.fall_thresholds)
        return self._fall_lvl

    def __level(self, thresholds):
        diff_value = abs(self._curr_node.sma().fast - self._curr_turn.sma().fast)
        diff_ratio = round(diff_value / self._base_price, 4)
        for threshold in reversed(thresholds):
            if diff_ratio > threshold:
                return thresholds.index(threshold)
        return -1


class StockBroker:
    def __init__(self):
        self.logs = []

    def trade(self, mkt: StockMarket, order_func: Callable):
        checker = Checker(mkt)
        if checker.is_buy():
            lvl = checker.rise_lvl()
            lots = mkt.curr_bus.nodes[-1].rise_lots()
            amount = CFG.pos.base_capital * lots[lvl] / mkt.curr_bus.base_price
            amount = round(amount / 100) * 100
            order_func(mkt.symbol, amount)
            self.logger(mkt, amount)
            lots[lvl] = 0.0
            return

        if checker.is_sell():
            lvl = checker.fall_lvl()
            lots = mkt.curr_bus.nodes[-1].fall_lots()
            amount = mkt.base_pos.avail_amount * lots[lvl]
            amount = round(amount / 100) * 100
            order_func(mkt.symbol, -amount)
            self.logger(mkt, -amount)
            lots[lvl] = 0.0
            return

    def logger(self, mkt: StockMarket, amount):
        logger = ExecLog(mkt.symbol, amount)
        logger.node(mkt.curr_bus.nodes[-1])
        logger.turn(mkt.curr_bus.turns[-1])
        self.logs.append(logger)


########################################################################################################################

# 全局配置
CFG = StockConfig()

##################################################

# """
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


########################################################################################################################
# """

def initialize(context):
    """启动时执行一次"""
    g.symbol = '512480.SS'
    set_universe(g.symbol)

    config = StockConfig(symbol=g.symbol)
    # g.market = StockMarket(config)
    pass


def before_trading_start(context, data):
    """每天交易开始之前执行一次"""
    his = get_history(20, frequency='1m', field=['datetime', 'open', 'close', 'volume'], security_list=None, fq='pre')
    first_bar = his.query(f'code in ["{g.symbol}"]').iloc[-1]
    first_pos = context.portfolio.positions.get(g.symbol)
    g.market.pre(first_bar)
    g.trader.pre(first_bar, first_pos)
    pass


def handle_data(context, data):
    """每个单位周期执行一次"""
    g.market.next(data[g.symbol])

    cur_pos = context.portfolio.positions.get(g.symbol)
    cur_trading = g.market.turnings[-1]  # 拐点：-1凹点，1凸点，0其他
    amount = 0
    if cur_trading.point.turn_val == 1:
        amount = -g.trader.__get_sell_amount(cur_pos, cur_trading)
    if cur_trading.point.turn_val == -1:
        amount = g.trader.__get_buy_amount(cur_pos, cur_trading)
    if data[g.symbol].datetime.strftime('%H:%M:%S').endswith('14:58:00'):
        amount = g.trader.get_supp_amount(cur_pos)
    if amount != 0:
        if cur_pos is not None:
            log.info(f'sell={amount}, pre_price={cur_trading.point.price}, cur_price={cur_pos.last_sale_price}')
        order(g.symbol, amount)
    pass


def after_trading_end(context, data):
    """每天交易结束之后执行一次"""
    pass
