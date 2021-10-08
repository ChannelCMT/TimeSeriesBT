import pandas as pd
import tables as tb
from datetime import datetime
from itertools import accumulate, chain
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import numpy as np
from enum import Enum
import logging
from typing import List, Union, Tuple, Dict


__version__ = "0.0.2"


class OrderStatus(Enum):
    """订单状态
    """

    Pending = 0
    Holding = 1
    Finished = 2
    Canceled = 3


@dataclass
class Order:
    """Order对象
    """
    
    orderId: str = ""
    symbol: str = ""
    volume: float = 0
    entryDt: datetime = None
    entryPrice: float = 0
    entryVolume : float = 0
    entryType: str = "default"
    exitDt: datetime = None
    exitPrice: float = 0
    exitVolume: float = 0
    exitType: str = "default"
    commission: float = 0
    status: OrderStatus = OrderStatus.Pending

    def fill_entry(self, dt: datetime, price: float):
        self.entryDt = dt
        self.entryPrice = price
        self.status = OrderStatus.Holding
    
    def fill_exit(self, dt: datetime, price: float):
        self.exitDt = dt
        self.exitPrice = price
        self.status = OrderStatus.Finished

@dataclass
class OrderEntry:
    
    order: Order
    volume: float = 0
    price: float = 0


@dataclass
class OrderExit:
    
    order: Order
    volume: float = 0
    price: float = 0
    

@dataclass
class Position:
    
    long: float = 0
    frozenLong: float = 0
    short: float = 0
    frozenShort: float = 0

        
@dataclass
class AutoExit:
    
    orderId: str
    symbol: str
    direction: int
    stoploss: float = 0
    takeprofit: float = 0

        
@dataclass
class TrailingStop:
        
    orderId: str
    symbol: str
    direction: int
    trailingPercentage: float
    trailingPrice: float
        
    def update(self, price: Dict):
        if self.direction > 0:
            if self.trailingPrice < price["high"]:
                self.trailingPrice = price["high"]
                return True
            else:
                return False
        else:
            if self.trailingPrice > price["low"]:
                self.trailingPrice = price["low"]
                return True
            else:
                return False


def bar2dict(s: pd.Series):
    dct = {name: {} for name in s.index.levels[0]}
    for key, value in s.to_dict().items():
        dct[key[0]][key[1]] = value
    dct["datetime"] = s.name
    return dct


class Portfolio(object):
    
    def __init__(self):
        self._order_entries = OrderedDict()
        self._order_exits = OrderedDict()
        self._orders = {}
        self._closed_orders = OrderedDict()
        self._trailing_stops = {}
        self._timestops = {}
        self._autoexits = {}
        self._positions = {}
        self._order_id = 0
        self.cash = 0
        self.history_holding: pd.DataFrame = None
        self.backtest_results = {}

    def init(self, symbols: list):
        for symbol in symbols:
            self._positions[(symbol, "long")] = 0
            self._positions[(symbol, "short")] = 0
    
    def algorithm(self, data: dict):
        pass
    
    def on_bar(self, s: pd.Series):
        s_dct = bar2dict(s)
        self.before_bar(s_dct)
        self.algorithm(s_dct)
        self.after_bar(s_dct)
        return self._positions.copy()
    
    def before_bar(self, s: Dict):
        self.handle_exits(s)
        self.handle_timestops(s)
        self.handle_entries(s)
        self.handle_autoexits(s)
    
    def after_bar(self, s: Dict):
        self.handle_trailings(s)
    
    def handle_exits(self, s: Dict):
        for order_exit in list(self._order_exits.values()):
            order_exit: OrderExit
            order: Order = order_exit.order
            trigger, price = self.is_fill_trigger(order_exit.price, s[order.symbol], order_exit.volume)
            if trigger:
                self.fill_exit(order, s["datetime"], price, order_exit.volume)
    
    def fill_exit(self, order: Order, dt: datetime, price: float, volume: float, exit_type: str = "default"):
        order.exitDt = dt
        order.exitPrice = price
        order.exitVolume += volume
        order.exitType = exit_type
        if abs(order.exitVolume) < abs(order.entryVolume):
            return logging.debug(f"[order exit incomplete] {order}")
        order.status = OrderStatus.Finished
        for tasks in (
            self._autoexits, 
            self._trailing_stops, 
            self._order_exits,
            self._timestops
        ):
            if order.orderId in tasks:
                del tasks[order.orderId]
        self._closed_orders[order.orderId] = self._orders.pop(order.orderId)
        if order.volume > 0:
            self._positions[(order.symbol, "long")] -= volume
        elif order.volume < 0:
            self._positions[(order.symbol, "short")] -= volume
        self.on_order(order)
        logging.debug(f"[order exit complete] {order}")

    def fill_entry(self, order: Order, dt: datetime, price: float, volume: float):
        order.entryDt = dt
        order.entryPrice = price
        order.status = OrderStatus.Holding
        order.entryVolume += volume
        del self._order_entries[order.orderId]
        if order.volume > 0:
            self._positions[(order.symbol, "long")] += volume
        elif order.volume < 0:
            self._positions[(order.symbol, "short")] += volume
        self.on_order(order)

    def is_fill_trigger(self, trigger_price: float, market_price: Dict, volume: float):
        if not trigger_price:
            return True, market_price["open"]
        if volume > 0:
            if market_price["open"] <= trigger_price:
                return True, market_price["open"]
            elif market_price["low"] <= trigger_price:
                return True, trigger_price
            else:
                return False, 0
        elif volume < 0:
            if market_price["open"] >= trigger_price:
                return True, market_price["open"]
            elif market_price["high"] >= trigger_price:
                return True, trigger_price
            else:
                return False, 0
        else:
            raise ValueError(f"Invalid volume: {volume}")

    def handle_entries(self, s: Dict):
        for entry in list(self._order_entries.values()):
            entry: OrderEntry
            order: Order = entry.order
            trigger, price = self.is_fill_trigger(entry.price, s[order.symbol], entry.volume)
            if trigger:
                self.fill_entry(order, s["datetime"], price, entry.volume)

    def entry_order(self, symbol: str, volume: float, price: float = 0):
        order = Order(self.next_order_id, symbol, volume)
        self._orders[order.orderId] = order
        self._order_entries[order.orderId] = OrderEntry(order, volume, price)
        return order.orderId

    def exit_order(self, order: Order, volume: float = 0, price: float = 0):
        remain_volume = order.entryVolume - order.exitVolume
        self._order_exits[order.orderId] = OrderExit(order, min(volume, remain_volume) or remain_volume, price)
    
    def get_holding_order(self, order_id: str):
        return self._orders[order_id]
    
    def get_order(self, order_id: str):
        if order_id in self._orders:
            return self._orders[order_id]
        if order_id in self._closed_orders:
            return self._closed_orders[order_id]
        raise KeyError(f"Order not found: order_id = {order_id}")

    @property
    def next_order_id(self):
        oid = self._order_id
        self._order_id += 1
        return str(oid)

    def timestop(self, order_id: str, expire_at: datetime):
        self._timestops[order_id] = expire_at

    def handle_timestops(self, s: Dict):
        for order_id, expire_at in tuple(self._timestops.items()):
            if s["datetime"] >= expire_at:
                logging.debug(f"[timestop {order_id}] expire_at={expire_at} current_time={s['datetime']}")
                order: Order = self._orders[order_id]
                self.fill_exit(
                    order,
                    s["datetime"],
                    s[order.symbol]["open"],
                    order.entryVolume - order.exitVolume,
                    "timestop"
                )
                
    def handle_autoexits(self, s: Dict):
        for order_id, ae in tuple(self._autoexits.items()):
            ae: AutoExit
            
            if ae.stoploss:
                trigger, price = self.is_fill_trigger(ae.stoploss, s[ae.symbol], ae.direction)
                
                if trigger:
                    logging.debug(f"[stoploss {ae.symbol} {order_id} {ae.direction}] expected={ae.stoploss} executable={price}")
                    if order_id in self._orders:
                        order = self._orders[order_id]
                        if order.status == OrderStatus.Holding:
                            self.fill_exit(order, s["datetime"], price, order.entryVolume, "stoploss")

            if ae.takeprofit:
                trigger, price = self.is_fill_trigger(ae.takeprofit, s[ae.symbol], -ae.direction)
                if trigger:
                    if order_id in self._orders:
                        order = self._orders[order_id]
                        if order.status == OrderStatus.Holding:
                            self.fill_exit(order, s["datetime"], price, order.entryVolume, "takeprofit")

    def set_autoexit(self, order_id: str, stoploss: float = 0, takeprofit: float = 0):
        if order_id not in self._orders:
            return
        if order_id not in self._autoexits:
            order: Order = self._orders[order_id]
            ae = AutoExit(
                order_id, 
                order.symbol,
                1 if order.volume > 0 else -1,
                stoploss, 
                takeprofit
            )
            self._autoexits[order_id] = ae
            logging.debug(f"[set autoexit] {ae}")
        else:
            ae: AutoExit = self._autoexits[order_id]
            if stoploss:
                ae.stoploss = stoploss
            if takeprofit:
                ae.takeprofit = takeprofit
            logging.debug(f"[update autoexit] {ae}")
    
    def mix_autoexit(self, bars: Dict, order_ids: List[str], stoploss: float=0, takeprofit: float=0):
        """混合止损止盈，在输入的订单总体盈利或损失超过止损止盈范围时平仓。

        :param bars: k线数据， 与algorithm中传入的数据一致
        :type bars: Dict
        :param order_ids: 需要同时止损止盈的订单号
        :type order_ids: List[str]
        :param stoploss: 止损百分比，0表示不止损。sample: 0.05表示%5止损。
        :type stoploss: float
        :param takeprofit: 止盈百分比，0表示不止盈。sample: 0.05表示%5止盈。
        :type takeprofit: float
        """
        total_profit = 0
        total_capital = 0
        for order_id in order_ids:
            order: Order = self.get_order(order_id)
            if order.status == OrderStatus.Pending:
                continue
            total_capital += abs(order.entryVolume * order.entryPrice)
            total_profit += (bars[order.symbol]["close"] - order.entryPrice) * order.entryVolume
        
        profit_pct = total_profit / total_capital
        if stoploss and (profit_pct <= - stoploss):
            logging.info(f"[mix autoexit] exit at stoploss: {stoploss}")
            self._batch_exit_orders(order_ids)
        
        if takeprofit and (profit_pct >= takeprofit):
            logging.info(f"[mix autoexit] exit at takeprofit: {stoploss}")
            self._batch_exit_orders(order_ids)

    def _batch_exit_orders(self, order_ids: List[str]):
        for order_id in order_ids:
            order: Order = self.get_order(order_id)
            if order.status == OrderStatus.Pending:
                self.cancel_entry(order_id)
                if order.entryVolume != 0:
                    self.exit_order(order)
            elif order.status == OrderStatus.Holding:
                self.exit_order(order)

    def handle_trailings(self, s: Dict):
        for order_id, ts in self._trailing_stops.items():
            ts: TrailingStop
            if ts.update(s[ts.symbol]):
                logging.debug(f"[trailing point updated] {ts}")
                self.set_autoexit(order_id, ts.trailingPrice*(1-ts.trailingPercentage*ts.direction))

    def set_trailing_stop(self, order_id: str, trailing_percentage: float, trailing_price: float):
        if order_id not in self._orders:
            return
        if order_id not in self._trailing_stops:
            order: Order = self._orders[order_id]
            ts = TrailingStop(
                order_id, 
                order.symbol,
                1 if order.volume > 0 else -1,
                trailing_percentage,
                trailing_price
            )
            self._trailing_stops[order_id] = ts
            logging.debug(f"[set trailing stop] {ts}")
        else:
            ts: TrailingStop = self._trailing_stops[order_id]
            ts.trailingPercentage = trailing_percentage
            ts.update({"high": trailing_price, "low": trailing_price})
            logging.debug(f"[modify trailing stop] {ts}")

    def on_order(self, order: Order):
        pass
    
    def cancel_entry(self, order_id: str):
        if order_id in self._order_entries:
            del self._order_entries[order_id]
            order: Order = self._orders.pop(order_id)
            order.status = OrderStatus.Canceled
            order.exitType = "canceled"
            self._closed_orders[order_id] = order
            self.on_order(order)
            return True
        else:
            return False
    
    def cancel_exit(self, order_id: str):
        if order_id in self._order_exits:
            del self._order_exits[order_id]
            return True
        else:
            return False
    
    def is_order_finished(self, order_id: str):
        if order_id in self._orders:
            return False
        elif order_id in self._closed_orders:            
            return True
        else:
            raise KeyError(f"Order not exists: {order_id}")
    
    def backtest(self, data: pd.DataFrame, symbols: list):

        self.init(symbols)
        result = data.apply(self.on_bar, axis=1)
        self.clear_holding_order(data.iloc[-1])
        balance = pd.DataFrame(list(result.values), result.index)
        balance.set_axis(pd.MultiIndex.from_tuples(balance.columns, names=["symbol", "position"]), axis=1, inplace=True)
        self.backtest_results["positions"] = balance 
        return balance

    def history_orders(self, filled_only: bool = True) -> pd.DataFrame:
        """[summary]

        :param filled_only: True只返回有成交的订单，False返回所有订单, defaults to True
        :type filled_only: bool, optional
        :return: 历史订单表
            >>> orderId symbol  volume                   entryDt  entryPrice  entryVolume  
            117     118    btc      -1 2020-01-07 12:30:00+00:00     7866.64           -1   
            118     117    eth      -1 2020-01-07 12:20:00+00:00      142.09           -1   
                entryType                    exitDt  exitPrice  exitVolume exitType  
            117   default 2020-01-07 14:25:00+00:00    7857.22          -1  default   
            118   default 2020-01-07 14:25:00+00:00     140.84          -1  default   
                commission                status  
            117           0  OrderStatus.Finished  
            118           0  OrderStatus.Finished  
        :rtype: pd.DataFrame
        """
        if "orders" not in self.backtest_results:
            self.backtest_results["orders"] = pd.DataFrame(list(map(lambda order: order.__dict__, self._closed_orders.values())))
        if filled_only:
            orders = self.backtest_results["orders"]
            return orders[orders["exitVolume"]!=0]
        else:
            return self.backtest_results["orders"]

    def clear_holding_order(self, prices: pd.Series):
        for order in tuple(self._orders.values()):
            order: Order
            if order.entryVolume != 0:
                self.fill_exit(
                    order, 
                    prices.name, 
                    prices[order.symbol]["close"],
                    order.entryVolume,
                    "closeOnStop"
                )
            else:
                self.cancel_entry(order.orderId)

    
    def _cal_order_performance(self, tag: str, orders: pd.DataFrame, contracts: dict=None):
        performance = cal_trade_performance(orders, contracts)
        result = perfomance_result(performance)
        self.backtest_results.setdefault("order_performance", {})[tag] = pd.concat([orders, performance], axis=1)
        self.backtest_results.setdefault("order_performance_statistics", {})[tag] = result

    def cal_order_performance(self, symbols: list=None, contracts: dict=None):
        """计算订单绩效

        :param symbols: 需要计算的品种，`None`表示所有品种一起计算, defaults to `None`
        :type symbols: list, optional
        :param contracts: 合约信息, defaults to None
            >>> contracts = {
                    "btc": {
                        "size": 1,
                        "rate": 0,
                        "slippage": 0
                    }
                }
        :type contracts: dict, optional

        """
        orders = self.history_orders()
        orders = orders[orders["exitVolume"]!=0]
        if not symbols:
            self._cal_order_performance("all", orders, contracts)
        
        else:
            orders = self.history_orders()
            for symbol in symbols:
                self._cal_order_performance(symbol, orders[orders.symbol==symbol], contracts)

    # def cal_pair_profit(self,orders: pd.DataFrame):
    #     """计算订单的综合收益

    #     :param orders: 订单表
    #     :type orders: pd.DataFrame
    #     :return: 综合收益 = 订单总收益 / 订单总成本
    #     :rtype: float
    #     """
    #     return ((orders["exitPrice"] - orders["entryPrice"]) * orders["exitVolume"]).sum() / (orders["entryPrice"] * orders["exitVolume"].abs()).sum()

    def cal_period_performance(self, bars: pd.DataFrame, contracts: dict = None):
        """计算持仓绩效

        :param bars: K线价格数据
        :type bars: pd.DataFrame
        :param contracts: 合约信息, defaults to None
            >>> contracts = {
                    "btc": {
                        "size": 1,
                        "rate": 0,
                        "slippage": 0
                    }
                }
        :type contracts: dict, optional
        """
        self.backtest_results["holding"] = cal_holding_value(self.backtest_results["positions"], bars)
        orders = self.history_orders()

        # orders = orders.groupby(orders["exitDt"]).apply(self.cal_pair_profit)

        self.backtest_results["trades"] = cal_cash_flow(orders[orders["exitVolume"]!=0], contracts)

    def get_period_statistics(self, symbols: Union[List[str], None] = None, init_cash: float = 0, periods_per_year: int = 365, freq: str = "") -> Tuple[pd.DataFrame, Dict]:
        """返回绩效

        :param symbols: 需要计算的品种，也可以所有品种一起计算, defaults to None
        :type symbols: Union[List[str], None], optional
        :param init_cash: 舒适资金, defaults to 0
        :type init_cash: float, optional
        :param periods_per_year: 每年周期数, defaults to 365
        :type periods_per_year: int, optional
        :param freq: 周期(10min, 2h, 1d), defaults to ""
        :type freq: str, optional
        :raises KeyError: 没有绩效，调用cal_period_performance生成绩效。
        :return: 绩效结果
        :rtype: Tuple[pd.DataFrame, Dict]
        

        """
        try:
            history_holding = self.backtest_results["holding"]
            cash_flow = self.backtest_results["trades"] 
        except KeyError:
            raise KeyError("holding or trades not in backtest_results, please run cal_period_performance first")

        if not symbols:
            return cal_market_value(history_holding, cash_flow, init_cash, periods_per_year, freq)
        else:
            return cal_market_value(
                history_holding[symbols], 
                cash_flow[cash_flow["symbol"].apply(lambda s: s in symbols)], 
                init_cash, 
                periods_per_year,
                freq
            )

    

class TradingResult(object):
    """每笔交易的结果"""

    # ----------------------------------------------------------------------
    def __init__(self, entryPrice, entryDt, entryID, exitPrice,
                 exitDt, exitID, volume, contracts=None, backtestResultType="Linear"):
        """Constructor"""
        self.entryPrice = entryPrice  # 开仓价格
        self.exitPrice = exitPrice  # 平仓价格

        self.entryDt = entryDt  # 开仓时间datetime
        self.exitDt = exitDt  # 平仓时间

        self.entryID = entryID
        self.exitID = exitID

        self.volume = volume  # 交易数量（+/-代表方向）
        if not isinstance(contracts, dict):
            contracts = {}
        size = contracts.get("size", 1)
        rate = contracts.get("rate", 0)
        slippage = contracts.get("slippage", 0)

        self.turnover = (self.entryPrice + self.exitPrice) * size * abs(volume)  # 成交金额

        if backtestResultType == "Inverse":
            self.commission = rate * self.turnover/ self.exitPrice  # 手续费成本
            self.slippage = slippage/self.entryPrice * size * abs(volume) + slippage/self.exitPrice * size * abs(volume)  # 滑点成本

            self.pnl = (self.exitPrice - self.entryPrice) * volume * size / self.exitPrice - self.commission - self.slippage# 净盈亏
        else:
            self.commission = self.turnover * rate  # 手续费成本
            self.slippage = slippage * 2 * size * abs(volume)  # 滑点成本

            self.pnl = (self.exitPrice - self.entryPrice) * volume * size - self.commission - self.slippage  # 净盈亏
    
    @classmethod
    def from_order(cls, order: pd.Series, contracts: dict=None):
        return TradingResult(
            order.entryPrice, order.entryDt, order.orderId, order.exitPrice, order.exitDt, order.orderId, order.exitVolume, 
            contracts.get(order.symbol, {}) if isinstance(contracts, dict) else {}
        )
    
    
def cal_trade_performance(orders: pd.DataFrame, contracts: dict=None):
    trade_series = orders.apply(TradingResult.from_order, axis=1, contracts=contracts)
    pnl = trade_series.apply(lambda t: t.pnl)
    capital = pnl.cumsum()
    max_capital = capital.cummax()
    draw_down = capital-max_capital
    return pd.DataFrame({
        "result": trade_series,
        "pnl": pnl,
        "capital": capital,
        "maxCapital": max_capital,
        "drawdown": draw_down
    })


def perfomance_result(performance: pd.DataFrame):
    totalResult = len(performance)
    winning = performance["pnl"] >= 0
    losing = ~ winning
    totalWinning = performance['pnl'][winning].sum()
    totalLosing = performance['pnl'][losing].sum()
    winningResult = winning.sum()
    losingResult = losing.sum()
    averageWinning = totalWinning / winningResult
    averageLosing = totalLosing / losingResult
    winningRate = winningResult / len(winning) * 100
    profitLossRatio = -averageWinning / averageLosing
    
    d = {}
    d['capital'] = performance["capital"].iloc[-1]
    d['maxCapital'] = performance["maxCapital"].iloc[-1]
    d['drawdown'] = performance["drawdown"].iloc[-1]
    d['totalResult'] = totalResult
    d['totalTurnover'] = performance["result"].apply(lambda r: r.turnover).sum()
    d['totalCommission'] = performance["result"].apply(lambda r: r.commission).sum()
    d['totalSlippage'] = performance["result"].apply(lambda r: r.slippage).sum()
    d['winningRate'] = winningRate
    d['averageWinning'] = averageWinning
    d['averageLosing'] = averageLosing
    d['profitLossRatio'] = profitLossRatio
    return d


def cal_holding_value(balance: pd.DataFrame, bars: pd.DataFrame):
    return balance.apply(lambda s, bars: s*bars[(s.name[0], "close")], bars=bars)


def cal_cash_flow(orders: pd.DataFrame, contracts: dict=None):
    if not isinstance(contracts, dict):
        contracts = {}
    _rates = {contract: info.get("rate", 0) for contract, info in contracts.items()}
    
    rate = orders["symbol"].map(
        defaultdict(
            lambda: 0,
            **{contract: info.get("rate", 0) for contract, info in contracts.items()}
        )
    )
    slippage = orders["symbol"].map(
        defaultdict(
            lambda: 0,
            **{contract: info.get("slippage", 0) for contract, info in contracts.items()}
        )
    )
    entry_flow = -1 * orders["entryPrice"] * orders["entryVolume"]
    entries = pd.DataFrame({
        "amount": entry_flow,
        "commission": entry_flow.abs() * rate,
        "slippage": orders["entryVolume"].abs() * slippage,
        "symbol": orders["symbol"],
        "price": orders["entryPrice"],
        "volume": orders["entryVolume"],
        "type": "entry"
    })
    entries.index = orders["entryDt"]
    
    exit_flow = orders["exitPrice"] * orders["exitVolume"]
    exits = pd.DataFrame({
        "amount": exit_flow,
        "commission": exit_flow.abs() * rate,
        "slippage": orders["exitVolume"].abs() * slippage,
        "symbol": orders["symbol"],
        "price": orders["exitPrice"],
        "volume": orders["exitVolume"],
        "type": "exit"
    })
    exits.index = orders["exitDt"]
    flow = pd.concat([entries, exits]).sort_index()
    flow["netAmount"] = flow["amount"] - flow["commission"] - flow["slippage"]
    return flow


def cal_market_value(holding_value: pd.DataFrame, cash_flow: pd.DataFrame, init_cash: float=0, periods_per_year: int = 365, freq: str = "") -> Tuple[pd.DataFrame, Dict]:
    if not freq:
        grouper = cash_flow.groupby(level=0)
    else:
        grouper = cash_flow.resample(freq)
        holding_value = holding_value.resample(freq).last()
    total_holding = holding_value.sum(axis=1)
    total_holding.iloc[-1] = 0
    init_cash = max( 
        init_cash,
        abs(cash_flow["amount"].cumsum().min()) + grouper[["amount", "commission", "slippage"]].apply(
            lambda df: (df["amount"].abs() + df["commission"] + df["slippage"]).sum()
        ).max()
    )
    df = grouper[["amount", "commission", "slippage", "netAmount"]].sum().reindex(index=total_holding.index).fillna(0)
    df["turnover"] = grouper["amount"].apply(lambda s: s.abs().sum())
    df["turnover"].fillna(0, inplace=True)
    df["tradeCount"] = grouper["amount"].count()
    df["tradeCount"].fillna(0, inplace=True)
    df["holding"] = total_holding
    df["cash"] = df["amount"].cumsum() + init_cash
    df["netCash"] = df["netAmount"].cumsum() + init_cash
    df["balance"] = df["netCash"] + df["holding"]
    df["totalBalance"] = df["cash"] + df["holding"]
    df["netPnl"] = df["balance"].diff(1).fillna(df["balance"].iloc[0]-init_cash)
    df["totalPnl"] = df["totalBalance"].diff(1).fillna(df["totalBalance"].iloc[0]-init_cash)
    df["return"] = df["netPnl"] / init_cash
    df["retWithoutFee"] = df["totalPnl"] / init_cash
    df["highlevel"] = df["balance"].cummax()
    df['drawdown'] = df['balance'] - df['highlevel']
    df['ddPercent'] = df['drawdown'] / df['highlevel'] * 100
    
    totalPeriods = len(df)
    profitPeriods = len(df[df['netPnl'] > 0])
    lossPeriods = len(df[df['netPnl'] < 0])

    endBalance = df['balance'].iloc[-1]
    maxDrawdown = df['drawdown'].min()
    maxDdPercent = df['ddPercent'].min()

    totalNetPnl = df['netPnl'].sum()
    dailyNetPnl = totalNetPnl / totalPeriods

    totalCommission = df['commission'].sum()
    dailyCommission = totalCommission / totalPeriods

    totalSlippage = df['slippage'].sum()
    dailySlippage = totalSlippage / totalPeriods
    
    totalTurnover = df['turnover'].sum()
    dailyTurnover = totalTurnover / totalPeriods

    totalTradeCount = df['tradeCount'].sum()
    dailyTradeCount = totalTradeCount / totalPeriods
    
    totalReturn = (endBalance / init_cash - 1) * 100
    annualizedReturn = totalReturn / totalPeriods * periods_per_year
    periodReturn = df['return'].mean() * 100
    returnStd = df['return'].std() * 100
    periodReturnWithoutFee = df['retWithoutFee'].mean() * 100
    returnWithoutFeeStd = df['retWithoutFee'].std() * 100

    if returnStd:
        sharpeRatio = periodReturn / returnStd * np.sqrt(periods_per_year)
    else:
        sharpeRatio = 0
    if returnWithoutFeeStd:
        SRWithoutFee = periodReturnWithoutFee / returnWithoutFeeStd * np.sqrt(periods_per_year)
    else:
        SRWithoutFee = 0
    theoreticalSRWithoutFee = 0.1155 * np.sqrt(dailyTradeCount * periods_per_year)
    calmarRatio = annualizedReturn/abs(maxDdPercent)
    
    result = {
        'totalDays': int(totalPeriods),
        'profitDays': int(profitPeriods),
        'lossDays': int(lossPeriods),
        'endBalance': float(endBalance),
        'maxDrawdown': float(maxDrawdown),
        'maxDdPercent': float(maxDdPercent),
        'totalNetPnl': float(totalNetPnl),
        'dailyNetPnl': float(dailyNetPnl),
        'totalCommission': float(totalCommission),
        'dailyCommission': float(dailyCommission),
        'totalSlippage': float(totalSlippage),
        'dailySlippage': float(dailySlippage),
        'totalTurnover': float(totalTurnover),
        'dailyTurnover': float(dailyTurnover),
        'totalTradeCount': int(totalTradeCount),
        'dailyTradeCount': float(dailyTradeCount),
        'totalReturn': float(totalReturn),
        'annualizedReturn': float(annualizedReturn),
        'calmarRatio': float(calmarRatio),
        'dailyReturn': float(periodReturn),
        'returnStd': float(returnStd),
        'sharpeRatio': float(sharpeRatio),
        'dailyReturnWithoutFee': float(periodReturnWithoutFee),
        'returnWithoutFeeStd': float(returnWithoutFeeStd),
        'SRWithoutFee': float(SRWithoutFee),
        'theoreticalSRWithoutFee': float(theoreticalSRWithoutFee)
    }
    
    return df, result


def ms2dict(s: pd.Series):
    container = {symbol: {} for symbol in s.index.levels[0]}
    for keys, value in s.to_dict().items():
        container[keys[0]][keys[1]] = value
    return container