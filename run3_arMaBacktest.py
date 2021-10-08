from vector import portfolio, data_source
import importlib
import talib as ta
from talib import abstract
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import tables as tb
from datetime import datetime, timedelta, timezone
import pickle
import os
import json
import time
import arMa_trader
import matplotlib.pyplot as plt

# # backtesting
importlib.reload(portfolio)
importlib.reload(arMa_trader)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
symbolSigData = load_obj('symbolSig_v1')
print('readDataDone:', symbolSigData.tail())

symbols = ['bnb', 'btc', 'eth', 'uni', 'link']
bars = symbolSigData
trader = arMa_trader.Trader()
balance = trader.backtest(bars, symbols)
# 获取 order
orders=trader.history_orders()
print(orders)

# get trader
trader.cal_period_performance(bars)

res = trader.get_period_statistics(init_cash=100000,freq='d')
print(res)
# chart of perf
res[0]['balance'].iloc[:].plot(figsize=(15,7))
plt.show()

orders=trader.history_orders()
orders.to_csv('order.csv')

import htmlplot
importlib.reload(htmlplot.core)

mp = htmlplot.core.MultiPlot()
mp.set_main(bars["eth"], orders[orders.symbol=="eth"])
# mp.set_line(bars["maDiff"], pos=1)
mp.show()