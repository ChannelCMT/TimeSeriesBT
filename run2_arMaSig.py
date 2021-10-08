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

importlib.reload(data_source)
# read param json
paramVersion = "_v1"
path = os.path.split(os.path.realpath(__file__))[0]
with open(path+"/sig_setting%s.json"%(paramVersion)) as param:
    setting = json.load(param)

sigPeriod = setting['sigPeriod']
ma_param = setting['ma_param']
ar_param = setting['ar_param']

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

dictDf = load_obj('symbolsData_'+sigPeriod[0])

# calculate symbols signal
def cal_sig(data: pd.DataFrame, cci_param):
    maDiff = data['close'] - ta.MA((data['high']+data['low']+data['close'])/3, ma_param)
    return pd.DataFrame({"maDiff": maDiff})

# data manager transform data type
class Signal(data_source.DataManager):
    def handle_symbol(self, symbol: str, freq: str, data: pd.DataFrame) -> pd.DataFrame:
        if freq == "15min":
            return cal_sig(data, ma_param)
        else:
            return super().handle_symbol(symbol, freq, data)

sigData = Signal(dictDf)
sigData.prepare_data()

symbolSigData = sigData.basic_data
sigMaDiff = symbolSigData.loc[:, pd.IndexSlice[:, "maDiff"]]
sigMaDiff[sigMaDiff>=0]=0
sigMaDiff[sigMaDiff<0]=-1
symbolSigData[sigMaDiff.columns] = sigMaDiff
# calculate overall environment

from sklearn.decomposition import PCA

symbolsPct = symbolSigData.loc[:, pd.IndexSlice[:, "close"]].pct_change().dropna()

def cal_pca(pctArray):
    pca = PCA(n_components=5)
    pca.fit(pctArray)
    return pca.explained_variance_ratio_[0]

pcaList = []
for i in range(len(symbolsPct)-ar_param+1):
    pctArray = np.array(symbolsPct.iloc[i:i+ar_param])
    pcaList.append(cal_pca(pctArray))

symbolsPca = symbolsPct.iloc[ar_param-1:]
symbolsPca['absorption'] = pcaList
symbolSigData["absorption"] = symbolsPca['absorption']

print(symbolSigData)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(symbolSigData, 'symbolSig_v1')
