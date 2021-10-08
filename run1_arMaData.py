from vector import portfolio, data_source
import importlib
from datetime import datetime, timedelta, timezone
import pickle
import os
import json
import tables as tb
 
# get original data
importlib.reload(data_source)
class DataSource(data_source.SourceManager):
    
    # 定义转换数据源命名转换规则: eth -> eth_usdt.spot:binance (MongoDB表名)
    def source_key_map(self, key: str):
        return f"{key}_usdt.spot:binance"
    
    # 定义本地缓存文件命名规则：eth -> eth
    def target_key_map(self, key: str):
        return key

def get_data(MONGODB_HOST, KLINE_DB, symbols, freqs, startTime, endTime):
    ds = DataSource.from_mongodb(
        MONGODB_HOST,
        KLINE_DB,
        root = 'vector_cache' #存放缓存的默认位置
    )
    # 从数据库拉取一分钟数据
    ds.pull(symbols, begin=startTime, end=endTime)
    # 合成不同周期k线
    print(symbols, freqs)
    ds.resample(symbols, freqs)
    result = ds.load(symbols, freqs, startTime, endTime)
    return result


paramVersion = "_v1"
path = os.path.split(os.path.realpath(__file__))[0]
with open(path+"/sig_setting%s.json"%(paramVersion)) as param:
    setting = json.load(param)

print('setting:', setting)
MONGODB_HOST = setting['MONGODB_HOST']
KLINE_DB = setting['KLINE_DB']
symbols = setting['symbols']
freqs = setting['sigPeriod']

utc = timezone(timedelta())
startTime = datetime(2021, 1, 1, tzinfo=utc).timestamp()
endTime = datetime(2021, 10, 1).timestamp()

dictDf = get_data(MONGODB_HOST, KLINE_DB, symbols, freqs, startTime, endTime)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(dictDf, 'symbolsData_'+freqs[0])