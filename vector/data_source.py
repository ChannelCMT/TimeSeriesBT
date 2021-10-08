import pandas as pd
from typing import Any, List, Tuple, Dict, Union
from sortedcollections import SortedDict
from pymongo import MongoClient
from pymongo.database import Database
import os
import re
from datetime import timezone, datetime, timedelta
import logging

UTC = timezone(timedelta())


def get_freq_str(freq: int):
    if freq < 3600:
        minutes, seconds = divmod(freq, 60)
        assert seconds == 0, f"Invalid freq: {freq}"
        return f"{minutes}Min"
    elif freq < 3600 * 24:
        hours, seconds = divmod(freq, 3600)
        assert seconds == 0, f"Invalid freq: {freq}"
        return f"{hours}h"
    else:
        days, seconds = divmod(freq, 3600*24)
        assert seconds == 0, f"Invalid freq: {freq}"
        return f"{days}d"


class DataManager(object):

    def __init__(self, datas: Dict[str, pd.DataFrame] = None) -> None:
        super().__init__()
        self.datas: Dict[str, pd.DataFrame] = SortedDict()
        self.basic_data: pd.DataFrame = None
        self.basic_freq = ""
        if isinstance(datas, dict):
            self.add(datas)

    def add(self, datas: Dict[str, pd.DataFrame]):
        self.datas.update(datas)
        if len(self.datas):
            self.basic_freq = min(self.datas.keys(), key=freq2minutes)
            self.basic_data = self.datas[self.basic_freq]
            self.basic_data.columns.set_names(["symbol", "field"], inplace=True)
    
    def handle_symbol(self, symbol: str, freq: str, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=data.index)
    
    def handle_all(self, freq: str, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=data.index, columns=pd.MultiIndex([[], []], [[], []], names=["symbol", "field"]))

    def prepare_data(self):
        results = {}
        for freq, data in self.datas.items():
            result_all = self.handle_all(freq, data)
            if len(result_all.columns.names) == 1:
                result_all.columns = result_all.columns.map(lambda t: ("params", t)).rename(["symbol", "field"])
            elif len(result_all.columns.names) > 2:
                raise ValueError(f"Invalid data column names: {result_all.columns.names}")
            symbols = data.columns.levels[0]
            for symbol in symbols:
                result = self.handle_symbol(symbol, freq, data[symbol])
                if len(result.columns):
                    result.columns = result.columns.map(lambda t: (symbol, t)).rename(["symbol", "field"])
                    result_all[result.columns] = result
            if freq == self.basic_freq:
                self.basic_data[result_all.columns] = result_all
                self.basic_data.sort_index(axis=1,level="symbol", inplace=True)
            else:
                result_all.sort_index(axis=1,level="symbol", inplace=True)
                results[freq] = result_all
        
        basic_freq_minutes = freq2minutes(self.basic_freq)

        for freq, data in results.items():
            freq_minutes = freq2minutes(freq)
            shift = int(freq_minutes / basic_freq_minutes - 1)
            rdata = data.resample(self.basic_freq).ffill().shift(shift)
            self.basic_data[rdata.columns] = rdata
        
        self.basic_data.sort_index(axis=1, level="symbol", inplace=True)
        self.basic_data.ffill(axis=0, inplace=True)


class DataSource(object):

    def read(self, name: str, begin: int=0, end: int=0, columns: List[str]=None, closed_interval=(True, True)) -> pd.DataFrame:
        raise NotImplementedError()


class EmptyDataSource(DataSource):

    def read(self, name: str, begin: int, end: int, columns: List[str], closed_interval) -> pd.DataFrame:
        return pd.DataFrame()


class MongoDBSource(DataSource):

    def __init__(self, db: Database, index_col="timestamp"):
        self.db = db
        self.index_col = "timestamp"
        self.default_columns = [self.index_col, "open", "high", "low", "close", "volume"]
    
    def read(self, name: str, begin: int=0, end: int=0, columns: List[str]=None, closed_interval=(True, True)) -> pd.DataFrame:
        if isinstance(columns, list):
            prj = dict.fromkeys(columns, 1)
        else:
            prj = dict.fromkeys(self.default_columns, 1)
        prj[self.index_col] = 1
        columns = list(prj.keys())
        prj["_id"] = 0
        ft = {}
        if begin:
            p = "$gte" if closed_interval[0] else "$gt"
            ft[self.index_col] = {p: begin}
        if end:
            p = "$lte" if closed_interval[1] else "$lt"
            ft.setdefault(self.index_col, {})[p] = end
        cursor = self.db[name].find(ft, projection=prj)
        data = pd.DataFrame(list(cursor), columns=columns)
        data.set_index(self.index_col, inplace=True)
        return data


def bi_search_index(table, value, side="right", hit_shift=0):
    begin = 0
    end = table.nrows
    while(begin<end):
        mid = int((begin+end)/2)
        _v = table[mid][0]
        if _v < value:
            begin = mid+1
        elif _v > value:
            end = mid
        else:
            return mid + hit_shift
    if side == "left":
        
        return begin
    else:
        return end


def search_sorted(store:pd.HDFStore, key: str, columns: Union[List[str], None], start, end, closed_interval=(True, True)):
    table = store.get_storer(key).table
    params: Dict[str, Any] = {"columns": columns}
    if start: 
        s_index = bi_search_index(table, start, 'right', 0 if closed_interval[0] else 1)
        params["start"] = s_index
    if end:

        e_index = bi_search_index(table, end, 'right', 1 if closed_interval[1] else 0)
        params["stop"] = e_index
    return store.select(key, **params)


class CachedHDFData(object):

    def __init__(self, store: pd.HDFStore) -> None:
        super().__init__()
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.store.close()

    @classmethod
    def from_params(cls, path: str, *args, **kwargs):
        return cls(pd.HDFStore(path, *args, **kwargs))

    def close(self):
        self.store._handle.close()
        self.store.close()

    def read(self, table_name: str, begin: int=0, end: int=0, columns=None, closed_interval=(True, True)) -> pd.DataFrame:
        return search_sorted(self.store, table_name, columns, begin, end, closed_interval)
        
    def write(self, table_name: str, data: pd.DataFrame):
        self.store.append(table_name, data)

    def delete(self, table_name: str, begin: int=0, end: int=0, closed_interval=(True, True)):
        table = self.store.get_storer(table_name).table
        params = {}
        params["start"] = bi_search_index(table, begin, "right", 0 if closed_interval[0] else 1) if begin else None
        params["stop"] = bi_search_index(table, end, "right", 1 if closed_interval[1] else 0) if end else None
        return self.store.remove(table_name, **params)
            
    def table_range(self, table_name: str) -> Tuple[int, int]:
        table = self.store.get_storer(table_name).table
        return (int(table[0][0]), int(table[-1][0]))
    
    def rename(self, table_name: str, new_table_name: str):
        self.store._handle.rename_node(f"/{table_name}", new_table_name)
    
    def remove(self, table_name: str):
        self.store._handle.remove_node("/", table_name, True)

    def has_key(self, table_name: str):
        return table_name in self.store.keys()


def default_cache_root():
    if "HOME" in os.environ:
        home = os.environ["HOME"]
    elif "HOMEPATH" in os.environ:
        home = os.environ["HOMEPATH"]
    else:
        home = "."

    return os.path.join(home, ".vector_data")



class HDFDataSource(DataSource):

    def __init__(self, root: str="", default_freq: str = "1min") -> None:
        self.root = root if root else default_cache_root()
        os.makedirs(self.root, exist_ok=True)
        self.default_freq = default_freq
    
    def read(self, name: str, begin: int=0, end: int=0, columns: List[str]=None, closed_interval=(True, True)) -> pd.DataFrame:
        store = pd.HDFStore(os.path.join(self.root, name), mode="r")
        try:
            data = search_sorted(store, f"kline_{self.default_freq}", columns, begin, end, closed_interval)
        except Exception as e:
            store.close()
            raise e
        store.close()
        return data
    
    def get_cache(self, file_name: str, *args, **kwargs) -> CachedHDFData:
        return CachedHDFData.from_params(os.path.join(self.root, file_name), *args, **kwargs)
     

FREQ_COMPILER = re.compile("(\d*)(Min|min|H|h|D|d|W|w)")

FREQ_MINITE_MAP = {
    "min": 1,
    "h": 60,
    "d": int(60*24),
    "w": int(60*24*7)
}

def freq2minutes(freq: str) -> int:
    result = FREQ_COMPILER.search(freq)
    if result:
        n, f = result.groups()
        return int(FREQ_MINITE_MAP[f.lower()] * int(n))
    else:
        raise ValueError(f"Invalid frequency: {freq}")


class SourceManager(object):


    def __init__(self, source: DataSource, target: HDFDataSource) -> None:
        super().__init__()
        self.source = source
        self.target = target
        self.default_freq = "1min"
    
    @classmethod
    def from_mongodb(cls, host: str, db: str, root: str = "", default_freq="1min"):
        """使用Mongodb作为数据源生成SourceManager

        :param host: Mongodb地址
        :type host: str
        :param db: 存储数据的数据名
        :type db: str
        :param root: 本地缓存根目录, defaults to "", 默认地址为$HOME/.vector_data
        :type root: str, optional
        :param default_freq: 数据源默认周期, defaults to "1min"
        :type default_freq: str, optional
        :return: SourceManager
        :rtype: SourceManager
        """
        return cls(
            MongoDBSource(MongoClient(host)[db]),
            HDFDataSource(root, default_freq)
        )

    @classmethod
    def local(cls, root: str = "", default_freq="1min"):
        """只使用本地缓存数据，不使用远程数据源。

        :param root: 本地缓存根目录, defaults to "", 默认地址为$HOME/.vector_data
        :type root: str, optional
        :param default_freq: 数据源默认周期, defaults to "1min"
        :type default_freq: str, optional
        :return: SourceManager
        :rtype: SourceManager
        """
        return cls(EmptyDataSource(), HDFDataSource(root, default_freq))

    
    def source_key_map(self, key: str):
        return key
    
    def target_key_map(self, key: str):
        return key
    
    def update(self, keys: List[str]):
        for key in keys:
            skey = self.source_key_map(key)
            tkey = self.target_key_map(key)
            with self.target.get_cache(tkey, mode="a") as cache:
                table_name = f"kline_{self.default_freq}"
                begin , end = cache.table_range(table_name)
                data = self.source.read(skey, end, closed_interval=(False, True))
                cache.write(table_name, data)

    def pull(self, keys: List[str], begin: int = 0, end: int = 0):
        table_name = f"kline_{self.default_freq}"
        for key in keys:
            logging.info(f"[pull data] [{key}] {begin} - {end}") 
            skey = self.source_key_map(key)
            tkey = self.target_key_map(key)
            with self.target.get_cache(tkey, mode="a") as cache:
                if cache.has_key(f"/{table_name}"):
                    b, e = cache.table_range(table_name)
                    if begin and begin < b:
                        logging.info(f"[pull data] [{key}] [new table] [{begin}, {b})") 
                        data = self.source.read(skey, begin, b, closed_interval=(True, False))
                        org_name = f"{table_name}_org"
                        cache.rename(table_name, org_name)
                        cache.write(table_name, data)
                        logging.info(f"[pull data] [{key}] [append] [{b}, )") 
                        data = cache.read(org_name)
                        cache.write(table_name, data)
                        cache.remove(org_name)
                    else:
                        logging.info(f"[pull data] [{key}] skip begin: {begin} {b}") 
                    
                    if end: 
                        if e < end:
                            logging.info(f"[pull data] [{key}] [append] ({e}, {end}]") 
                            data = self.source.read(skey, e, end, closed_interval=(False, True))
                            cache.write(table_name, data)
                        else:
                            logging.info(f"[pull data] [{key}] skip end: {end} {e}") 
                    else:
                        logging.info(f"[pull data] [{key}] [append] ({e}, )")
                        data = self.source.read(skey, e, closed_interval=(False, True))
                        if len(data):
                            cache.write(table_name, data)
                else:
                    logging.info(f"[pull data] [{key}] [new table] [{begin}, {end}]") 
                    data = self.source.read(skey, begin, end)
                    cache.write(table_name, data)
    
    def resample(self, keys: List[str] , freqs: List[str], begin: int = 0, end: int = 0):
        for key in keys:
            with self.target.get_cache(self.target_key_map(key)) as cache:
                for freq in freqs:
                    if freq == self.default_freq:
                        logging.info(f"[Resample] [{key} {freq}] skip basic frequency.")
                        continue
                    logging.info(f"[Resample] [{key} {freq}] begin.")
                    self._resample(cache, freq, begin, end, key)
    
    def _resample(self, cache: CachedHDFData, freq: str, begin: int = 0, end: int = 0, key=""):
        target_table_name = f"kline_{freq}"
        basic_table_name = f"kline_{self.default_freq}"
        target_minutes = freq2minutes(freq)
        basic_minutes = freq2minutes(self.default_freq)
        if not cache.has_key(f"/{target_table_name}"):
            logging.info(f"[Resample] [{key} {freq}] new table: [{begin}, {end}]")
            data = cache.read(basic_table_name, begin, end).rename(lambda ts: datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=0))))
            result = data.resample(freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).rename(lambda t: int(t.timestamp()))
            cache.write(target_table_name, result)
        else:
            logging.info(f"[Resample] [{key} {freq}] table exists.")
            basic_begin, basic_end = cache.table_range(basic_table_name)
            if not begin:
                logging.debug(f"[Resample] [{key} {freq}] set begin to {basic_begin}")
                begin = basic_begin
            if not end:
                logging.debug(f"[Resample] [{key} {freq}] set end to {basic_end}")
                end = basic_end
            b, e = cache.table_range(target_table_name)
            if begin < b:
                cache.rename(target_table_name, f"{target_table_name}_org")
                logging.info(f"[Resample] [{key} {freq}] new table: [{begin}, {b})")
                data = cache.read(basic_table_name, begin, b, closed_interval=(True, False)).rename(lambda ts: datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=0))))
                result = data.resample(freq).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).rename(lambda t: int(t.timestamp()))
                cache.write(target_table_name, result)
                logging.info(f"[Resample] [{key} {freq}] upate table: [{b}, )")
                origin =  cache.read(f"{target_table_name}_org")
                cache.write(target_table_name, origin)
            
            if e < end:
                logging.info(f"[Resample] [{key} {freq}] upate table: [{e}, {end}]")
                data = cache.read(basic_table_name, e, end).rename(lambda ts: datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=0))))
                result = data.resample(freq).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).rename(lambda t: int(t.timestamp()))
                cache.delete(target_table_name, begin=result.index[0])
                cache.write(target_table_name, result)

    def load(self, keys: List[str], freqs: List[str], begin: int=0, end: int=0, columns: List[str]=None, closed_interval=(True, False), tzinfo=UTC):
        frames = {freq: dict() for freq in freqs}
        for key in keys:
            tkey = self.target_key_map(key)
            with self.target.get_cache(tkey, mode="r") as cache:
                for freq in freqs:
                    table_name = f"/kline_{freq}"
                    if cache.has_key(table_name):
                        data = cache.read(table_name, begin, end, columns, closed_interval)
                        data.columns = data.columns.map(lambda f: (key, f))
                        data.rename(lambda t: datetime.fromtimestamp(t, tz=tzinfo), inplace=True)
                        frames[freq][key] = data
                        
                    else:
                        raise ValueError(f"Kline of {freq} does not exit")
            
        results = {}
        for freq, dct in frames.items():
            data = pd.concat(list(dct.values()), axis=1)
            data.ffill(inplace=True)
            results[freq] = data
        return results
