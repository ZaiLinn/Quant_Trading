"""本地数据源：CSV 文件、合成数据（离线演示与测试用）。"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .base import DataSource, freq_to_timedelta, normalize, to_timestamp


class CsvSource(DataSource):
    """读取 {path}/{symbol}.csv，需包含 date/datetime 列与 open/high/low/close/volume 列。"""

    name = "csv"

    def __init__(self, path: str = "./data"):
        self.path = Path(path)

    def fetch(self, symbol: str, start=None, end=None, freq: str = "1d") -> pd.DataFrame:
        file = self.path / f"{symbol.replace('/', '-')}.csv"
        df = pd.read_csv(file)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in ("date", "datetime", "time", "timestamp") if c in df.columns), None)
        if date_col is None:
            raise ValueError(f"{file} 缺少日期列")
        df = df.set_index(pd.to_datetime(df[date_col]))
        return normalize(df).loc[to_timestamp(start):to_timestamp(end)]


class SyntheticSource(DataSource):
    """带趋势/震荡切换的几何布朗运动。同一 symbol + seed 结果确定。"""

    name = "synthetic"
    ORIGIN = pd.Timestamp("2005-01-03")

    def __init__(self, seed: int = 42, start_price: float = 10.0, vol: float = 0.25):
        self.seed = seed
        self.start_price = start_price
        self.vol = vol

    def fetch(self, symbol: str, start=None, end=None, freq: str = "1d") -> pd.DataFrame:
        start_ts = to_timestamp(start) or pd.Timestamp("2015-01-01")
        end_ts = to_timestamp(end) or pd.Timestamp("2024-12-31")
        # 日线路径从固定起点生成再切片：不同请求区间看到的是同一条价格路径
        if freq == "1d":
            idx = pd.bdate_range(self.ORIGIN, max(end_ts, pd.Timestamp("2030-12-31")))
            dt = 1 / 252
        else:
            step = freq_to_timedelta(freq)
            idx = pd.date_range(start_ts, end_ts, freq=step)
            dt = step / pd.Timedelta(days=365)
        rng = np.random.default_rng(self.seed + zlib.crc32(symbol.encode()))
        n = len(idx)
        # 马尔可夫切换的漂移项：模拟牛熊/震荡阶段，使趋势类与反转类策略都有用武之地
        regimes = np.array([0.35, -0.30, 0.0])
        state, drift = 2, np.empty(n)
        for i in range(n):
            if rng.random() < 0.01:
                state = rng.integers(0, 3)
            drift[i] = regimes[state]
        vol = self.vol * np.exp(0.3 * rng.standard_normal(n).cumsum() / np.sqrt(n))
        ret = (drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * rng.standard_normal(n)
        close = self.start_price * np.exp(np.cumsum(ret))
        prev = np.concatenate([[self.start_price], close[:-1]])
        gap = 1 + 0.2 * vol * np.sqrt(dt) * rng.standard_normal(n)
        open_ = prev * gap
        span = np.abs(rng.standard_normal(n)) * vol * np.sqrt(dt) * 0.8
        high = np.maximum(open_, close) * (1 + span)
        low = np.minimum(open_, close) * (1 - span)
        volume = rng.lognormal(13, 0.4, n)
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                           "volume": volume}, index=idx)
        return normalize(df).loc[start_ts:end_ts]
