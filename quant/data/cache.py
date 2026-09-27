"""Parquet 本地缓存。

复权数据每次除权后历史价格都会整体变化，增量追加会混入不同复权基准（很多开源项目的坑）。
因此这里采用"过期整段重拉"：缓存覆盖请求区间且未过期则直接读，否则整段重新下载覆盖。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .base import DataSource, to_timestamp

log = logging.getLogger(__name__)


class CachedSource(DataSource):
    PAD = pd.Timedelta(days=730)

    def __init__(self, source: DataSource, cache_dir: str = "./data_cache", ttl_hours: float = 12):
        self.source = source
        self.name = source.name
        self.dir = Path(cache_dir)
        self.ttl = ttl_hours * 3600

    def cache_key(self, symbol: str, freq: str) -> str:
        return self.source.cache_key(symbol, freq)

    def _path(self, symbol: str, freq: str) -> Path:
        return self.dir / f"{self.cache_key(symbol, freq)}.parquet"

    def fetch(self, symbol: str, start=None, end=None, freq: str = "1d") -> pd.DataFrame:
        path = self._path(symbol, freq)
        start_ts, end_ts = to_timestamp(start), to_timestamp(end)
        if path.exists():
            fresh = time.time() - path.stat().st_mtime < self.ttl
            df = pd.read_parquet(path)
            meta_start = pd.Timestamp(df.attrs.get("req_start")) if df.attrs.get("req_start") else None
            covers_start = start_ts is None or (meta_start is not None and meta_start <= start_ts) or (
                len(df) and df.index[0] <= start_ts)
            covers_end = end_ts is not None and len(df) and df.index[-1] >= end_ts
            if covers_start and (fresh or covers_end):
                return df.loc[start_ts:end_ts]
        # 多取一段历史：之后换策略（预热更长）或调整开始日期时不必重新下载
        fetch_start = start_ts - self.PAD if start_ts is not None else None
        df = self.source.fetch(symbol, fetch_start, end, freq)
        self.dir.mkdir(parents=True, exist_ok=True)
        df.attrs["req_start"] = str(fetch_start) if fetch_start is not None else ""
        df.to_parquet(path)
        log.info("缓存 %s: %d 行", path.name, len(df))
        return df.loc[start_ts:end_ts]
