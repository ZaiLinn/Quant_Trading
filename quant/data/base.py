"""数据源接口与标准化。

所有数据源输出统一格式：DatetimeIndex（tz-naive，名为 date）+ open/high/low/close/volume 浮点列，
按时间升序且无重复。上层代码只依赖这一格式。
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import pandas as pd

log = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]

_UNIT = {"m": "min", "h": "h", "d": "D", "w": "W"}


def freq_to_timedelta(freq: str) -> pd.Timedelta:
    """'5m' / '1h' / '1d' / '1w' -> Timedelta；'1M'（月）按 30 天近似。"""
    n, unit = int(freq[:-1] or 1), freq[-1]
    if unit == "M":
        return pd.Timedelta(days=30 * n)
    if unit not in _UNIT:
        raise ValueError(f"不支持的周期: {freq}")
    return pd.Timedelta(n, _UNIT[unit])


def is_intraday(freq: str) -> bool:
    return freq_to_timedelta(freq) < pd.Timedelta(days=1)


def to_timestamp(x) -> pd.Timestamp | None:
    if x is None or x == "":
        return None
    return pd.Timestamp(x)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """清洗为标准 OHLCV：排序、去重、类型转换、剔除无效行、修正高低价。"""
    if df.empty:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], name="date"), dtype=float)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少列: {missing}")
    out = df[OHLCV].apply(pd.to_numeric, errors="coerce").astype(float)
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx.rename("date")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    # 缺失的开高低用收盘价补齐，并保证 high >= max(open, close) >= min(open, close) >= low
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    body_hi = out[["open", "close"]].max(axis=1)
    body_lo = out[["open", "close"]].min(axis=1)
    out["high"] = out["high"].where(out["high"] >= body_hi, body_hi)
    out["low"] = out["low"].where(out["low"] <= body_lo, body_lo)
    return out


def retry(fn: Callable, tries: int = 3, delay: float = 1.5, what: str = ""):
    """网络请求重试（指数退避）。免费数据源经常偶发失败。"""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - 数据源异常类型五花八门
            if i == tries - 1:
                raise
            wait = delay * (2 ** i)
            log.warning("%s 失败(%s)，%.1fs 后重试", what or fn, type(e).__name__, wait)
            time.sleep(wait)


class DataSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, symbol: str, start=None, end=None, freq: str = "1d") -> pd.DataFrame:
        """返回标准 OHLCV（已 normalize）。"""

    def cache_key(self, symbol: str, freq: str) -> str:
        """缓存文件名；复权方式等会影响价格的参数必须体现在 key 中。"""
        return f"{self.name}_{symbol.replace('/', '-')}_{freq}"
