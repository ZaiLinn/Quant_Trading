"""加密货币 K 线（ccxt，支持 100+ 交易所）。时间统一为 UTC（tz-naive）。"""
from __future__ import annotations

import pandas as pd

from .base import DataSource, freq_to_timedelta, normalize, retry, to_timestamp


class CcxtSource(DataSource):
    name = "ccxt"

    def __init__(self, exchange: str = "binance", options: dict | None = None,
                 drop_incomplete: bool = True):
        import ccxt

        self.exchange_id = exchange
        self.ex = getattr(ccxt, exchange)({"enableRateLimit": True, **(options or {})})
        self.drop_incomplete = drop_incomplete

    def cache_key(self, symbol: str, freq: str) -> str:
        return f"{self.name}_{self.exchange_id}_{symbol.replace('/', '-').replace(':', '-')}_{freq}"

    def fetch(self, symbol: str, start=None, end=None, freq: str = "1d") -> pd.DataFrame:
        step = freq_to_timedelta(freq)
        start_ts = to_timestamp(start) or pd.Timestamp("2017-01-01")
        end_ts = to_timestamp(end) or pd.Timestamp.now(tz="UTC").tz_localize(None)
        since = int(start_ts.tz_localize("UTC").timestamp() * 1000)
        end_ms = int(end_ts.tz_localize("UTC").timestamp() * 1000)
        rows: list[list] = []
        while since <= end_ms:
            batch = retry(lambda: self.ex.fetch_ohlcv(symbol, freq, since=since, limit=1000),
                          what=f"ccxt {symbol}")
            if not batch:
                break
            rows.extend(batch)
            nxt = batch[-1][0] + int(step.total_seconds() * 1000)
            if nxt <= since:
                break
            since = nxt
        if not rows:
            return normalize(pd.DataFrame())
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["ts"], unit="ms")
        df = normalize(df).loc[start_ts:end_ts]
        if self.drop_incomplete and len(df):
            # 最后一根 K 线若尚未收盘则丢弃，否则信号会用到未完成的数据（实盘常见坑）
            now = pd.Timestamp.now(tz="UTC").tz_localize(None)
            if df.index[-1] + step > now:
                df = df.iloc[:-1]
        return df
