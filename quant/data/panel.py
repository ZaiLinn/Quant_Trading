"""多标的对齐面板。

把 {symbol: OHLCV} 对齐到统一时间轴，得到 field -> DataFrame(时间 × 标的)。
停牌/未上市处为 NaN。策略直接对 panel.close 等做向量化运算即可。
"""
from __future__ import annotations

import pandas as pd

from .base import OHLCV


class Panel:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        if not frames:
            raise ValueError("Panel 至少需要一个标的")
        self.symbols: list[str] = list(frames)
        index = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))), name="date")
        self.index = index
        self._fields: dict[str, pd.DataFrame] = {}
        for field in OHLCV:
            self._fields[field] = pd.DataFrame(
                {s: frames[s][field].reindex(index) for s in self.symbols}, index=index
            )

    @classmethod
    def from_fields(cls, fields: dict[str, pd.DataFrame]) -> "Panel":
        obj = cls.__new__(cls)
        ref = fields["close"]
        obj.symbols = list(ref.columns)
        obj.index = ref.index
        obj._fields = {k: v.copy() for k, v in fields.items()}
        return obj

    def __getitem__(self, field: str) -> pd.DataFrame:
        return self._fields[field]

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        if not len(self):
            return "Panel(empty)"
        return f"Panel({len(self.symbols)} symbols, {len(self)} bars, {self.index[0]} ~ {self.index[-1]})"

    open = property(lambda self: self._fields["open"])
    high = property(lambda self: self._fields["high"])
    low = property(lambda self: self._fields["low"])
    close = property(lambda self: self._fields["close"])
    volume = property(lambda self: self._fields["volume"])

    def slice(self, start=None, end=None) -> "Panel":
        """按时间切片（含两端）。"""
        return Panel.from_fields({k: v.loc[start:end] for k, v in self._fields.items()})

    def iloc(self, sl: slice) -> "Panel":
        return Panel.from_fields({k: v.iloc[sl] for k, v in self._fields.items()})

    def select(self, symbols: list[str]) -> "Panel":
        return Panel.from_fields({k: v[symbols] for k, v in self._fields.items()})

    def frame(self, symbol: str) -> pd.DataFrame:
        """取单个标的的 OHLCV。"""
        return pd.DataFrame({f: self._fields[f][symbol] for f in OHLCV}).dropna(subset=["close"])
