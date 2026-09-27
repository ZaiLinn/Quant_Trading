"""策略基类与通用工具。

策略只做一件事：根据截至 t 收盘的数据，输出 t 时刻的"目标权重"表（时间 × 标的）。
- 权重 = 目标持仓市值 / 账户权益，负数为做空；
- NaN = 保持当前持仓不动（用于定期调仓：非调仓日填 NaN）；
- 成交由引擎在 t+1 开盘执行 —— 从机制上杜绝"用收盘价算信号又按收盘价成交"的未来函数。
同一份 generate() 同时用于回测与实盘。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..data.panel import Panel

STRATEGIES: dict[str, type["Strategy"]] = {}


def register(cls: type["Strategy"]) -> type["Strategy"]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} 缺少 name")
    STRATEGIES[cls.name] = cls
    return cls


def get_strategy(name: str, **params) -> "Strategy":
    if name not in STRATEGIES:
        raise ValueError(f"未知策略 {name!r}，可选: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)


class Strategy(ABC):
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    params: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.params)
        if unknown:
            raise ValueError(f"{self.name} 不支持参数 {sorted(unknown)}，可用: {sorted(self.params)}")
        self.p: dict[str, Any] = {**self.params, **params}

    def __getattr__(self, item: str):
        # 允许 self.fast 这种写法访问参数
        p = self.__dict__.get("p", {})
        if item in p:
            return p[item]
        raise AttributeError(item)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.p.items())
        return f"{self.name}({args})"

    @abstractmethod
    def generate(self, panel: Panel) -> pd.DataFrame:
        """返回目标权重 DataFrame（index 与 columns 同 panel）。"""

    def warmup(self) -> int:
        """指标预热所需 K 线数，实盘据此决定拉取多少历史。默认取整数参数最大值的 3 倍。"""
        ints = [v for v in self.p.values() if isinstance(v, int) and not isinstance(v, bool)]
        return max(ints, default=20) * 3 + 10


# ---------------------------------------------------------------- 工具函数


def hold_until(entries: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    """入场信号后持有直到出场信号（同一根同时出现时出场优先），返回 0/1。"""
    state = pd.DataFrame(np.nan, index=entries.index, columns=entries.columns)
    state = state.mask(entries.fillna(False).astype(bool), 1.0)
    state = state.mask(exits.fillna(False).astype(bool), 0.0)
    return state.ffill().fillna(0.0)


def normalize_weights(signal: pd.DataFrame, gross: float = 1.0,
                      max_weight: float | None = None) -> pd.DataFrame:
    """把信号强度（0/±1 或打分）按行归一为权重，总敞口 = gross。超过 max_weight 的部分留作现金。"""
    s = signal.fillna(0.0)
    total = s.abs().sum(axis=1).replace(0, np.nan)
    w = s.div(total, axis=0).fillna(0.0) * gross
    if max_weight is not None:
        w = w.clip(-max_weight, max_weight)
    return w


def rebalance_mask(index: pd.DatetimeIndex, freq: str | None) -> pd.Series:
    """每个周期（D/W/M/Q/Y）的第一根 K 线为 True。freq 为空时每根都为 True。

    用"周期首根"而不是"周期末根"：判断末根需要知道下一根的日期（隐性未来信息），
    且实盘当天无法确定今天是否本周最后一个交易日；首根只依赖历史，回测与实盘行为一致。
    """
    if not freq or freq == "D":
        return pd.Series(True, index=index)
    period = pd.Series(index.to_period(freq), index=index)
    return period != period.shift(1)


def on_rebalance(weights: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    """只在调仓日给出目标权重，其余为 NaN（保持持仓）。"""
    mask = rebalance_mask(weights.index, freq)
    return weights.where(mask, np.nan)
