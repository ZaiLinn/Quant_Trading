"""多策略组合：把多个子策略的目标权重按资金比例加总，同一账户运行。

例如 50% ETF 动量轮动 + 50% 风险平价：两者相关性低时，组合回撤通常明显小于任一单策略。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.panel import Panel
from .base import Strategy, get_strategy, register
from .validate import validate_targets


@register
class Blend(Strategy):
    name = "blend"
    description = "多策略组合：components 为子策略列表 [{name, params, weight}]，权重为资金占比"
    params = {"components": [{"name": "momentum_rotation", "params": {"top_k": 2}, "weight": 0.5},
                             {"name": "risk_parity", "weight": 0.5}]}

    def _children(self) -> list[tuple[Strategy, float]]:
        comps = self.components
        if not comps:
            raise ValueError("blend 需要至少一个子策略：components: [{name: ..., params: {...}, weight: 0.5}]")
        default = 1.0 / len(comps)
        return [(get_strategy(c["name"], **(c.get("params") or {})), float(c.get("weight", default)))
                for c in comps]

    def warmup(self) -> int:
        return max(s.warmup() for s, _ in self._children())

    def generate(self, panel: Panel) -> pd.DataFrame:
        total = None
        for strat, w in self._children():
            # 子策略的 NaN（保持）先展开为持仓权重，再按资金占比加总
            held = validate_targets(strat.generate(panel), panel).ffill().fillna(0.0) * w
            total = held if total is None else total + held
        # 只在组合目标变化时给出新目标，其余为 NaN，避免对漂移的子仓位做无意义再平衡
        changed = total.ne(total.shift(1)).any(axis=1).to_numpy()
        return total.where(np.broadcast_to(changed[:, None], total.shape), np.nan)
