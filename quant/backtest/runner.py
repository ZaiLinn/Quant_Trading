"""一行跑回测：策略 -> 风控 -> 引擎。"""
from __future__ import annotations

import pandas as pd

from ..data.panel import Panel
from ..market import MarketRules
from ..pipeline import build_targets
from .engine import BacktestEngine, BacktestResult

ENGINE_KEYS = ("stop_loss", "take_profit", "trailing_stop", "drift_threshold", "min_order_value",
               "max_volume_pct")


def run_backtest(strategy, panel: Panel, rules: MarketRules, initial_cash: float = 1_000_000,
                 risk: dict | None = None, benchmark: pd.Series | None = None,
                 start=None) -> BacktestResult:
    """start：从该日期开始交易，之前的数据只用于指标预热（避免预热期空仓拉低年化）。"""
    risk = risk or {}
    targets = build_targets(strategy, panel, rules, risk)
    engine = BacktestEngine(rules, initial_cash, **{k: risk.get(k) for k in ENGINE_KEYS if risk.get(k) is not None})
    if start is not None:
        carried = targets.ffill()
        panel = panel.slice(start, None)
        targets = targets.loc[panel.index].copy()
        # 开始交易的第一根沿用此前最近一次的目标：定期调仓（NaN=保持）的策略不必空仓等到下个调仓日
        if len(targets):
            targets.iloc[0] = targets.iloc[0].fillna(carried.loc[panel.index[0]])
    result = engine.run(panel, targets, benchmark)
    result.meta["strategy"] = repr(strategy)
    return result
