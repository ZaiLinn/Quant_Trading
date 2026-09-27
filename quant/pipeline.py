"""策略输出 -> 风控调整后的目标权重。回测与实盘调用同一个函数。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .data.panel import Panel
from .market import MarketRules
from .strategy.validate import validate_targets


def bars_per_year(index: pd.DatetimeIndex, rules: MarketRules) -> float:
    """根据 K 线周期推断年化因子：日线用市场规则（A 股 252 / 加密 365），分钟线乘以每日根数。"""
    if len(index) < 3:
        return float(rules.periods_per_year)
    step = pd.Series(index).diff().median()
    if step < pd.Timedelta(hours=20):
        per_day = pd.Series(1, index=index).groupby(index.normalize()).count().median()
        return float(rules.periods_per_year * per_day)
    days = step / pd.Timedelta(days=1)
    return float(rules.periods_per_year) if days <= 1.5 else 365.25 / days


def market_proxy(panel: Panel) -> pd.Series:
    """股票池等权指数（各标的日收益的截面均值累乘），用作大盘择时的参照。"""
    c = panel.close.ffill()
    ret = (c / c.shift(1) - 1).mean(axis=1).fillna(0.0)
    return (1 + ret).cumprod()


def apply_risk(targets: pd.DataFrame, panel: Panel, rules: MarketRules,
               max_weight: float | None = None, vol_target: float | None = None,
               vol_window: int = 20, weight_step: float | None = None,
               regime_ma: int | None = None, regime_scale: float = 0.0) -> pd.DataFrame:
    """对目标权重做组合层面的风控调整（保留 NaN = 不调仓 的语义）。

    - vol_target：按标的波动率缩放仓位，使组合年化波动率不超过目标（假设相关性为 1，偏保守）；
    - max_weight：单标的权重上限；
    - 总敞口不超过 rules.max_leverage；
    - weight_step：权重按步长取整，过滤微小变动带来的无效换手；
    - regime_ma：大盘择时，股票池等权指数跌破 N 日均线时仓位乘以 regime_scale（0 = 空仓）。
    """
    w = targets.copy()
    live = w.notna()
    if regime_ma:
        idx = market_proxy(panel)
        bear = (idx < idx.rolling(regime_ma, min_periods=regime_ma).mean()).to_numpy()
        # 市场状态切换当天强制给出目标（否则 NaN=不调仓 的行会错过切换）
        switch = pd.Series(bear, index=w.index).ne(pd.Series(bear, index=w.index).shift(1))
        w = w.ffill().fillna(0.0).where(live | switch.to_numpy()[:, None])
        live = w.notna()
        w = w.mul(np.where(bear, regime_scale, 1.0), axis=0)
    if vol_target:
        vol = ind.realized_vol(panel.close.ffill(), vol_window, int(bars_per_year(panel.index, rules)))
        port_vol = (w.abs() * vol).sum(axis=1, min_count=1)
        scale = (vol_target / port_vol.replace(0, np.nan)).clip(upper=rules.max_leverage)
        w = w.mul(scale.fillna(1.0), axis=0)
    if max_weight is not None:
        w = w.clip(-max_weight, max_weight)
    gross = w.abs().sum(axis=1)
    cap = (rules.max_leverage / gross.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    w = w.mul(cap, axis=0)
    if weight_step:
        w = (np.floor(w.abs() / weight_step + 1e-9) * weight_step * np.sign(w)).round(10)
    return w.where(live)


def build_targets(strategy, panel: Panel, rules: MarketRules, risk: dict | None = None) -> pd.DataFrame:
    risk = risk or {}
    raw = validate_targets(strategy.generate(panel), panel)
    return apply_risk(raw, panel, rules,
                      max_weight=risk.get("max_weight"),
                      vol_target=risk.get("vol_target"),
                      vol_window=risk.get("vol_window", 20),
                      weight_step=risk.get("weight_step"),
                      regime_ma=risk.get("regime_ma"),
                      regime_scale=risk.get("regime_scale", 0.0))
