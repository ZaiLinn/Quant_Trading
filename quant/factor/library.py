"""因子库。命名规则 "<因子名>_<参数1>_<参数2>"，例如 momentum_60_5 = 60 日动量、跳过最近 5 日。

所有因子在 t 时刻只使用 t 及以前的数据。因子值越大代表预期收益越高还是越低，
由组合时的权重符号决定（例如低波动异象：volatility_20 取 -1）。
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..data.panel import Panel

FACTORS: dict[str, tuple[Callable, str, tuple]] = {}


def factor(name: str, doc: str, defaults: tuple = ()):
    def deco(fn):
        FACTORS[name] = (fn, doc, defaults)
        return fn
    return deco


def _ret(p: Panel) -> pd.DataFrame:
    c = p.close.ffill()
    return c / c.shift(1) - 1


@factor("momentum", "N 日动量（可跳过最近 skip 日以避开短期反转）", (20, 0))
def momentum(p: Panel, n: int = 20, skip: int = 0) -> pd.DataFrame:
    c = p.close.ffill()
    return c.shift(skip) / c.shift(n + skip) - 1


@factor("reversal", "N 日反转（= 负的短期收益）", (5,))
def reversal(p: Panel, n: int = 5) -> pd.DataFrame:
    return -momentum(p, n)


@factor("volatility", "N 日收益波动率", (20,))
def volatility(p: Panel, n: int = 20) -> pd.DataFrame:
    return _ret(p).rolling(n, min_periods=n).std()


@factor("downvol", "N 日下行波动率", (20,))
def downside_vol(p: Panel, n: int = 20) -> pd.DataFrame:
    r = _ret(p)
    return np.sqrt((r.clip(upper=0) ** 2).rolling(n, min_periods=n).mean())


@factor("bias", "收盘价相对 N 日均线的偏离", (20,))
def ma_bias(p: Panel, n: int = 20) -> pd.DataFrame:
    c = p.close.ffill()
    return c / ind.sma(c, n) - 1


@factor("rsi", "N 日 RSI", (14,))
def rsi(p: Panel, n: int = 14) -> pd.DataFrame:
    return ind.rsi(p.close.ffill(), n)


@factor("volratio", "短期(5日)/长期(N日)成交量比", (20,))
def volume_ratio(p: Panel, n: int = 20) -> pd.DataFrame:
    v = p.volume
    return v.rolling(5, min_periods=5).mean() / v.rolling(n, min_periods=n).mean().replace(0, np.nan)


@factor("amihud", "Amihud 非流动性：|收益| / 成交额 的 N 日均值（取对数）", (20,))
def amihud(p: Panel, n: int = 20) -> pd.DataFrame:
    amount = (p.volume * p.close).replace(0, np.nan)
    return np.log((_ret(p).abs() / amount).rolling(n, min_periods=n // 2).mean() * 1e9 + 1e-12)


@factor("maxret", "N 日内最大单日收益（彩票效应）", (20,))
def max_return(p: Panel, n: int = 20) -> pd.DataFrame:
    return _ret(p).rolling(n, min_periods=n).max()


@factor("skew", "N 日收益偏度", (20,))
def skewness(p: Panel, n: int = 20) -> pd.DataFrame:
    return _ret(p).rolling(n, min_periods=n).skew()


@factor("trend", "N 日对数价格与时间的相关系数（趋势平滑度）", (20,))
def trend(p: Panel, n: int = 20) -> pd.DataFrame:
    lc = np.log(p.close.ffill())
    t = pd.Series(np.arange(len(lc), dtype=float), index=lc.index)
    return lc.rolling(n, min_periods=n).corr(t)


@factor("sharpe", "N 日收益均值 / 波动（风险调整动量）", (60,))
def rolling_sharpe(p: Panel, n: int = 60) -> pd.DataFrame:
    r = _ret(p)
    return r.rolling(n, min_periods=n).mean() / r.rolling(n, min_periods=n).std().replace(0, np.nan)


def parse_spec(spec: str) -> tuple[str, list[int]]:
    parts = spec.split("_")
    name, args = parts[0], parts[1:]
    if name not in FACTORS:
        raise ValueError(f"未知因子 {name!r}，可选: {sorted(FACTORS)}")
    try:
        return name, [int(a) for a in args]
    except ValueError:
        raise ValueError(f"因子参数必须为整数: {spec}") from None


def compute_factor(panel: Panel, spec: str) -> pd.DataFrame:
    name, args = parse_spec(spec)
    fn, _, defaults = FACTORS[name]
    args = args + list(defaults[len(args):])
    out = fn(panel, *args)
    # 停牌/未上市处置空，避免 ffill 后的陈旧价格产生伪信号
    return out.where(panel.close.notna())


def factor_warmup(spec: str) -> int:
    _, args = parse_spec(spec)
    return sum(args) if args else 60


def zscore_cs(f: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """截面标准化并按 ±winsor 个标准差缩尾（去极值）。"""
    mu = f.mean(axis=1)
    sd = f.std(axis=1).replace(0, np.nan)
    z = f.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-winsor, winsor)


def combine(panel: Panel, weights: dict[str, float]) -> pd.DataFrame:
    """多因子合成：各因子截面 z-score 后按权重相加（某因子缺失时用其余因子）。"""
    total = None
    for spec, w in weights.items():
        z = zscore_cs(compute_factor(panel, spec)) * float(w)
        total = z if total is None else total.add(z, fill_value=0.0).where(total.notna() | z.notna())
    return total
