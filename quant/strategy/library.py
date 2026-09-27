"""内置策略。每个策略几十行，向量化实现，便于阅读和改写。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..data.panel import Panel
from .base import Strategy, hold_until, normalize_weights, on_rebalance, rebalance_mask, register


@register
class BuyAndHold(Strategy):
    name = "buy_and_hold"
    description = "等权买入并持有（基准/对照组）"
    params = {}

    def generate(self, panel: Panel) -> pd.DataFrame:
        listed = panel.close.notna().astype(float)
        w = normalize_weights(listed)
        # 只在标的集合变化时调仓，避免每天被动再平衡
        changed = listed.ne(listed.shift(1)).any(axis=1)
        return w.where(changed, np.nan)

    def warmup(self) -> int:
        return 1


@register
class SmaCross(Strategy):
    name = "sma_cross"
    description = "双均线趋势：快线在慢线之上持有（可选做空），多标的等权"
    params = {"fast": 10, "slow": 30, "allow_short": False}

    def generate(self, panel: Panel) -> pd.DataFrame:
        if self.fast >= self.slow:
            raise ValueError("fast 必须小于 slow")
        c = panel.close
        f, s = ind.sma(c, self.fast), ind.sma(c, self.slow)
        sig = (f > s).astype(float)
        if self.allow_short:
            sig = sig - (f < s).astype(float)
        return normalize_weights(sig.where(s.notna(), 0.0))


@register
class MacdTrend(Strategy):
    name = "macd"
    description = "MACD 趋势：DIF 上穿 DEA 且（可选）DIF>0 时持有"
    params = {"fast": 12, "slow": 26, "signal": 9, "zero_filter": True}

    def generate(self, panel: Panel) -> pd.DataFrame:
        dif, dea, _ = ind.macd(panel.close, self.fast, self.slow, self.signal)
        long = dif > dea
        if self.zero_filter:
            long &= dif > 0
        return normalize_weights(long.astype(float))


@register
class RsiReversion(Strategy):
    name = "rsi_reversion"
    description = "RSI 均值回归：超卖买入、回到中性卖出；可加长期均线过滤只做上升趋势中的回调"
    params = {"n": 14, "entry": 30.0, "exit": 55.0, "trend": 200}

    def generate(self, panel: Panel) -> pd.DataFrame:
        c = panel.close
        r = ind.rsi(c, self.n)
        entries = r < self.entry
        if self.trend:
            entries &= c > ind.sma(c, self.trend)
        pos = hold_until(entries, r > self.exit)
        return normalize_weights(pos)


@register
class BollingerReversion(Strategy):
    name = "bollinger_reversion"
    description = "布林带回归：收盘跌破下轨买入，回到中轨卖出"
    params = {"n": 20, "k": 2.0}

    def generate(self, panel: Panel) -> pd.DataFrame:
        c = panel.close
        mid, _, lower = ind.bollinger(c, self.n, self.k)
        pos = hold_until(c < lower, c > mid)
        return normalize_weights(pos)


@register
class DonchianBreakout(Strategy):
    name = "donchian_breakout"
    description = "海龟式通道突破：突破 N 日高点入场、跌破 M 日低点离场，按 ATR 风险定仓"
    params = {"entry": 55, "exit": 20, "atr_n": 20, "risk": 0.01, "atr_mult": 2.0,
              "max_weight": 0.5, "allow_short": False}

    def generate(self, panel: Panel) -> pd.DataFrame:
        h, l, c = panel.high, panel.low, panel.close
        up_e, lo_e = ind.donchian(h, l, self.entry)
        up_x, lo_x = ind.donchian(h, l, self.exit)
        long = hold_until(c > up_e, c < lo_x)
        pos = long
        if self.allow_short:
            short = hold_until(c < lo_e, c > up_x)
            pos = long - short
        # 风险定仓：单笔止损距离 atr_mult*ATR 对应 risk 比例的权益。仓位在入场时锁定，持仓期间不随 ATR 抖动
        a = ind.atr(h, l, c, self.atr_n)
        size = (self.risk / (self.atr_mult * a / c)).clip(upper=self.max_weight)
        entry_bar = (pos != 0) & (pos != pos.shift(1))
        locked = size.where(entry_bar).ffill()
        w = (pos * locked).fillna(0.0)
        gross = w.abs().sum(axis=1)
        scale = (1.0 / gross).clip(upper=1.0).fillna(1.0)  # 总敞口不超过 1
        return w.mul(scale, axis=0)


@register
class MomentumRotation(Strategy):
    name = "momentum_rotation"
    description = "动量轮动：定期买入过去 N 期涨幅最高的 K 个标的；绝对动量过滤为负时空仓"
    params = {"lookback": 20, "top_k": 1, "rebalance": "W", "abs_momentum": True, "skip": 0}

    def generate(self, panel: Panel) -> pd.DataFrame:
        c = panel.close
        score = ind.roc(c.shift(self.skip), self.lookback)
        rank = score.rank(axis=1, ascending=False, method="first")
        pick = (rank <= self.top_k) & score.notna()
        if self.abs_momentum:
            pick &= score > 0
        # 未选满 top_k 时剩余资金留作现金（与 top_k 挂钩，而非把钱全压在少数标的上）
        w = pick.astype(float) / self.top_k
        return on_rebalance(w, self.rebalance)

    def warmup(self) -> int:
        return self.lookback + self.skip + 10


@register
class KdjCross(Strategy):
    name = "kdj_cross"
    description = "KDJ 金叉买入、死叉卖出，可限定金叉发生在超卖区"
    params = {"n": 9, "m1": 3, "m2": 3, "oversold": 100.0}

    def generate(self, panel: Panel) -> pd.DataFrame:
        k, d, _ = ind.kdj(panel.high, panel.low, panel.close, self.n, self.m1, self.m2)
        entries = ind.crossover(k, d) & (d < self.oversold)
        pos = hold_until(entries, ind.crossunder(k, d))
        return normalize_weights(pos)


@register
class RiskParity(Strategy):
    name = "risk_parity"
    description = "风险平价：按波动率倒数（inverse_vol）或等风险贡献（erc，考虑相关性）定期配置全部标的"
    params = {"window": 60, "rebalance": "M", "method": "erc"}

    def generate(self, panel: Panel) -> pd.DataFrame:
        c = panel.close.ffill()
        rets = np.log(c / c.shift(1))
        reb = rebalance_mask(panel.index, self.rebalance)
        out = pd.DataFrame(np.nan, index=panel.index, columns=panel.symbols)
        R = rets.to_numpy()
        for t in np.nonzero(reb.to_numpy())[0]:
            if t < self.window:
                out.iloc[t] = 0.0
                continue
            win = R[t - self.window + 1: t + 1]
            ok = np.isfinite(win).all(axis=0) & (np.nanstd(win, axis=0) > 0)
            w = np.zeros(len(panel.symbols))
            if ok.sum() == 1:
                w[ok] = 1.0
            elif ok.sum() > 1:
                cov = np.cov(win[:, ok], rowvar=False)
                w[ok] = erc_weights(cov) if self.method == "erc" else _inv_vol(cov)
            out.iloc[t] = w
        return out

    def warmup(self) -> int:
        return self.window + 5


def _inv_vol(cov: np.ndarray) -> np.ndarray:
    iv = 1 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def erc_weights(cov: np.ndarray, iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """等风险贡献权重（乘法迭代，Σ 正定时收敛）：每个资产对组合波动的贡献相等。"""
    w = _inv_vol(cov)
    for _ in range(iters):
        rc = w * (cov @ w)
        new = w * np.sqrt(rc.mean() / np.maximum(rc, 1e-18))
        new /= new.sum()
        if np.abs(new - w).max() < tol:
            return new
        w = new
    return w


@register
class GridTrading(Strategy):
    name = "grid"
    description = "网格交易：价格在区间内越低仓位越高，按网格格数离散化，只在跨越网格线时调仓；区间可固定，或按均线±N倍标准差定期重置"
    params = {"levels": 10, "lower": None, "upper": None, "window": 60, "width": 2.0, "reset": "W"}

    def generate(self, panel: Panel) -> pd.DataFrame:
        c = panel.close
        if self.lower is not None and self.upper is not None:
            lo = pd.DataFrame(float(self.lower), index=c.index, columns=c.columns)
            hi = pd.DataFrame(float(self.upper), index=c.index, columns=c.columns)
        else:
            # 均线 ± width 倍标准差，只在每个 reset 周期首根重置：区间每天漂移会导致网格线跟着动，
            # 价格不变也反复买卖（实测 BTC 小时线换手 300 倍/年，手续费吃掉 20%+ 本金）
            mid = ind.sma(c, self.window)
            sd = c.rolling(self.window, min_periods=self.window).std()
            reset = rebalance_mask(c.index, self.reset)
            lo = (mid - self.width * sd).where(reset, axis=0).ffill()
            hi = (mid + self.width * sd).where(reset, axis=0).ffill()
        frac = ((hi - c) / (hi - lo).replace(0, np.nan)).clip(0, 1)
        desired = (frac * self.levels).to_numpy()
        # 带一格滞回的网格：价格比上次成交再跌一整格才加一格仓位、再涨一整格才减一格。
        # 否则价格在同一条网格线附近来回穿越会"原价买、原价卖"，白白付手续费。
        units = np.zeros_like(desired)
        u = np.zeros(desired.shape[1])
        for t in range(len(desired)):
            d = desired[t]
            ok = np.isfinite(d)
            up = ok & (d >= u + 1)
            down = ok & (d <= u - 1)
            u = np.where(up, np.floor(d), np.where(down, np.ceil(d), u))
            u = np.where(ok, u, 0.0)
            units[t] = u
        w = pd.DataFrame(units / self.levels, index=c.index, columns=c.columns)
        return w / len(panel.symbols)
