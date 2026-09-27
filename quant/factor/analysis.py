"""单因子检验（思路来自 alphalens / qlib）：IC、分层收益、多空收益、换手与自相关。

前瞻收益按"t+1 开盘买入、t+1+h 开盘卖出"计算，与回测引擎的成交时点一致，
避免很多研究代码里"用 t 日收盘价因子、按 t 日收盘价成交"的隐性未来函数。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.panel import Panel
from .library import compute_factor


def forward_returns(panel: Panel, horizons=(1, 5, 20)) -> dict[int, pd.DataFrame]:
    o = panel.open
    entry = o.shift(-1)
    return {h: o.shift(-1 - h) / entry - 1 for h in horizons}


def _rowwise_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int) -> pd.Series:
    valid = a.notna() & b.notna()
    a, b = a.where(valid), b.where(valid)
    n = valid.sum(axis=1)
    am = a.sub(a.mean(axis=1), axis=0)
    bm = b.sub(b.mean(axis=1), axis=0)
    cov = (am * bm).sum(axis=1)
    den = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))
    return (cov / den.replace(0, np.nan)).where(n >= min_names)


def ic_series(fac: pd.DataFrame, fwd: pd.DataFrame, rank: bool = True, min_names: int = 5) -> pd.Series:
    """逐期截面相关系数；rank=True 为 RankIC（Spearman）。"""
    if rank:
        valid = fac.notna() & fwd.notna()
        fac = fac.where(valid).rank(axis=1)
        fwd = fwd.where(valid).rank(axis=1)
    return _rowwise_corr(fac, fwd, min_names).dropna()


def ic_stats(ic: pd.Series, horizon: int, periods_per_year: int = 252) -> dict:
    """IC 均值、标准差、ICIR、t 值、胜率。h>1 时样本重叠，t 值按 sqrt(n/h) 保守折算。"""
    n = len(ic)
    mean, std = ic.mean(), ic.std()
    icir = mean / std if std > 0 else np.nan
    return {"horizon": horizon, "ic_mean": mean, "ic_std": std, "icir": icir,
            "icir_ann": icir * np.sqrt(periods_per_year / horizon) if np.isfinite(icir) else np.nan,
            "t_stat": icir * np.sqrt(n / horizon) if np.isfinite(icir) else np.nan,
            "ic_pos": float((ic > 0).mean()) if n else np.nan, "n": n}


def quantile_labels(fac: pd.DataFrame, q: int) -> pd.DataFrame:
    """按截面排名分 q 组（1 = 因子值最小，q = 最大）。"""
    pct = fac.rank(axis=1, pct=True)
    return np.ceil(pct * q).clip(1, q)


def quantile_returns(fac: pd.DataFrame, fwd1: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """每期各分组等权平均的下一期收益（每期换仓、不计成本的研究视角）。"""
    lab = quantile_labels(fac, q)
    out = {}
    for k in range(1, q + 1):
        out[f"Q{k}"] = fwd1.where(lab == k).mean(axis=1)
    return pd.DataFrame(out)


def top_turnover(fac: pd.DataFrame, q: int = 5) -> float:
    """最高分组每期成分变化比例的均值。"""
    top = quantile_labels(fac, q) == q
    prev = top.shift(1, fill_value=False)
    changed = (top & ~prev).sum(axis=1)
    size = top.sum(axis=1).replace(0, np.nan)
    return float((changed / size).iloc[1:].mean())


def rank_autocorr(fac: pd.DataFrame, lag: int = 1) -> float:
    r = fac.rank(axis=1)
    return float(_rowwise_corr(r, r.shift(lag), 5).mean())


@dataclass
class FactorReport:
    spec: str
    ic: pd.DataFrame            # 每个 horizon 一行的统计
    ic_daily: pd.Series         # h=1 的 RankIC 序列
    quantile: pd.DataFrame      # 各分组每期收益
    turnover: float
    autocorr: float

    @property
    def long_short(self) -> pd.Series:
        cols = self.quantile.columns
        return (self.quantile[cols[-1]] - self.quantile[cols[0]]).fillna(0.0)

    def summary(self, periods_per_year: int = 252) -> dict:
        row = self.ic.set_index("horizon")
        h1 = row.iloc[0]
        qm = self.quantile.mean()
        ls = self.long_short
        # 分组单调性：分组均值收益的排名与组号的相关系数（Spearman，不依赖 scipy）
        mono = pd.Series(qm.rank().to_numpy()).corr(pd.Series(np.arange(len(qm), dtype=float)))
        return {"factor": self.spec, "rank_ic": h1["ic_mean"], "icir_ann": h1["icir_ann"], "t_stat": h1["t_stat"],
                "ic_pos": h1["ic_pos"], "ls_ann": float(ls.mean() * periods_per_year),
                "ls_sharpe": float(ls.mean() / ls.std() * np.sqrt(periods_per_year)) if ls.std() > 0 else np.nan,
                "monotonic": float(mono), "top_turnover": self.turnover, "autocorr": self.autocorr}


def evaluate(panel: Panel, spec: str, horizons=(1, 5, 20), q: int = 5, start=None,
             periods_per_year: int = 252, fac: pd.DataFrame | None = None) -> FactorReport:
    horizons = tuple(sorted(set(horizons)))
    fac = compute_factor(panel, spec) if fac is None else fac
    fwd = forward_returns(panel, sorted(set(horizons) | {1}))
    tradable = panel.open.shift(-1).notna() & (panel.volume.shift(-1) > 0)
    fac = fac.where(tradable)  # 次日停牌无法买入的样本剔除
    if start is not None:
        fac = fac.loc[start:]
        fwd = {h: f.loc[start:] for h, f in fwd.items()}
    ics = {h: ic_series(fac, fwd[h]) for h in horizons}
    stats = pd.DataFrame([ic_stats(ics[h], h, periods_per_year) for h in horizons])
    qr = quantile_returns(fac, fwd[1], q).dropna(how="all")
    return FactorReport(spec, stats, ic_series(fac, fwd[1]), qr, top_turnover(fac, q), rank_autocorr(fac))


def factor_html(rep: FactorReport, path: str | Path, periods_per_year: int = 252) -> Path:
    from ..analysis.report import date_labels, downsample_index, page, series_values

    cum = (1 + rep.quantile.fillna(0.0)).cumprod()
    ls = (1 + rep.long_short).cumprod()
    keep = downsample_index(cum.index)
    ic_roll = rep.ic_daily.rolling(60, min_periods=20).mean().reindex(cum.index)
    s = rep.summary(periods_per_year)
    data = {
        "title": f"因子检验 · {rep.spec}",
        "dates": date_labels(cum.index[keep]),
        "quantiles": [{"name": c, "values": series_values(cum[c][keep]), "color": f"--q{i + 1}"}
                      for i, c in enumerate(cum.columns[:5])],
        "ls": series_values(ls[keep]),
        "ic_roll": series_values(ic_roll[keep]),
        "tiles": [["RankIC 均值", f"{s['rank_ic']:.4f}"], ["年化 ICIR", f"{s['icir_ann']:.2f}"],
                  ["IC>0 占比", f"{s['ic_pos']:.1%}"], ["多空年化", f"{s['ls_ann']:.1%}"],
                  ["分组单调性", f"{s['monotonic']:.2f}"], ["头部换手", f"{s['top_turnover']:.1%}"]],
        "ic_table": [[int(r.horizon), f"{r.ic_mean:.4f}", f"{r.ic_std:.4f}", f"{r.icir_ann:.2f}",
                      f"{r.t_stat:.2f}", f"{r.ic_pos:.1%}", int(r.n)] for r in rep.ic.itertuples()],
        "q_table": [[c, f"{rep.quantile[c].mean() * periods_per_year:.2%}",
                     f"{cum[c].iloc[-1] - 1:.2%}"] for c in rep.quantile.columns],
    }
    return page(data["title"], _FACTOR_BODY, _FACTOR_SCRIPT, data, path)


_FACTOR_BODY = r"""
  <h1 id="title"></h1>
  <div class="sub" id="period"></div>
  <div class="tiles" id="tiles"></div>
  <section class="card">
    <h2>分组累计收益（Q1 因子值最小 → Q5 最大，每期等权换仓、未计成本）</h2>
    <div class="legend" id="q-legend"></div>
    <div class="chart" id="q-chart"></div>
  </section>
  <section class="card">
    <h2>多空组合累计净值（最高组 − 最低组）</h2>
    <div class="chart" id="ls-chart"></div>
  </section>
  <section class="card">
    <h2>RankIC 60 日滚动均值</h2>
    <div class="chart" id="ic-chart"></div>
  </section>
  <div class="grid2">
    <section class="card"><h2>分组收益</h2><table id="qt"></table></section>
    <section class="card"><h2>IC 统计（按持有期）</h2><div class="scroll"><table id="ict"></table></div>
      <div class="note">持有期 h&gt;1 时样本重叠，t 值已按 √(n/h) 折算；|t|&gt;3 才较有说服力。</div></section>
  </div>
"""

_FACTOR_SCRIPT = r"""
document.getElementById("title").textContent = D.title;
document.getElementById("period").textContent = `${D.dates[0]} ~ ${D.dates[D.dates.length - 1]}`;
renderTiles(document.getElementById("tiles"), D.tiles);
renderLegend(document.getElementById("q-legend"), D.quantiles);
lineChart(document.getElementById("q-chart"), D.dates, D.quantiles, {height: 320, fmt: v => v.toFixed(2)});
lineChart(document.getElementById("ls-chart"), D.dates, [{name: "多空", values: D.ls, color: "--s1"}],
  {height: 200, fmt: v => v.toFixed(2)});
lineChart(document.getElementById("ic-chart"), D.dates, [{name: "RankIC", values: D.ic_roll, color: "--s1"}],
  {height: 180, fmt: v => v.toFixed(3), zeroBase: true});
renderTable(document.getElementById("qt"), ["分组", "年化(单期均值×年化)", "累计"], D.q_table);
renderTable(document.getElementById("ict"), ["持有期", "IC均值", "IC标准差", "年化ICIR", "t值", "IC>0", "样本"], D.ic_table);
"""
