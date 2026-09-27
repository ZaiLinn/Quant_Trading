"""绩效指标（自实现，不依赖已停更的 empyrical/pyfolio）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown_duration(equity: pd.Series) -> int:
    """最长回撤持续期（K 线根数，从前高到收复前高）。"""
    underwater = (equity < equity.cummax()).to_numpy()
    longest = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return int(longest)


def sharpe(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    ex = returns - rf / periods_per_year
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


def sortino(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    ex = returns - rf / periods_per_year
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else np.nan


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return np.nan
    return float((max(equity.iloc[-1], 0) / equity.iloc[0]) ** (1 / years) - 1)


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """年 × 月 收益表，最后一列为全年。"""
    m = equity.resample("ME").last().pct_change()
    first = equity.resample("ME").last().iloc[0] / equity.iloc[0] - 1
    m.iloc[0] = first
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.to_numpy()})
    out = tbl.pivot(index="year", columns="month", values="ret")
    out = out.reindex(columns=range(1, 13))
    yearly = equity.resample("YE").last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / equity.iloc[0] - 1
    out["year"] = yr.to_numpy()
    return out


def trade_stats(trades: pd.DataFrame) -> dict:
    closed = trades[trades["exit_reason"] != "open"] if len(trades) else trades
    if not len(closed):
        return {"trades": 0, "win_rate": np.nan, "profit_factor": np.nan, "avg_trade_return": np.nan,
                "avg_win": np.nan, "avg_loss": np.nan, "avg_bars_held": np.nan}
    pnl = closed["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_loss = -losses.sum()
    return {
        "trades": int(len(closed)),
        "win_rate": float(len(wins) / len(closed)),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else np.inf,
        "avg_trade_return": float(closed["return"].mean()),
        "avg_win": float(closed.loc[pnl > 0, "return"].mean()) if len(wins) else np.nan,
        "avg_loss": float(closed.loc[pnl <= 0, "return"].mean()) if len(losses) else np.nan,
        "avg_bars_held": float(closed["bars"].mean()),
    }


def compute_metrics(result, rf: float = 0.0) -> dict:
    from ..pipeline import bars_per_year

    eq = result.equity
    ppy = bars_per_year(eq.index, result.rules)
    rets = eq.pct_change().fillna(0.0)
    dd = drawdown(eq)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    c = cagr(eq)
    mdd = float(dd.min())
    traded = result.fills["value"].sum() if len(result.fills) else 0.0
    m = {
        "start": eq.index[0].strftime("%Y-%m-%d"),
        "end": eq.index[-1].strftime("%Y-%m-%d"),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": c,
        "ann_vol": float(rets.std() * np.sqrt(ppy)),
        "sharpe": sharpe(rets, ppy, rf),
        "sortino": sortino(rets, ppy, rf),
        "max_drawdown": mdd,
        "max_dd_bars": max_drawdown_duration(eq),
        "calmar": float(c / -mdd) if mdd < 0 else np.nan,
        "exposure": float((result.positions.abs().sum(axis=1) > 0).mean()),
        "turnover": float(traded / eq.mean() / years),  # 年化双边换手（倍）
        "fees": float(result.fills["fee"].sum()) if len(result.fills) else 0.0,
    }
    m.update(trade_stats(result.trades))
    if result.benchmark is not None and len(result.benchmark) > 2:
        b = result.benchmark.reindex(eq.index).ffill()
        br = b.pct_change().fillna(0.0)
        m["bench_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
        m["bench_cagr"] = cagr(b.dropna())
        m["bench_max_drawdown"] = float(drawdown(b.dropna()).min())
        var = br.var()
        beta = float(rets.cov(br) / var) if var > 0 else np.nan
        m["beta"] = beta
        m["alpha"] = float((rets.mean() - beta * br.mean()) * ppy) if np.isfinite(beta) else np.nan
        active = rets - br
        te = active.std()
        m["info_ratio"] = float(active.mean() / te * np.sqrt(ppy)) if te > 0 else np.nan
    return m


PCT_KEYS = {"total_return", "cagr", "ann_vol", "max_drawdown", "exposure", "win_rate",
            "avg_trade_return", "avg_win", "avg_loss", "bench_return", "bench_cagr",
            "bench_max_drawdown", "alpha"}

LABELS = {
    "start": "开始", "end": "结束", "final_equity": "期末权益", "total_return": "总收益",
    "cagr": "年化收益", "ann_vol": "年化波动", "sharpe": "夏普", "sortino": "索提诺",
    "max_drawdown": "最大回撤", "max_dd_bars": "最长回撤(根)", "calmar": "卡玛",
    "exposure": "持仓时间占比", "turnover": "年换手(倍)", "fees": "总费用", "trades": "交易次数",
    "win_rate": "胜率", "profit_factor": "盈亏比(总)", "avg_trade_return": "平均每笔收益",
    "avg_win": "平均盈利", "avg_loss": "平均亏损", "avg_bars_held": "平均持仓(根)",
    "bench_return": "基准总收益", "bench_cagr": "基准年化", "bench_max_drawdown": "基准最大回撤",
    "alpha": "Alpha(年化)", "beta": "Beta", "info_ratio": "信息比率",
}


def format_metrics(m: dict) -> list[tuple[str, str]]:
    rows = []
    for k, v in m.items():
        if isinstance(v, float):
            if np.isnan(v):
                s = "-"
            elif k in PCT_KEYS:
                s = f"{v:.2%}"
            elif k in ("final_equity", "fees"):
                s = f"{v:,.2f}"
            else:
                s = f"{v:.2f}"
        else:
            s = str(v)
        rows.append((LABELS.get(k, k), s))
    return rows
