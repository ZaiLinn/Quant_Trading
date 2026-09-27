"""参数优化：网格搜索 + 滚动前向验证（Walk-Forward）。

网格搜索只能说明"过去哪组参数最好"，极易过拟合；
Walk-Forward 在训练窗选参、在紧随其后的样本外窗检验，拼接出的样本外曲线才是更可信的预期。
"""
from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .data.panel import Panel
from .market import MarketRules
from .strategy import get_strategy

log = logging.getLogger(__name__)

_CTX: dict = {}


def expand_grid(grid: dict[str, list]) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _init(ctx: dict) -> None:
    _CTX.clear()
    _CTX.update(ctx)


def _evaluate(params: dict) -> dict:
    c = _CTX
    try:
        strat = get_strategy(c["strategy"], **{**c["base_params"], **params})
        res = run_backtest(strat, c["panel"], c["rules"], c["initial_cash"], c["risk"], start=c["start"])
        m = res.metrics()
    except ValueError as e:  # 非法参数组合（如 fast >= slow）
        return {**params, "score": -np.inf, "error": str(e)}
    score = m.get(c["objective"], np.nan)
    if not np.isfinite(score) or m["trades"] < c["min_trades"]:
        score = -np.inf
    return {**params, "score": float(score), **{k: m[k] for k in
            ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar", "trades", "win_rate", "turnover")}}


def grid_search(strategy: str, panel: Panel, rules: MarketRules, grid: dict[str, list],
                base_params: dict | None = None, risk: dict | None = None,
                initial_cash: float = 1_000_000, objective: str = "sharpe", start=None,
                min_trades: int = 0, jobs: int = 1) -> pd.DataFrame:
    """返回按 score 降序的全部参数结果。objective 为 metrics 中任一越大越好的指标。"""
    combos = expand_grid(grid)
    ctx = dict(strategy=strategy, base_params=base_params or {}, panel=panel, rules=rules,
               risk=risk or {}, initial_cash=initial_cash, objective=objective, start=start,
               min_trades=min_trades)
    jobs = (os.cpu_count() or 1) if jobs == -1 else max(1, jobs)
    if jobs == 1 or len(combos) < 4:
        _init(ctx)
        rows = [_evaluate(p) for p in combos]
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=(ctx,)) as ex:
            rows = list(ex.map(_evaluate, combos, chunksize=max(1, len(combos) // (jobs * 4))))
    df = pd.DataFrame(rows)
    for k, vals in grid.items():  # 保持整数参数的类型（DataFrame 可能把它转成 float）
        if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            df[k] = df[k].astype(int)
    return df.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame          # 每个窗口：训练/测试区间、选出的参数、样本内/外得分
    oos: BacktestResult            # 拼接后的样本外回测
    efficiency: float              # 样本外得分均值 / 样本内得分均值（越接近 1 越不过拟合）


def _concat_results(parts: list[BacktestResult]) -> BacktestResult:
    first = parts[0]
    return BacktestResult(
        equity=pd.concat([p.equity for p in parts]),
        cash=pd.concat([p.cash for p in parts]),
        positions=pd.concat([p.positions for p in parts]).fillna(0.0),
        weights=pd.concat([p.weights for p in parts]).fillna(0.0),
        fills=pd.concat([p.fills for p in parts], ignore_index=True),
        trades=pd.concat([p.trades for p in parts], ignore_index=True),
        targets=pd.concat([p.targets for p in parts]),
        rules=first.rules, initial_cash=first.initial_cash, benchmark=None,
    )


def walk_forward(strategy: str, panel: Panel, rules: MarketRules, grid: dict[str, list],
                 train: int, test: int, base_params: dict | None = None, risk: dict | None = None,
                 initial_cash: float = 1_000_000, objective: str = "sharpe", anchored: bool = False,
                 min_trades: int = 0, jobs: int = 1) -> WalkForwardResult:
    base_params = base_params or {}
    T = len(panel)
    if train + test > T:
        raise ValueError(f"数据只有 {T} 根，不足以做 train={train} + test={test} 的前向验证")
    warm = max(get_strategy(strategy, **{**base_params, **p}).warmup() for p in expand_grid(grid))
    rows, parts, equity = [], [], float(initial_cash)
    k = 0
    while True:
        tr_s = 0 if anchored else k * test
        tr_e = train + k * test
        te_e = min(tr_e + test, T)
        if tr_e >= T:
            break
        tr_panel = panel.iloc(slice(max(0, tr_s - warm), tr_e))
        tr_start = panel.index[tr_s] if tr_s > 0 else None
        res = grid_search(strategy, tr_panel, rules, grid, base_params, risk, initial_cash,
                          objective, tr_start, min_trades, jobs)
        best = res.iloc[0]
        params = {c: next(v for v in grid[c] if v == best[c]) for c in grid}
        te_panel = panel.iloc(slice(max(0, tr_e - warm), te_e))
        strat = get_strategy(strategy, **{**base_params, **params})
        oos = run_backtest(strat, te_panel, rules, equity, risk, start=panel.index[tr_e])
        m = oos.metrics()
        equity = float(oos.equity.iloc[-1])
        parts.append(oos)
        rows.append({"train_start": panel.index[tr_s].date(), "train_end": panel.index[tr_e - 1].date(),
                     "test_start": panel.index[tr_e].date(), "test_end": panel.index[te_e - 1].date(),
                     **params, "is_score": best["score"], "oos_score": m.get(objective, np.nan),
                     "oos_return": m["total_return"]})
        log.info("窗口 %d: %s 样本内 %.3f 样本外 %.3f", k + 1, params, best["score"], rows[-1]["oos_score"])
        k += 1
        if te_e >= T:
            break
    windows = pd.DataFrame(rows)
    is_mean = windows["is_score"].replace(-np.inf, np.nan).mean()
    oos_mean = windows["oos_score"].replace([np.inf, -np.inf], np.nan).mean()
    eff = float(oos_mean / is_mean) if is_mean and np.isfinite(is_mean) and is_mean > 0 else np.nan
    return WalkForwardResult(windows=windows, oos=_concat_results(parts), efficiency=eff)
