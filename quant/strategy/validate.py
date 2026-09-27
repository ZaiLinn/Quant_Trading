"""未来函数检测。

原理：若策略在 t 时刻的输出只依赖 t 及以前的数据，那么"用全量数据算出的 t 行"
必须等于"只用截至 t 的数据算出的最后一行"。shift(-1)、居中滚动窗口、全样本标准化等
都会被这个截断测试抓出来。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.panel import Panel


def check_lookahead(strategy, panel: Panel, n_checks: int = 8, tol: float = 1e-9,
                    seed: int = 0) -> list[str]:
    """返回发现问题的描述列表；空列表表示未发现未来函数。"""
    full = strategy.generate(panel)
    rng = np.random.default_rng(seed)
    lo = min(strategy.warmup(), len(panel) - 2)
    if lo >= len(panel) - 1:
        return []
    points = sorted(set(rng.integers(lo, len(panel) - 1, size=n_checks).tolist()))
    problems = []
    for t in points:
        part = strategy.generate(panel.iloc(slice(0, t + 1)))
        a = full.iloc[t].to_numpy(float)
        b = part.iloc[-1].to_numpy(float)
        same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)
        if not same.all():
            bad = [s for s, ok in zip(panel.symbols, same) if not ok]
            problems.append(f"{panel.index[t]}: {bad} 全量={a[~same]} 截断={b[~same]}")
    return problems


def validate_targets(targets: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    """对齐并检查策略输出。"""
    if not isinstance(targets, pd.DataFrame):
        raise TypeError("generate() 必须返回 DataFrame")
    extra = set(targets.columns) - set(panel.symbols)
    if extra:
        raise ValueError(f"目标权重包含未知标的: {sorted(extra)}")
    t = targets.reindex(index=panel.index, columns=panel.symbols).astype(float)
    if np.isinf(t.to_numpy()).any():
        raise ValueError("目标权重包含 inf")
    return t
