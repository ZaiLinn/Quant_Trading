"""行情数据质量检查：缺失、零成交、异常跳变、价格不一致。

免费数据源常见问题：复权口径混用（出现 ±30% 以上的"假跳空"）、停牌日缺行、零价格、高低价颠倒。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..market import MarketRules
from .panel import Panel


def quality_report(panel: Panel, rules: MarketRules) -> pd.DataFrame:
    """每个标的一行：有效 K 线数、缺失率、零成交、最大单根涨跌、超限跳变次数与日期。"""
    rows = []
    for sym in panel.symbols:
        c = panel.close[sym]
        first, last = c.first_valid_index(), c.last_valid_index()
        if first is None:
            rows.append({"symbol": sym, "bars": 0, "issues": "无数据"})
            continue
        span = c.loc[first:last]
        ret = span.ffill().pct_change()
        limit = rules.limit_for(sym)
        # 超过涨跌停幅度 1.5 倍（或无涨跌停市场的 50%）视为可疑跳变（复权/拆分/数据错误）
        thresh = limit * 1.5 if limit else 0.5
        jumps = ret[ret.abs() > thresh]
        vol = panel.volume[sym].loc[first:last]
        hi, lo, op = (panel[f][sym].loc[first:last] for f in ("high", "low", "open"))
        bad_ohlc = int(((hi < lo) | (op > hi * 1.0001) | (op < lo * 0.9999)).sum())
        issues = []
        if len(jumps):
            issues.append(f"异常跳变 {len(jumps)} 次(如 {jumps.index[0]:%Y-%m-%d} {jumps.iloc[0]:+.1%})")
        if bad_ohlc:
            issues.append(f"OHLC 不一致 {bad_ohlc} 根")
        stale = int((span.ffill().diff() == 0).rolling(10).sum().eq(10).sum())
        if stale:
            issues.append(f"连续 10 根价格不变 {stale} 处")
        rows.append({
            "symbol": sym, "start": first.strftime("%Y-%m-%d"), "end": last.strftime("%Y-%m-%d"),
            "bars": int(span.notna().sum()), "missing": float(span.isna().mean()),
            "zero_volume": int((vol.fillna(0) <= 0).sum()),
            "max_up": float(ret.max()) if len(ret.dropna()) else np.nan,
            "max_down": float(ret.min()) if len(ret.dropna()) else np.nan,
            "issues": "；".join(issues),
        })
    return pd.DataFrame(rows)
