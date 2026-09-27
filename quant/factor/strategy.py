"""因子选股策略：多因子打分 -> 定期持有得分最高的 top_k 只（带缓冲区降低换手）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..strategy.base import Strategy, rebalance_mask, register
from .library import combine, factor_warmup


@register
class FactorTopK(Strategy):
    name = "factor_topk"
    description = "多因子选股：因子截面标准化加权打分，定期持有前 top_k 名；已持有标的排名仍在 top_k+buffer 内则不换（参考 qlib TopkDropout）"
    params = {"factors": {"momentum_60_5": 1.0, "volatility_20": -1.0}, "top_k": 10, "buffer": 5,
              "rebalance": "W", "min_bars": 60}

    def warmup(self) -> int:
        return max(factor_warmup(s) for s in self.factors) + 10

    def generate(self, panel: Panel) -> pd.DataFrame:
        score = combine(panel, self.factors)
        # 剔除上市不足 min_bars 根的新股（次新股波动极端、且早期数据不足）
        score = score.where(panel.close.notna().cumsum() >= self.min_bars)
        rank = score.rank(axis=1, ascending=False)
        reb = rebalance_mask(panel.index, self.rebalance).to_numpy()
        cols = list(panel.symbols)
        out = np.full((len(panel), len(cols)), np.nan)
        held: list[str] = []
        R = rank.to_numpy()
        pos = {c: i for i, c in enumerate(cols)}
        for t in np.nonzero(reb)[0]:
            r = R[t]
            if np.isnan(r).all():
                continue
            keep = [s for s in held if np.isfinite(r[pos[s]]) and r[pos[s]] <= self.top_k + self.buffer]
            order = [cols[i] for i in np.argsort(np.where(np.isfinite(r), r, np.inf)) if np.isfinite(r[i])]
            new = [s for s in order if s not in keep][: max(self.top_k - len(keep), 0)]
            held = keep + new
            row = np.zeros(len(cols))
            for s in held:
                row[pos[s]] = 1.0 / self.top_k
            out[t] = row
        return pd.DataFrame(out, index=panel.index, columns=cols)
