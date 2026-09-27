"""示例：自定义策略 + Python API 回测（不依赖网络，使用合成数据）。

运行：.venv/bin/python examples/custom_strategy.py
"""
from __future__ import annotations

import pandas as pd

from quant import get_rules, run_backtest
from quant import indicators as ind
from quant.analysis import html_report, print_summary
from quant.data import SyntheticSource, load_panel
from quant.strategy import Strategy, check_lookahead, hold_until, normalize_weights, register


@register
class VolumeBreakout(Strategy):
    """放量突破：收盘价创 N 日新高且成交量大于 M 日均量的 k 倍时入场，跌破 exit 日均线离场。"""

    name = "volume_breakout"
    description = "放量突破 N 日新高入场，跌破均线离场"
    params = {"n": 20, "m": 20, "k": 1.5, "exit": 10}

    def generate(self, panel) -> pd.DataFrame:
        c, v = panel.close, panel.volume
        high_n = c.rolling(self.n, min_periods=self.n).max().shift(1)  # 不含当根，避免未来函数
        entries = (c > high_n) & (v > self.k * ind.sma(v, self.m))
        exits = c < ind.sma(c, self.exit)
        return normalize_weights(hold_until(entries, exits))


def main() -> None:
    panel = load_panel(SyntheticSource(), ["AAA", "BBB", "CCC", "DDD"], "2016-01-01", "2024-12-31")
    rules = get_rules("ashare")  # A 股：100 股一手、T+1、涨跌停、印花税、最低 5 元佣金
    strat = VolumeBreakout(n=30, k=1.3)

    problems = check_lookahead(strat, panel)
    print("未来函数检测:", "通过" if not problems else problems)

    result = run_backtest(strat, panel, rules, initial_cash=500_000,
                          risk={"trailing_stop": 0.12, "max_weight": 0.5},
                          benchmark=panel.close["AAA"], start="2017-01-01")
    m = result.metrics()
    print_summary(m, repr(strat))
    path = html_report(result, m, "runs/example_volume_breakout.html", "放量突破示例")
    print("报告:", path.resolve())


if __name__ == "__main__":
    main()
