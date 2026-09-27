import numpy as np
import pandas as pd
import pytest

from quant.data import Panel, SyntheticSource, load_panel


@pytest.fixture(scope="session")
def panel() -> Panel:
    return load_panel(SyntheticSource(), ["AAA", "BBB", "CCC"], "2018-01-01", "2022-12-31")


def make_panel(prices: dict[str, list[float]], opens: dict[str, list[float]] | None = None,
               start: str = "2024-01-01", volume: float = 1e6) -> Panel:
    """用给定价格构造面板；high/low 取开收盘的最大/最小值。"""
    frames = {}
    for sym, closes in prices.items():
        c = np.asarray(closes, float)
        o = np.asarray(opens[sym], float) if opens and sym in opens else c.copy()
        idx = pd.bdate_range(start, periods=len(c))
        frames[sym] = pd.DataFrame({"open": o, "high": np.fmax(o, c), "low": np.fmin(o, c),
                                    "close": c, "volume": volume}, index=idx)
    return Panel(frames)


def const_targets(panel: Panel, weights: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(weights, index=panel.index)
