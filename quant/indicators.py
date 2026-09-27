"""技术指标（纯 pandas 实现，Series 与 DataFrame(时间 × 标的) 通用）。

所有指标只用当期及以前的数据，不会引入未来函数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Data = pd.Series | pd.DataFrame


def sma(x: Data, n: int) -> Data:
    return x.rolling(n, min_periods=n).mean()


def ema(x: Data, n: int) -> Data:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x: Data, n: int) -> Data:
    """Wilder 平滑（RSI/ATR 使用），等价于 alpha=1/n 的 EMA。"""
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def roc(x: Data, n: int) -> Data:
    """n 期收益率。"""
    return x / x.shift(n) - 1


def rsi(close: Data, n: int = 14) -> Data:
    diff = close.diff()
    up = wilder(diff.clip(lower=0), n)
    down = wilder(-diff.clip(upper=0), n)
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # 只涨不跌时 RSI=100，完全横盘时取 50
    return out.mask((down == 0) & (up > 0), 100.0).mask((down == 0) & (up == 0), 50.0)


def macd(close: Data, fast: int = 12, slow: int = 26, signal: int = 9):
    """返回 (dif, dea, hist)。hist 采用国际口径 dif-dea（国内软件常乘以 2）。"""
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return dif, dea, dif - dea


def bollinger(close: Data, n: int = 20, k: float = 2.0):
    """返回 (mid, upper, lower)。"""
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * std, mid - k * std


def true_range(high: Data, low: Data, close: Data) -> Data:
    pc = close.shift(1)
    tr = np.fmax(high - low, np.fmax((high - pc).abs(), (low - pc).abs()))
    return tr


def atr(high: Data, low: Data, close: Data, n: int = 14) -> Data:
    return wilder(true_range(high, low, close), n)


def donchian(high: Data, low: Data, n: int = 20):
    """返回 (upper, lower)：前 n 根（不含当根）的最高/最低价，便于判断当根突破。"""
    return high.rolling(n, min_periods=n).max().shift(1), low.rolling(n, min_periods=n).min().shift(1)


def kdj(high: Data, low: Data, close: Data, n: int = 9, m1: int = 3, m2: int = 3):
    """返回 (k, d, j)，国内常用参数 9,3,3。"""
    lo = low.rolling(n, min_periods=n).min()
    hi = high.rolling(n, min_periods=n).max()
    rsv = (close - lo) / (hi - lo).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def zscore(x: Data, n: int) -> Data:
    mean = x.rolling(n, min_periods=n).mean()
    std = x.rolling(n, min_periods=n).std()
    return (x - mean) / std.replace(0, np.nan)


def realized_vol(close: Data, n: int = 20, periods_per_year: int = 252) -> Data:
    """年化滚动波动率。"""
    return np.log(close).diff().rolling(n, min_periods=n).std() * np.sqrt(periods_per_year)


def obv(close: Data, volume: Data) -> Data:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def crossover(a: Data, b: Data) -> Data:
    """a 上穿 b（当根成立）。"""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: Data, b: Data) -> Data:
    return (a < b) & (a.shift(1) >= b.shift(1))
