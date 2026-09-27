"""A 股数据（akshare）。

优先东方财富接口（支持复权、日期区间）；东财限流/断连时自动切换新浪接口：
股票日线支持复权；ETF 为不复权价（分红较少的宽基 ETF 影响有限）；指数无复权问题。
"""
from __future__ import annotations

import logging
import threading

import pandas as pd

from .base import DataSource, normalize, retry, to_timestamp

log = logging.getLogger(__name__)

# 新浪接口内部用 V8（mini_racer）解码，多线程并发调用会让进程直接崩溃，必须串行
_SINA_LOCK = threading.Lock()

_CN_COLS = {"日期": "date", "时间": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"}
_PERIOD = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
_MINUTE = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60", "1h": "60"}
_ETF_PREFIX = ("51", "52", "56", "58", "15", "16", "18")


def guess_asset_type(symbol: str) -> str:
    return "etf" if symbol.startswith(_ETF_PREFIX) else "stock"


def exchange_prefix(symbol: str) -> str:
    """600519 -> sh600519；000001 -> sz000001；8/4/92 开头 -> bj。"""
    if symbol.startswith(("6", "5", "9")) and not symbol.startswith("92"):
        return "sh" + symbol
    if symbol.startswith(("8", "4", "92")):
        return "bj" + symbol
    return "sz" + symbol


class AkshareSource(DataSource):
    name = "akshare"

    def __init__(self, asset_type: str = "auto", adjust: str = "qfq", timeout: float = 15):
        """
        asset_type: auto | stock | etf | index
        adjust: qfq(前复权) | hfq(后复权) | ""(不复权)。
          前复权保证价格连续、手数/金额量级接近真实；涨跌停按相对昨收比例判断，不受乘法复权影响。
        """
        import akshare  # noqa: F401  延迟导入，未安装时其他模块仍可用

        self.asset_type = asset_type
        self.adjust = adjust
        self.timeout = timeout
        self._em_down = False  # 东财接口失败后本实例直接走新浪，避免每个标的都等待重试

    def cache_key(self, symbol: str, freq: str) -> str:
        return f"{self.name}_{self._type(symbol)}_{symbol}_{freq}_{self.adjust or 'raw'}"

    def _type(self, symbol: str) -> str:
        return guess_asset_type(symbol) if self.asset_type == "auto" else self.asset_type

    def fetch(self, symbol: str, start=None, end=None, freq: str = "1d") -> pd.DataFrame:
        import akshare as ak

        start_ts = to_timestamp(start) or pd.Timestamp("1990-01-01")
        end_ts = to_timestamp(end) or pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        s, e = start_ts.strftime("%Y%m%d"), end_ts.strftime("%Y%m%d")
        kind = self._type(symbol)

        if freq in _MINUTE:
            if kind == "index":
                raise ValueError("指数分钟线暂不支持")
            fn = ak.stock_zh_a_hist_min_em if kind == "stock" else ak.fund_etf_hist_min_em
            sd, ed = start_ts.strftime("%Y-%m-%d 09:00:00"), end_ts.strftime("%Y-%m-%d 15:30:00")
            adjust = "" if _MINUTE[freq] == "1" else self.adjust  # 1 分钟线不支持复权
            raw = retry(lambda: fn(symbol=symbol, start_date=sd, end_date=ed,
                                   period=_MINUTE[freq], adjust=adjust), what=f"akshare {symbol}")
        elif freq in _PERIOD:
            period = _PERIOD[freq]
            if kind == "stock":
                primary = lambda: ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=s,  # noqa: E731
                                                     end_date=e, adjust=self.adjust, timeout=self.timeout)
                backup = lambda: ak.stock_zh_a_daily(symbol=exchange_prefix(symbol), start_date=s,  # noqa: E731
                                                     end_date=e, adjust=self.adjust)
            elif kind == "etf":
                primary = lambda: ak.fund_etf_hist_em(symbol=symbol, period=period, start_date=s,  # noqa: E731
                                                      end_date=e, adjust=self.adjust)
                backup = lambda: ak.fund_etf_hist_sina(symbol=exchange_prefix(symbol))  # noqa: E731
            elif kind == "index":
                primary = lambda: ak.index_zh_a_hist(symbol=symbol, period=period,  # noqa: E731
                                                     start_date=s, end_date=e)
                backup = lambda: ak.stock_zh_index_daily(symbol=index_prefix(symbol))  # noqa: E731
            else:
                raise ValueError(f"未知 asset_type: {kind}")
            raw = self._with_fallback(primary, backup if period == "daily" else None, symbol, kind)
        else:
            raise ValueError(f"akshare 不支持周期 {freq}")

        df = raw.rename(columns=_CN_COLS)
        df = df.set_index(pd.to_datetime(df["date"]))
        return normalize(df).loc[start_ts:end_ts]

    def _with_fallback(self, primary, backup, symbol: str, kind: str) -> pd.DataFrame:
        if not self._em_down or backup is None:
            try:
                return retry(primary, tries=2, what=f"东财 {kind} {symbol}")
            except Exception as e:  # noqa: BLE001
                if backup is None:
                    raise
                self._em_down = True
                log.warning("东财接口不可用(%s)，改用新浪接口", type(e).__name__)
        if kind == "etf" and self.adjust:
            log.warning("%s 使用新浪不复权数据（分红除息日会出现价格缺口）", symbol)
        with _SINA_LOCK:
            return retry(backup, what=f"新浪 {kind} {symbol}")


def index_prefix(symbol: str) -> str:
    """指数代码：399xxx 为深证，其余（000xxx 等）按上证处理。"""
    return ("sz" if symbol.startswith("399") else "sh") + symbol


INDEX_UNIVERSE = {"sse50": "000016", "csi300": "000300", "csi500": "000905", "csi1000": "000852",
                  "chinext": "399006", "star50": "000688"}


def index_constituents(index: str) -> list[str]:
    """指数当前成分股代码。注意：用"当前"成分股回测历史存在幸存者偏差（结果偏乐观）。"""
    import akshare as ak

    code = INDEX_UNIVERSE.get(index, index)
    try:
        df = retry(lambda: ak.index_stock_cons_csindex(symbol=code), what=f"中证成分 {code}")
        return sorted(df["成分券代码"].astype(str).str.zfill(6).unique())
    except Exception:  # noqa: BLE001
        df = retry(lambda: ak.index_stock_cons_sina(symbol=code), what=f"新浪成分 {code}")
        return sorted(df["code"].astype(str).str.zfill(6).unique())


def trade_calendar() -> pd.DatetimeIndex:
    """A 股交易日历（含当年已公布的未来交易日）。"""
    import akshare as ak

    df = retry(ak.tool_trade_date_hist_sina, what="交易日历")
    return pd.DatetimeIndex(pd.to_datetime(df["trade_date"]))
