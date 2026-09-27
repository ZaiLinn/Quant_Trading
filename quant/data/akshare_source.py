"""A 股数据（akshare）。

优先东方财富接口（支持复权、日期区间）；东财限流/断连时自动切换备用源：
- 股票日线：新浪（支持复权）；
- ETF 日线：腾讯（支持复权，能正确处理份额拆分）→ 新浪（不复权，最后兜底）；
- 指数日线：新浪（指数无复权问题）。
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np
import pandas as pd

from .base import DataSource, normalize, retry, to_timestamp

log = logging.getLogger(__name__)

# 新浪接口内部用 V8（mini_racer）解码，多线程并发调用会让进程直接崩溃，必须串行
_SINA_LOCK = threading.Lock()
os.environ.setdefault("TQDM_DISABLE", "1")  # akshare 部分接口会打印进度条

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
        asset_type: auto | stock | etf | index | us（美股/美股 ETF，如 AAPL）| hk（港股，如 00700）
        adjust: qfq(前复权) | hfq(后复权) | ""(不复权)。
          前复权保证价格连续、手数/金额量级接近真实；涨跌停按相对昨收比例判断，不受乘法复权影响。
        """
        import akshare  # noqa: F401  延迟导入，未安装时其他模块仍可用

        self.asset_type = asset_type
        self.adjust = adjust
        self.timeout = timeout
        self._em_down = False  # 东财接口失败后本实例直接走新浪，避免每个标的都等待重试

    def cache_key(self, symbol: str, freq: str) -> str:
        return f"{self.name}_v4_{self._type(symbol)}_{symbol}_{freq}_{self.adjust or 'raw'}"

    def _type(self, symbol: str) -> str:
        return guess_asset_type(symbol) if self.asset_type == "auto" else self.asset_type

    def fetch(self, symbol: str, start=None, end=None, freq: str = "1d") -> pd.DataFrame:
        import akshare as ak

        start_ts = to_timestamp(start) or pd.Timestamp("1990-01-01")
        end_ts = to_timestamp(end) or pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        s, e = start_ts.strftime("%Y%m%d"), end_ts.strftime("%Y%m%d")
        kind = self._type(symbol)

        if kind in ("us", "hk"):
            if freq != "1d":
                raise ValueError("美股/港股免费数据仅支持日线；分钟线请用 source: ibkr")
            with _SINA_LOCK:  # 新浪接口，同样不能并发
                if kind == "us":
                    raw = us_adjusted(symbol.upper(), self.adjust)
                else:
                    raw = retry(lambda: ak.stock_hk_daily(symbol=symbol.zfill(5), adjust=self.adjust),
                                what=f"新浪 hk {symbol}")
        elif freq in _MINUTE:
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
                backup = self._etf_backup(symbol, s, e)
            elif kind == "index":
                primary = lambda: ak.index_zh_a_hist(symbol=symbol, period=period,  # noqa: E731
                                                     start_date=s, end_date=e)
                backup = lambda: ak.stock_zh_index_daily(symbol=index_prefix(symbol))  # noqa: E731
            else:
                raise ValueError(f"未知 asset_type: {kind}")
            raw = self._with_fallback(primary, backup if period == "daily" else None, symbol, kind)
        else:
            raise ValueError(f"akshare 不支持周期 {freq}")

        em = "成交量" in raw.columns
        df = raw.rename(columns=_CN_COLS)
        if em:
            # 东财成交量单位是"手"（100 股），新浪是"股"：统一为股，否则换源后量价因子、容量估算全错
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100
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
                log.warning("东财接口不可用(%s)，改用备用接口（腾讯/新浪）", type(e).__name__)
        with _SINA_LOCK:
            return retry(backup, what=f"备用源 {kind} {symbol}")

    def _etf_backup(self, symbol: str, s: str, e: str):
        import akshare as ak

        def fetch():
            if self.adjust:
                try:
                    return retry(lambda: ak.stock_zh_a_hist_tx(symbol=exchange_prefix(symbol), start_date=s,
                                                               end_date=e, adjust=self.adjust,
                                                               timeout=self.timeout),
                                 tries=3, what=f"腾讯 etf {symbol}")
                except Exception as err:  # noqa: BLE001
                    log.warning("腾讯接口失败(%s)，%s 改用新浪不复权数据（拆分/分红日会出现价格缺口）",
                                type(err).__name__, symbol)
            return ak.fund_etf_hist_sina(symbol=exchange_prefix(symbol))
        return fetch


def us_adjusted(symbol: str, adjust: str) -> pd.DataFrame:
    """美股乘法复权。

    新浪美股的"前复权"对拆分用乘法、对分红用累计减法（qfq = raw × factor + adjust），
    长历史下早期价格被减去大量分红（SPY 2011 年从 124 变成 48），涨跌幅被放大、年化收益虚高。
    这里用原始价 + 复权因子表重建：拆分按 factor，分红按 1 − 分红 / 除息前收盘 连乘。
    """
    import akshare as ak

    raw = retry(lambda: ak.stock_us_daily(symbol=symbol, adjust=""), what=f"新浪 us {symbol}")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    if not adjust:
        return raw
    tbl = retry(lambda: ak.stock_us_daily(symbol=symbol, adjust="qfq-factor"), what=f"新浪 us 因子 {symbol}")
    tbl["date"] = pd.to_datetime(tbl["date"])
    tbl = tbl.sort_values("date").astype({"qfq_factor": float, "adjust": float}).reset_index(drop=True)
    # 每个交易日所属的因子区间（区间起始日 <= 当日）
    pos = tbl["date"].searchsorted(raw["date"], side="right") - 1
    pos = pos.clip(0, len(tbl) - 1)
    split = tbl["qfq_factor"].to_numpy()[pos]
    add = tbl["adjust"].to_numpy()[pos]
    px_cols = ["open", "high", "low", "close"]
    base = raw[px_cols].astype(float).mul(split, axis=0)  # 仅拆分调整后的价格
    close = base["close"].to_numpy()
    mult = np.ones(len(raw))
    # 分红事件：adjust 在某日跳升（例如 -0.27 -> 0），跳升幅度即除息金额（已按拆分调整）
    jumps = np.nonzero(np.diff(add) > 1e-9)[0]
    for j in jumps:
        ex = j + 1
        div = add[ex] - add[j]
        if close[j] > div > 0:
            mult[:ex] *= 1 - div / close[j]
    out = raw.copy()
    out[px_cols] = base.mul(mult, axis=0)
    if adjust == "hfq":
        out[px_cols] = out[px_cols] / (mult[0] * split[0]) if mult[0] * split[0] else out[px_cols]
    return out


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


def sina_quotes(symbols: list[str], asset_type: str = "auto") -> pd.DataFrame:
    """新浪实时行情（A 股股票/ETF/指数），不复权原始价格。

    返回 index=symbol，列 open/prev_close/price/high/low/volume/amount/time（time 为行情时间戳）。
    停牌或无数据的标的不在结果中。
    """
    import requests

    def code(sym: str) -> str:
        return index_prefix(sym) if asset_type == "index" else exchange_prefix(sym)

    rows = {}
    for i in range(0, len(symbols), 200):
        batch = symbols[i:i + 200]
        url = "https://hq.sinajs.cn/list=" + ",".join(code(s) for s in batch)
        r = retry(lambda: requests.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=10),
                  what="新浪实时行情")
        r.encoding = "gbk"
        by_code = {}
        for line in r.text.splitlines():
            if "=\"" not in line:
                continue
            key = line.split("=")[0].rsplit("_", 1)[-1]
            by_code[key] = line.split("\"")[1].split(",")
        for sym in batch:
            f = by_code.get(code(sym))
            if not f or len(f) < 32:
                continue
            try:
                price = float(f[3])
                if price <= 0:
                    continue
                rows[sym] = {"open": float(f[1]), "prev_close": float(f[2]), "price": price,
                             "high": float(f[4]), "low": float(f[5]), "volume": float(f[8]),
                             "amount": float(f[9]), "time": pd.Timestamp(f"{f[30]} {f[31]}")}
            except (ValueError, IndexError):
                continue
    return pd.DataFrame.from_dict(rows, orient="index")


def trade_calendar() -> pd.DatetimeIndex:
    """A 股交易日历（含当年已公布的未来交易日）。"""
    import akshare as ak

    df = retry(ak.tool_trade_date_hist_sina, what="交易日历")
    return pd.DatetimeIndex(pd.to_datetime(df["trade_date"]))
