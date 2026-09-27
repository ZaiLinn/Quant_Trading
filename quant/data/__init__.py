from __future__ import annotations

import logging

import pandas as pd

from .base import OHLCV, DataSource, freq_to_timedelta, is_intraday, normalize
from .cache import CachedSource
from .local_sources import CsvSource, SyntheticSource
from .panel import Panel

log = logging.getLogger(__name__)

__all__ = ["OHLCV", "DataSource", "CachedSource", "CsvSource", "SyntheticSource", "Panel",
           "make_source", "load_panel", "load_series", "resolve_symbols", "freq_to_timedelta", "is_intraday",
           "normalize"]


def make_source(cfg: dict, cache: bool = True) -> DataSource:
    """根据配置 data 段创建数据源。"""
    kind = cfg.get("source", "synthetic")
    if kind == "akshare":
        from .akshare_source import AkshareSource

        src = AkshareSource(asset_type=cfg.get("asset_type", "auto"), adjust=cfg.get("adjust", "qfq"))
    elif kind == "ccxt":
        from .ccxt_source import CcxtSource

        src = CcxtSource(exchange=cfg.get("exchange", "binance"), options=cfg.get("exchange_options"),
                         drop_incomplete=cfg.get("drop_incomplete", True))
    elif kind == "csv":
        return CsvSource(cfg.get("path", "./data"))
    elif kind == "synthetic":
        return SyntheticSource(seed=cfg.get("seed", 42))
    else:
        raise ValueError(f"未知数据源: {kind}")
    if cache:
        return CachedSource(src, cfg.get("cache_dir", "./data_cache"), cfg.get("cache_ttl_hours", 12))
    return src


def resolve_symbols(cfg: dict) -> list[str]:
    """data.symbols 为列表；或 data.universe 为指数名（csi300/csi500/sse50/...，仅 akshare）。"""
    if cfg.get("universe"):
        from .akshare_source import index_constituents

        syms = index_constituents(cfg["universe"])
        log.warning("使用指数 %s 当前成分股 %d 只：历史回测存在幸存者偏差，结果偏乐观", cfg["universe"], len(syms))
        return syms
    return list(cfg["symbols"])


def load_panel(source: DataSource, symbols: list[str], start=None, end=None,
               freq: str = "1d", min_bars: int = 2, workers: int = 1) -> Panel:
    """workers>1 时多线程并行下载（大股票池首次下载时有用）。"""
    frames: dict[str, pd.DataFrame] = {}

    def one(sym: str):
        try:
            return sym, source.fetch(sym, start, end, freq)
        except Exception as e:  # noqa: BLE001 单个标的失败不影响整体
            log.warning("%s 下载失败: %s", sym, e)
            return sym, None

    if workers > 1 and len(symbols) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(one, symbols))
    else:
        results = [one(s) for s in symbols]
    for sym, df in results:
        if df is None or len(df) < min_bars:
            if df is not None:
                log.warning("%s 数据不足(%d 行)，已跳过", sym, len(df))
            continue
        frames[sym] = df
    if not frames:
        raise RuntimeError("没有可用数据，请检查代码、日期范围或网络")
    return Panel(frames)


def load_series(source: DataSource, symbol: str, start=None, end=None, freq: str = "1d") -> pd.Series:
    return source.fetch(symbol, start, end, freq)["close"]
