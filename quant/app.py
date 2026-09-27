"""把配置装配成可运行对象（CLI、脚本、Notebook 共用）。"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data import Panel, freq_to_timedelta, load_panel, make_source, resolve_symbols
from .market import MarketRules, get_rules
from .strategy import Strategy, get_strategy

log = logging.getLogger(__name__)


@dataclass
class Setup:
    cfg: dict
    strategy: Strategy
    rules: MarketRules
    panel: Panel
    benchmark: pd.Series | None
    trade_start: pd.Timestamp | None


def make_rules(cfg: dict) -> MarketRules:
    return get_rules(cfg["market"], **(cfg.get("market_overrides") or {}))


def make_strategy(cfg: dict, **params) -> Strategy:
    s = cfg["strategy"]
    return get_strategy(s["name"], **{**(s.get("params") or {}), **params})


def warmup_start(start, bars: int, freq: str) -> pd.Timestamp | None:
    """交易开始日往前推足够的日历时间以覆盖指标预热（日线按 1 根≈1.6 个日历日估算，含节假日）。"""
    if start is None:
        return None
    step = freq_to_timedelta(freq)
    factor = 1.6 if step >= pd.Timedelta(days=1) else 1.2
    return pd.Timestamp(start) - step * int(bars * factor + 5)


def load_benchmark(cfg: dict, start, end) -> pd.Series | None:
    b = cfg.get("benchmark")
    if not b:
        return None
    b = {"symbol": b} if isinstance(b, str) else dict(b)
    src_cfg = {**cfg["data"], **{k: v for k, v in b.items() if k != "symbol"}}
    try:
        src = make_source(src_cfg)
        return src.fetch(b["symbol"], start, end, cfg["data"].get("freq", "1d"))["close"]
    except Exception as e:  # noqa: BLE001 基准失败不影响回测
        log.warning("基准 %s 加载失败: %s", b["symbol"], e)
        return None


def prepare(cfg: dict, extra_warmup: int = 0) -> Setup:
    d = cfg["data"]
    strategy = make_strategy(cfg)
    rules = make_rules(cfg)
    source = make_source(d)
    fetch_start = warmup_start(d.get("start"), strategy.warmup() + extra_warmup, d.get("freq", "1d"))
    panel = load_panel(source, resolve_symbols(d), fetch_start, d.get("end"), d.get("freq", "1d"),
                       workers=int(d.get("workers") or (8 if d.get("universe") else 1)))
    trade_start = pd.Timestamp(d["start"]) if d.get("start") else None
    bench = load_benchmark(cfg, fetch_start, d.get("end"))
    return Setup(cfg, strategy, rules, panel, bench, trade_start)


def run_dir(cfg: dict, kind: str) -> Path:
    path = Path(cfg.get("output_dir", "./runs")) / f"{cfg['name']}_{kind}_{datetime.now():%Y%m%d_%H%M%S}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    def default(x):
        if isinstance(x, (pd.Timestamp, datetime)):
            return str(x)
        if hasattr(x, "item"):
            return x.item()
        return str(x)

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default), encoding="utf-8")
