"""YAML 配置加载：默认值深度合并 + 命令行 --set a.b=c 覆盖。"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "name": "backtest",
    "market": "generic",
    "market_overrides": {},
    "initial_cash": 1_000_000,
    "benchmark": None,
    "data": {
        "source": "synthetic",
        "symbols": ["AAA", "BBB", "CCC"],
        "universe": None,     # 或指数名 csi300 / csi500 / sse50 / csi1000（A 股，自动获取成分股）
        "workers": None,      # 并行下载线程数
        "start": None,        # 回测开始交易日期；指标预热数据会自动向前多取
        "end": None,
        "freq": "1d",
        "cache_dir": "./data_cache",
        "cache_ttl_hours": 12,
    },
    "strategy": {"name": "sma_cross", "params": {}},
    "risk": {},
    "optimize": {"objective": "sharpe", "grid": {}, "min_trades": 5, "jobs": 1},
    "walkforward": {"train": 504, "test": 126, "anchored": False, "select": "best"},  # select: best | smooth
    "factor": {"specs": [], "horizons": [1, 5, 20], "quantiles": 5},
    "live": {
        "broker": "paper",           # paper | ccxt | ibkr
        "capital": None,             # 策略可用资金上限（真实账户只拿一部分钱跑策略时设置）
        "cron": None,                # 例如 "35 9 * * mon-fri"
        "timezone": None,            # 默认 A 股 Asia/Shanghai、美股 America/New_York、港股 Asia/Hong_Kong，其他 UTC
        "state_dir": "./live_state",
        "dry_run": True,             # ccxt 实盘默认只打印不下单
        "sandbox": False,
        "quote": "USDT",
        "max_daily_loss": 0.05,
        "max_order_value": None,
        "kill_switch_file": "./live_state/STOP",
        "webhook": None,             # 通知 webhook 地址，也可用环境变量 QUANT_WEBHOOK_URL
        "webhook_kind": "generic",   # generic | slack | feishu | dingtalk
    },
    # 自动驾驶（quant autopilot），详见 quant/autopilot.py
    "autopilot": {"candidates": None, "train_bars": 756, "holdout_bars": 126, "objective": None,
                  "min_improvement": 0.2, "min_trades": 3, "cooldown_days": 20,
                  "reopt_cron": "0 18 1 * *", "max_live_drawdown": 0.25, "jobs": 1},
    # IBKR 连接（data.source: ibkr 或 live.broker: ibkr 时使用），详见 quant/ibkr.py
    "ibkr": {"host": "127.0.0.1", "port": 7497, "client_id": 17, "exchange": "SMART", "currency": "USD",
             "account": "", "order_type": "MKT", "fill_timeout": 60},
    "output_dir": "./runs",
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("grid", "params"):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """--set strategy.params.fast=5 --set data.symbols=[a,b]；值按 YAML 语法解析。"""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"覆盖项格式应为 key=value: {item}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    user = {}
    if path:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    return apply_overrides(cfg, overrides)
