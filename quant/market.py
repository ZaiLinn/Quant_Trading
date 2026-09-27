"""市场规则与交易成本。

回测引擎、模拟盘、实盘共用同一份规则，保证三者的成交/费用口径一致。
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np


@dataclass(frozen=True)
class MarketRules:
    name: str = "generic"
    lot_size: float = 0.0            # 最小交易单位；0 表示可小数成交（加密货币）
    t_plus_1: bool = False           # 当日买入次日才能卖出
    price_limit: float | None = None  # 涨跌停幅度（相对昨收），None 表示无限制
    allow_short: bool = False
    max_leverage: float = 1.0        # 总敞口 / 权益 上限
    commission: float = 0.0          # 佣金率（双边）
    min_commission: float = 0.0      # 单笔最低佣金
    stamp_tax: float = 0.0           # 印花税（仅卖出）
    transfer_fee: float = 0.0        # 过户费（双边）
    slippage: float = 0.0            # 滑点，按价格比例
    periods_per_year: int = 252      # 日线年化周期数

    def fee(self, value: float, is_sell: bool) -> float:
        """单笔成交费用。value 为成交金额（正数）。"""
        if value <= 0:
            return 0.0
        f = max(value * self.commission, self.min_commission)
        f += value * self.transfer_fee
        if is_sell:
            f += value * self.stamp_tax
        return f

    def limit_for(self, symbol: str) -> float | None:
        """按代码前缀识别 A 股不同板块涨跌停幅度。"""
        if self.price_limit is None or not self.name.startswith("ashare"):
            return self.price_limit
        code = symbol.split(".")[0][-6:]
        if code.startswith(("300", "301", "688", "689")):  # 创业板 / 科创板
            return 0.20
        if code.startswith(("8", "4", "92")):  # 北交所
            return 0.30
        return self.price_limit

    def round_qty(self, qty: np.ndarray | float) -> np.ndarray | float:
        """按最小交易单位向零取整。"""
        if self.lot_size and self.lot_size > 0:
            return np.trunc(np.asarray(qty) / self.lot_size) * self.lot_size
        return qty

    def with_overrides(self, **kw) -> "MarketRules":
        valid = {f.name for f in fields(self)}
        unknown = set(kw) - valid
        if unknown:
            raise ValueError(f"未知的市场规则字段: {sorted(unknown)}")
        return replace(self, **kw)


# 费率为常见默认值，按自己券商/交易所实际费率在配置中覆盖。
# 印花税：2023-08-28 起减半为 0.05%，早于该日期的回测成本会略被低估。
PRESETS: dict[str, MarketRules] = {
    "generic": MarketRules(),
    "ashare": MarketRules(
        name="ashare", lot_size=100, t_plus_1=True, price_limit=0.10,
        commission=0.00025, min_commission=5.0, stamp_tax=0.0005,
        transfer_fee=0.00001, slippage=0.0005, periods_per_year=252,
    ),
    # 场内 ETF：免印花税、过户费；跨境/黄金/债券 ETF 实际为 T+0，这里统一按 T+1 保守处理
    "ashare_etf": MarketRules(
        name="ashare_etf", lot_size=100, t_plus_1=True, price_limit=0.10,
        commission=0.0002, min_commission=5.0, slippage=0.0003, periods_per_year=252,
    ),
    "crypto": MarketRules(
        name="crypto", commission=0.001, slippage=0.0005, periods_per_year=365,
    ),
    "crypto_perp": MarketRules(
        name="crypto_perp", allow_short=True, max_leverage=3.0,
        commission=0.0005, slippage=0.0005, periods_per_year=365,
    ),
}


def get_rules(name: str = "generic", **overrides) -> MarketRules:
    if name not in PRESETS:
        raise ValueError(f"未知市场 {name!r}，可选: {sorted(PRESETS)}")
    rules = PRESETS[name]
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return rules.with_overrides(**overrides) if overrides else rules
