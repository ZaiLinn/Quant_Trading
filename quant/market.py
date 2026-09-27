"""市场规则与交易成本。

回测引擎、模拟盘、实盘共用同一份规则，保证三者的成交/费用口径一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

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
    commission_per_share: float = 0.0  # 按股收佣（美股 IBKR 固定费率 0.005 USD/股），非 0 时替代 commission
    max_commission_pct: float = 0.0  # 按股收佣时单笔佣金上限（占成交额比例）
    stamp_tax: float = 0.0           # 印花税（仅卖出）
    buy_tax: float = 0.0             # 买入方税费（港股印花税双边收取）
    transfer_fee: float = 0.0        # 过户费（双边）
    slippage: float = 0.0            # 滑点，按价格比例
    periods_per_year: int = 252      # 日线年化周期数
    symbol_limits: dict = field(default_factory=dict)  # 个别标的涨跌停幅度覆盖，如 {"159915": 0.2}
    symbol_lots: dict = field(default_factory=dict)    # 个别标的每手股数（港股每只不同），如 {"00005": 400}

    def fee(self, value: float, is_sell: bool, qty: float = 0.0) -> float:
        """单笔成交费用。value 为成交金额（正数），qty 为数量（按股收佣时需要）。"""
        if value <= 0:
            return 0.0
        if self.commission_per_share:
            c = max(abs(qty) * self.commission_per_share, self.min_commission)
            if self.max_commission_pct:
                c = min(c, value * self.max_commission_pct)
        else:
            c = max(value * self.commission, self.min_commission)
        return c + value * (self.transfer_fee + (self.stamp_tax if is_sell else self.buy_tax))

    def lots_for(self, symbols: list[str]) -> np.ndarray:
        return np.array([float(self.symbol_lots.get(s, self.lot_size)) for s in symbols])

    def limit_for(self, symbol: str) -> float | None:
        """按代码前缀识别 A 股不同板块涨跌停幅度。"""
        code = symbol.split(".")[0][-6:]
        if symbol in self.symbol_limits or code in self.symbol_limits:
            return self.symbol_limits.get(symbol, self.symbol_limits.get(code))
        if self.price_limit is None or not self.name.startswith("ashare"):
            return self.price_limit
        # 创业板 / 科创板股票，以及科创板 ETF（588 开头）；创业板 ETF（如 159915）需在 symbol_limits 中指定
        if code.startswith(("300", "301", "688", "689", "588")):
            return 0.20
        if code.startswith(("8", "4", "92")):  # 北交所
            return 0.30
        return self.price_limit

    def round_qty(self, qty: np.ndarray | float, lot: np.ndarray | float | None = None) -> np.ndarray | float:
        """按最小交易单位向零取整。lot 可为逐标的数组（见 lots_for），缺省用 lot_size。"""
        lot = self.lot_size if lot is None else lot
        lot_a = np.asarray(lot, dtype=float)
        if not np.any(lot_a > 0):
            return qty
        safe = np.where(lot_a > 0, lot_a, 1.0)
        q = np.asarray(qty, dtype=float)
        with np.errstate(invalid="ignore"):
            out = np.where(lot_a > 0, np.trunc(q / safe) * safe, q)
        return out if (np.ndim(qty) or np.ndim(lot)) else float(out)

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
    # 美股（IBKR 固定费率：0.005 USD/股，最低 1 USD，最高成交额 1%；卖出另有 SEC/FINRA 费约 0.003%）
    "us": MarketRules(
        name="us", lot_size=1, commission_per_share=0.005, min_commission=1.0, max_commission_pct=0.01,
        stamp_tax=0.00003, slippage=0.0005, periods_per_year=252,
    ),
    # 港股（IBKR 固定费率 0.08%、最低 18 HKD；印花税 0.1% 双边；交易所费用合计约 0.01%）
    # 每手股数因股票而异，默认 100，其余在 symbol_lots 中指定
    "hk": MarketRules(
        name="hk", lot_size=100, commission=0.0008, min_commission=18.0, stamp_tax=0.001, buy_tax=0.001,
        transfer_fee=0.0001, slippage=0.001, periods_per_year=247,
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
