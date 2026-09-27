"""实盘事前风控：熔断开关、单日亏损上限、单笔金额上限。"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..market import MarketRules

log = logging.getLogger(__name__)


class RiskGuard:
    def __init__(self, rules: MarketRules, max_daily_loss: float | None = 0.05,
                 max_order_value: float | None = None, kill_switch_file: str | None = None):
        self.rules = rules
        self.max_daily_loss = max_daily_loss
        self.max_order_value = max_order_value
        self.kill_switch = Path(kill_switch_file) if kill_switch_file else None

    def halt_reason(self, equity: float, day_start_equity: float | None) -> str | None:
        """返回暂停开仓的原因；None 表示正常。暂停期间只允许减仓。"""
        if self.kill_switch and self.kill_switch.exists():
            return f"检测到熔断文件 {self.kill_switch}"
        if self.max_daily_loss and day_start_equity:
            loss = 1 - equity / day_start_equity
            if loss >= self.max_daily_loss:
                return f"当日亏损 {loss:.2%} 超过上限 {self.max_daily_loss:.2%}"
        return None

    def filter(self, delta: np.ndarray, pos: np.ndarray, price: np.ndarray,
               halted: bool, lots: np.ndarray | None = None) -> np.ndarray:
        out = delta.copy()
        if halted:
            # 只允许减仓：反向单最多平到 0，同向（加仓）单取消
            reducing = np.sign(out) == -np.sign(pos)
            out = np.where(reducing, np.sign(out) * np.minimum(np.abs(out), np.abs(pos)), 0.0)
        if self.max_order_value:
            px = np.where(np.isfinite(price) & (price > 0), price, np.inf)
            cap = self.rules.round_qty(self.max_order_value / px, lots)
            too_big = np.abs(out) > cap
            if too_big.any():
                log.warning("单笔金额超过 %.0f，已截断", self.max_order_value)
            out = np.where(too_big, np.sign(out) * cap, out)
        return out
