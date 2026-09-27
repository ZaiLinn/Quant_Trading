"""券商/交易所接口。

接入新券商只需实现 Broker 的 4 个方法（positions / cash / sellable / execute），
例如 A 股可基于 QMT(xtquant)、掘金等官方量化接口实现；不建议用模拟点击客户端的方案（易碎、难排错）。
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np

from ..execution import exec_price
from ..market import MarketRules

log = logging.getLogger(__name__)


class Broker(ABC):
    rules: MarketRules

    @abstractmethod
    def positions(self) -> dict[str, float]:
        """当前持仓数量（负数为空头）。"""

    @abstractmethod
    def cash(self) -> float:
        """可用现金（计价货币）。"""

    def sellable(self) -> dict[str, float]:
        """当前可卖数量（T+1 市场需扣除当日买入）。默认等于持仓。"""
        return self.positions()

    @abstractmethod
    def execute(self, symbol: str, qty: float, price: float) -> dict | None:
        """市价下单，qty 正买负卖；price 为参考价。返回成交信息或 None（未成交）。"""


class PaperBroker(Broker):
    """模拟盘：按参考价 + 滑点成交，费用与回测一致，状态持久化到 JSON。"""

    def __init__(self, rules: MarketRules, state_file: str | Path, initial_cash: float = 1_000_000):
        self.rules = rules
        self.path = Path(state_file)
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {"cash": float(initial_cash), "positions": {}, "avg_cost": {},
                          "bought_today": {}, "date": "", "fills": []}
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # 原子写，避免中途崩溃损坏状态

    def _roll_day(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state["date"] != today:
            self.state["date"] = today
            self.state["bought_today"] = {}

    def positions(self) -> dict[str, float]:
        return {k: v for k, v in self.state["positions"].items() if v}

    def cash(self) -> float:
        return float(self.state["cash"])

    def sellable(self) -> dict[str, float]:
        self._roll_day()
        pos = self.positions()
        if not self.rules.t_plus_1:
            return pos
        return {s: q - self.state["bought_today"].get(s, 0.0) for s, q in pos.items()}

    def execute(self, symbol: str, qty: float, price: float) -> dict | None:
        self._roll_day()
        if qty == 0 or not np.isfinite(price) or price <= 0:
            return None
        px = exec_price(price, qty, self.rules)
        value = abs(qty) * px
        fee = self.rules.fee(value, qty < 0)
        if qty > 0 and not (self.rules.allow_short or self.rules.max_leverage > 1) \
                and value + fee > self.state["cash"] + 1e-6:
            log.warning("模拟盘资金不足，放弃买入 %s %.4f", symbol, qty)
            return None
        old = self.state["positions"].get(symbol, 0.0)
        new = old + qty
        avg = self.state["avg_cost"].get(symbol, 0.0)
        if old == 0 or np.sign(old) == np.sign(qty):
            avg = (avg * abs(old) + px * abs(qty)) / abs(new)
        elif abs(qty) > abs(old):
            avg = px
        self.state["positions"][symbol] = 0.0 if abs(new) < 1e-12 else new
        self.state["avg_cost"][symbol] = avg if new else 0.0
        self.state["cash"] -= qty * px + fee
        if qty > 0:
            self.state["bought_today"][symbol] = self.state["bought_today"].get(symbol, 0.0) + qty
        fill = {"time": datetime.now().isoformat(timespec="seconds"), "symbol": symbol, "qty": qty,
                "price": px, "fee": fee}
        self.state["fills"] = (self.state["fills"] + [fill])[-2000:]
        self._save()
        return fill


class CcxtBroker(Broker):
    """加密货币现货实盘（ccxt）。API Key 只从环境变量读取，绝不写进配置文件：
    QUANT_<EXCHANGE>_API_KEY / QUANT_<EXCHANGE>_SECRET / QUANT_<EXCHANGE>_PASSWORD(部分交易所需要)
    dry_run=True 时只打印订单不下单。
    """

    def __init__(self, exchange: str, rules: MarketRules, symbols: list[str], quote: str = "USDT",
                 sandbox: bool = False, dry_run: bool = True, options: dict | None = None):
        import ccxt

        self.rules = rules
        self.symbols = symbols
        self.quote = quote
        self.dry_run = dry_run
        env = f"QUANT_{exchange.upper()}_"
        creds = {k: os.environ.get(env + v) for k, v in
                 (("apiKey", "API_KEY"), ("secret", "SECRET"), ("password", "PASSWORD"))}
        creds = {k: v for k, v in creds.items() if v}
        if not dry_run and "apiKey" not in creds:
            raise RuntimeError(f"实盘需要设置环境变量 {env}API_KEY / {env}SECRET")
        self.ex = getattr(ccxt, exchange)({"enableRateLimit": True, **creds, **(options or {})})
        if sandbox:
            self.ex.set_sandbox_mode(True)
        self.ex.load_markets()
        self._authed = "apiKey" in creds

    def _balance(self) -> dict:
        if not self._authed:
            return {"total": {}, "free": {}}
        return self.ex.fetch_balance()

    def positions(self) -> dict[str, float]:
        total = self._balance()["total"]
        return {s: float(total.get(self.ex.market(s)["base"]) or 0.0) for s in self.symbols}

    def cash(self) -> float:
        return float(self._balance()["free"].get(self.quote) or 0.0)

    def execute(self, symbol: str, qty: float, price: float) -> dict | None:
        market = self.ex.market(symbol)
        amount = float(self.ex.amount_to_precision(symbol, abs(qty)))
        lim = market.get("limits", {})
        min_amt = (lim.get("amount") or {}).get("min") or 0
        min_cost = (lim.get("cost") or {}).get("min") or 0
        if amount <= 0 or amount < min_amt or amount * price < min_cost:
            log.info("%s 下单量 %.8f 低于交易所最小限制，跳过", symbol, amount)
            return None
        side = "buy" if qty > 0 else "sell"
        if self.dry_run:
            log.info("[DRY-RUN] %s %s %.8f @~%.4f", side, symbol, amount, price)
            return {"symbol": symbol, "qty": np.sign(qty) * amount, "price": price, "fee": 0.0, "dry_run": True}
        order = self.ex.create_order(symbol, "market", side, amount)
        filled = float(order.get("filled") or amount)
        avg = float(order.get("average") or order.get("price") or price)
        fee = float((order.get("fee") or {}).get("cost") or 0.0)
        log.info("成交 %s %s %.8f @ %.4f (id=%s)", side, symbol, filled, avg, order.get("id"))
        return {"symbol": symbol, "qty": np.sign(qty) * filled, "price": avg, "fee": fee, "id": order.get("id")}
