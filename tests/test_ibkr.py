"""IBKR 接入测试：用假的 IB 对象，不需要运行 TWS。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ib_async")

from conftest import const_targets, make_panel  # noqa: E402
from quant import get_rules  # noqa: E402
from quant import ibkr  # noqa: E402
from quant.backtest import BacktestEngine  # noqa: E402


class FakeTrade:
    def __init__(self, qty, price, fill_ratio=1.0):
        self.order = SimpleNamespace(orderId=1)
        filled = qty * fill_ratio
        self.orderStatus = SimpleNamespace(filled=filled, avgFillPrice=price, status="Filled")
        self.fills = [SimpleNamespace(commissionReport=SimpleNamespace(commission=1.0))]
        self._done = fill_ratio == 1.0

    def isDone(self):
        return self._done


class FakeIB:
    def __init__(self, fill_ratio=1.0):
        self.orders = []
        self.fill_ratio = fill_ratio
        self.cancelled = 0

    def qualifyContracts(self, *cs):
        for i, c in enumerate(cs):
            c.conId = 1000 + hash(c.symbol) % 1000
        return list(cs)

    def positions(self, account=""):
        c = SimpleNamespace(secType="STK", symbol="SPY", exchange="", primaryExchange="ARCA",
                            currency="USD", conId=1000 + hash("SPY") % 1000)
        other = SimpleNamespace(secType="STK", symbol="TSLA", exchange="", primaryExchange="NASDAQ",
                                currency="USD", conId=5)
        return [SimpleNamespace(contract=c, position=10.0), SimpleNamespace(contract=other, position=3.0)]

    def accountValues(self, account=""):
        return [SimpleNamespace(tag="CashBalance", currency="USD", value="5000"),
                SimpleNamespace(tag="CashBalance", currency="HKD", value="100")]

    def placeOrder(self, contract, order):
        self.orders.append((contract.symbol, order.action, order.totalQuantity, getattr(order, "algoStrategy", "")))
        return FakeTrade(order.totalQuantity, 101.0, self.fill_ratio)

    def cancelOrder(self, order):
        self.cancelled += 1

    def sleep(self, s):
        pass

    def reqContractDetails(self, c):
        return [SimpleNamespace(liquidHours="20260928:0930-20260928:1600;20260929:CLOSED")]


def make_broker(monkeypatch, dry_run=False, fill_ratio=1.0, **cfg):
    fake = FakeIB(fill_ratio)
    monkeypatch.setattr(ibkr, "connect", lambda c, readonly=False: fake)
    b = ibkr.IbkrBroker(get_rules("us"), ["SPY", "QQQ"], {"fill_timeout": 0, **cfg}, dry_run=dry_run)
    return b, fake


def test_parse_symbol():
    assert ibkr.parse_symbol("AAPL") == {"symbol": "AAPL", "exchange": "SMART", "currency": "USD", "primaryExchange": ""}
    hk = ibkr.parse_symbol("00700:SEHK:HKD")
    assert hk["symbol"] == "700" and hk["exchange"] == "SEHK" and hk["currency"] == "HKD"
    assert ibkr.parse_symbol("00005", {"currency": "HKD"})["exchange"] == "SEHK"
    assert ibkr.parse_symbol("AAPL:SMART:USD:NASDAQ")["primaryExchange"] == "NASDAQ"


def test_broker_positions_cash_and_orders(monkeypatch):
    b, fake = make_broker(monkeypatch, order_type="ADAPTIVE")
    pos = b.positions()
    assert pos["SPY"] == 10.0 and pos["TSLA"] == 3.0  # 其他持仓也能看到，但运行器只管理本策略标的
    assert b.cash() == 5000.0
    f = b.execute("QQQ", 5, 100.0)
    assert f["qty"] == 5 and f["price"] == 101.0 and f["fee"] == 1.0
    assert fake.orders[-1] == ("QQQ", "BUY", 5.0, "Adaptive")
    f = b.execute("SPY", -4, 100.0)
    assert f["qty"] == -4 and fake.orders[-1][1] == "SELL"


def test_broker_partial_fill_and_dry_run(monkeypatch):
    b, fake = make_broker(monkeypatch, fill_ratio=0.5)
    f = b.execute("QQQ", 10, 100.0)
    assert f["qty"] == 5 and fake.cancelled == 1
    d, fake2 = make_broker(monkeypatch, dry_run=True)
    assert d.execute("QQQ", 10, 100.0)["dry_run"] and fake2.orders == []


def test_market_open_today(monkeypatch):
    b, _ = make_broker(monkeypatch)
    assert b.market_open_today(pd.Timestamp("2026-09-28 09:40"))
    assert not b.market_open_today(pd.Timestamp("2026-09-29 09:40"))


def test_connect_error_message():
    with pytest.raises(ConnectionError, match="TWS"):
        ibkr.connect({"port": 1, "client_id": 991, "timeout": 2})


def test_us_per_share_commission():
    r = get_rules("us")
    assert r.fee(100 * 50.0, False, 100) == pytest.approx(1.0)          # 最低 1 美元
    assert r.fee(1000 * 50.0, False, 1000) == pytest.approx(5.0)        # 0.005/股
    assert r.fee(1000 * 0.2, False, 1000) == pytest.approx(2.0)         # 上限为成交额 1%
    assert r.fee(1000 * 50.0, True, 1000) > 5.0                         # 卖出含 SEC 费


def test_hk_symbol_lots_in_engine():
    p = make_panel({"00005": [60.0] * 3, "00700": [400.0] * 3})
    rules = get_rules("hk", slippage=0.0, symbol_lots={"00005": 400})
    r = BacktestEngine(rules, 1_000_000).run(p, const_targets(p, {"00005": [0.1, np.nan, np.nan],
                                                                  "00700": [0.1, np.nan, np.nan]}))
    q = r.fills.set_index("symbol")["qty"]
    assert q["00005"] % 400 == 0 and q["00700"] % 100 == 0
    fee = r.fills.set_index("symbol")["fee"]["00700"]
    value = q["00700"] * 400.0
    assert fee == pytest.approx(max(value * 0.0008, 18) + value * (0.0001 + 0.001))  # 买入也收印花税
