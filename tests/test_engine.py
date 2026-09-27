import numpy as np
import pandas as pd
import pytest

from conftest import const_targets, make_panel
from quant import get_rules, get_strategy, run_backtest
from quant.backtest import BacktestEngine

NAN = np.nan


def test_executes_next_bar_open_without_costs():
    p = make_panel({"A": [10, 11, 12, 13]}, opens={"A": [10, 10.5, 11.5, 12.5]})
    t = const_targets(p, {"A": [1.0, NAN, NAN, NAN]})
    r = BacktestEngine(get_rules("generic"), 1000).run(p, t)
    assert r.fills.iloc[0]["date"] == p.index[1]
    assert r.fills.iloc[0]["price"] == pytest.approx(10.5)
    assert r.positions["A"].iloc[1] == pytest.approx(1000 / 10.5)
    assert r.equity.iloc[0] == 1000
    assert r.equity.iloc[-1] == pytest.approx(1000 / 10.5 * 13)


def test_equity_reconciles_with_trade_pnl(panel):
    rules = get_rules("ashare")
    r = run_backtest(get_strategy("sma_cross", fast=5, slow=20), panel, rules, 1_000_000,
                     risk={"stop_loss": 0.08, "take_profit": 0.3})
    assert len(r.trades) > 10
    # 每个回合（含期末未平仓的浮动盈亏）加总应恰好等于账户盈亏
    assert r.trades["pnl"].sum() == pytest.approx(r.equity.iloc[-1] - r.initial_cash, rel=1e-9, abs=1e-6)
    assert (r.cash >= -1e-6).all()
    assert set(r.trades["exit_reason"]) <= {"signal", "stop", "take_profit", "open"}


def test_fees_match_rules():
    p = make_panel({"A": [10, 10, 10, 10]})
    rules = get_rules("ashare")
    t = const_targets(p, {"A": [0.5, 0.5, 0.0, 0.0]})
    r = BacktestEngine(rules, 100_000).run(p, t)
    buy, sell = r.fills.iloc[0], r.fills.iloc[1]
    assert buy["qty"] % 100 == 0 and sell["qty"] == -buy["qty"]
    assert buy["fee"] == pytest.approx(rules.fee(buy["value"], False))
    assert sell["fee"] == pytest.approx(rules.fee(sell["value"], True))
    assert sell["fee"] > buy["fee"]  # 卖出有印花税
    assert r.equity.iloc[-1] == pytest.approx(100_000 - r.fills["fee"].sum()
                                              - (buy["price"] - sell["price"]) * buy["qty"])


def test_ashare_lot_size_and_min_commission():
    p = make_panel({"A": [33.3] * 3})
    rules = get_rules("ashare", slippage=0.0)
    r = BacktestEngine(rules, 10_000).run(p, const_targets(p, {"A": [1.0, NAN, NAN]}))
    qty = r.fills.iloc[0]["qty"]
    assert qty == 300  # 10000/33.3=300.3 -> 300 股
    assert r.fills.iloc[0]["fee"] == pytest.approx(5.0 + 300 * 33.3 * 0.00001)


def test_t_plus_1_blocks_same_day_stop():
    # 开盘买入后当天大跌触发止损，但 T+1 不能卖，次日才能卖
    p = make_panel({"A": [10, 10, 8, 8]}, opens={"A": [10, 10, 10, 8]})
    rules = get_rules("ashare", slippage=0.0, price_limit=None)
    eng = BacktestEngine(rules, 100_000, stop_loss=0.05)
    r = eng.run(p, const_targets(p, {"A": [NAN, 1.0, NAN, NAN]}))
    stop = r.fills[r.fills["reason"] == "stop"]
    assert len(stop) == 1 and stop.iloc[0]["date"] == p.index[3]
    # 无 T+1 的市场当天即可止损，按止损价成交
    r2 = BacktestEngine(get_rules("crypto", slippage=0.0), 100_000, stop_loss=0.05).run(
        p, const_targets(p, {"A": [NAN, 1.0, NAN, NAN]}))
    s2 = r2.fills[r2.fills["reason"] == "stop"].iloc[0]
    assert s2["date"] == p.index[2] and s2["price"] == pytest.approx(9.5)


def test_limit_up_blocks_buy_then_retries():
    # 第 2 根开盘一字涨停（+10%）买不进，第 3 根继续尝试成交
    p = make_panel({"A": [10, 11, 11.5, 11.5]}, opens={"A": [10, 11, 11.2, 11.5]})
    rules = get_rules("ashare", slippage=0.0)
    r = BacktestEngine(rules, 100_000).run(p, const_targets(p, {"A": [1.0, NAN, NAN, NAN]}))
    assert r.fills.iloc[0]["date"] == p.index[2]


def test_suspension_defers_order():
    p = make_panel({"A": [10, NAN, 10, 10], "B": [5, 5, 5, 5]})
    rules = get_rules("generic")
    r = BacktestEngine(rules, 1000).run(p, const_targets(p, {"A": [0.5, NAN, NAN, NAN], "B": [0.5, NAN, NAN, NAN]}))
    dates = r.fills.groupby("symbol")["date"].first()
    assert dates["B"] == p.index[1] and dates["A"] == p.index[2]


def test_rebalances_only_on_target_change():
    p = make_panel({"A": list(np.linspace(10, 20, 30))})
    r = BacktestEngine(get_rules("generic"), 1000).run(p, const_targets(p, {"A": [0.5] * 30}))
    assert len(r.fills) == 1
    r2 = BacktestEngine(get_rules("generic"), 1000, drift_threshold=0.05).run(p, const_targets(p, {"A": [0.5] * 30}))
    assert len(r2.fills) > 1


def test_short_and_leverage_limits():
    p = make_panel({"A": [10, 10, 9, 8]})
    t = const_targets(p, {"A": [-1.0, NAN, NAN, NAN]})
    long_only = BacktestEngine(get_rules("crypto"), 1000).run(p, t)
    assert len(long_only.fills) == 0
    perp = BacktestEngine(get_rules("crypto_perp", slippage=0.0, commission=0.0), 1000).run(p, t)
    assert perp.positions["A"].iloc[1] == pytest.approx(-100)
    assert perp.equity.iloc[-1] == pytest.approx(1200)
    lev = BacktestEngine(get_rules("crypto_perp", slippage=0.0, commission=0.0), 1000).run(
        p, const_targets(p, {"A": [5.0, NAN, NAN, NAN]}))
    assert lev.positions["A"].iloc[1] == pytest.approx(300)  # max_leverage=3


def test_cash_never_negative_with_many_symbols(panel):
    r = run_backtest(get_strategy("momentum_rotation", lookback=10, top_k=2, rebalance="D"),
                     panel, get_rules("ashare"), 50_000)
    assert (r.cash >= -1e-6).all()
    assert (r.positions % 100 == 0).all().all()


def test_trade_start_excludes_warmup(panel):
    start = pd.Timestamp("2020-01-01")
    r = run_backtest(get_strategy("sma_cross", fast=10, slow=50), panel, get_rules("generic"), start=start)
    assert r.equity.index[0] >= start
    assert r.positions.iloc[1:].abs().sum().sum() > 0


def test_volume_participation_splits_large_orders():
    p = make_panel({"A": [10.0] * 6}, volume=1000)
    rules = get_rules("generic")
    r = BacktestEngine(rules, 100_000, max_volume_pct=0.1).run(p, const_targets(p, {"A": [1.0] + [NAN] * 5}))
    # 每根最多成交上一根成交量的 10% = 100 股，目标 10000 股需要分多根执行
    assert (r.fills["qty"] <= 100 + 1e-9).all()
    assert len(r.fills) == 5


def test_start_carries_existing_target(panel):
    # 月度调仓策略从月中开始回测：应立即按最近一次目标建仓，而不是空仓等到下个月
    r = run_backtest(get_strategy("risk_parity", window=20, rebalance="M"), panel, get_rules("generic"),
                     start="2020-06-15")
    assert r.fills["date"].min() <= r.equity.index[2]
