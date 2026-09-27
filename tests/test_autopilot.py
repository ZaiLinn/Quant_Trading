import json

import pandas as pd

from quant.autopilot import Autopilot
from quant.config import load_config
from quant.data import SyntheticSource


class NowSource(SyntheticSource):
    def fetch(self, symbol, start=None, end=None, freq="1d"):
        end = pd.Timestamp.now().normalize()
        return super().fetch(symbol, end - pd.Timedelta(days=1500), end, freq)


def make_cfg(tmp_path, **extra):
    sets = [f"live.state_dir={tmp_path}", f"live.kill_switch_file={tmp_path / 'STOP'}",
            "data.symbols=[AAA, BBB, CCC]", "market=crypto", "strategy.name=sma_cross",
            "strategy.params={fast: 10, slow: 50}",
            "autopilot.candidates=[{name: sma_cross, grid: {fast: [5, 20], slow: [40, 80]}}, "
            "{name: momentum_rotation, params: {rebalance: M}, grid: {lookback: [20, 60]}}]",
            "autopilot.train_bars=300", "autopilot.holdout_bars=100", "autopilot.min_improvement=0.0"]
    sets += [f"{k}={v}" for k, v in extra.items()]
    return load_config(None, sets)


def test_reoptimize_records_decision(tmp_path):
    ap = Autopilot(make_cfg(tmp_path), source=NowSource())
    d = ap.reoptimize()
    assert len(d["candidates"]) == 2 and d["reason"]
    state = json.loads((tmp_path / "backtest_autopilot.json").read_text())
    assert state["last_reopt"] and len(state["history"]) == 1
    if d["switched"]:
        assert state["active"] == d["new"]
    assert not ap.due(ap.now())


def test_threshold_blocks_switch(tmp_path):
    ap = Autopilot(make_cfg(tmp_path, **{"autopilot.min_improvement": 99}), source=NowSource())
    d = ap.reoptimize()
    assert not d["switched"] and ap.state["active"]["name"] == "sma_cross"


def test_trade_uses_active_strategy_and_health_check(tmp_path):
    cfg = make_cfg(tmp_path, **{"autopilot.max_live_drawdown": 0.0001})
    ap = Autopilot(cfg, source=NowSource())
    ap.state["active"] = {"name": "buy_and_hold", "params": {}}
    s = ap.trade(source=NowSource())
    assert len(s["orders"]) == 3
    runner_state = tmp_path / "backtest_runner.json"
    st = json.loads(runner_state.read_text())
    st["equity_log"] = [["t0", 100.0], ["t1", 90.0]]
    runner_state.write_text(json.dumps(st))
    from quant.live import LiveRunner

    assert ap.health_check(LiveRunner(ap.active_cfg(), source=NowSource()))
    assert (tmp_path / "STOP").exists()


def test_simulate_runs_continuously(tmp_path):
    from quant.data import load_panel

    cfg = make_cfg(tmp_path)
    ap = Autopilot(cfg)
    panel = load_panel(SyntheticSource(), ["AAA", "BBB", "CCC"], "2016-01-01", "2021-12-31")
    res, decisions = ap.simulate(panel, "2018-01-01", every=120)
    assert len(decisions) >= 3
    assert res.equity.index.is_monotonic_increasing and not res.equity.index.duplicated().any()
    assert res.trades["pnl"].sum() == __import__("pytest").approx(res.equity.iloc[-1] - res.initial_cash, rel=1e-6)
