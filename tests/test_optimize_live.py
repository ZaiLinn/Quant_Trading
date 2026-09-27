import json

import numpy as np
import pandas as pd
import pytest

from quant import get_rules
from quant.config import load_config
from quant.data import CachedSource, SyntheticSource, load_panel
from quant.live import LiveRunner, PaperBroker, RiskGuard
from quant.optimize import expand_grid, grid_search, walk_forward


def test_grid_search_sorted(panel):
    res = grid_search("sma_cross", panel, get_rules("generic"), {"fast": [5, 10], "slow": [30, 60]})
    assert len(res) == 4
    assert res["score"].is_monotonic_decreasing
    assert res["fast"].dtype.kind == "i"


def test_grid_search_parallel_matches_serial(panel):
    grid = {"fast": [5, 10], "slow": [30, 60]}
    a = grid_search("sma_cross", panel, get_rules("generic"), grid, jobs=1)
    b = grid_search("sma_cross", panel, get_rules("generic"), grid, jobs=2)
    pd.testing.assert_frame_equal(a, b)


def test_walk_forward_windows(panel):
    wf = walk_forward("sma_cross", panel, get_rules("generic"), {"fast": [5, 10], "slow": [30, 60]},
                      train=400, test=150)
    assert len(wf.windows) >= 3
    assert wf.oos.equity.index.is_monotonic_increasing
    assert wf.oos.equity.index[0] == panel.index[400]
    assert not wf.oos.equity.index.duplicated().any()


def test_plateau_scores_prefer_flat_region():
    from quant.optimize import plateau_scores

    grid = {"a": [1, 2, 3, 4, 5]}
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "score": [0.1, 0.2, 2.0, 0.2, 0.1]})
    sm = plateau_scores(df, grid)
    vals = np.array([0.2, 2.0, 0.2])
    assert sm[2] == pytest.approx(vals.mean() - vals.std())
    df2 = pd.DataFrame({"a": [1, 2, 3, 4, 5], "score": [0.9, 1.0, 0.95, 0.1, 3.0]})
    sm2 = plateau_scores(df2, grid)
    assert sm2.idxmax() == 1  # 平台 (0.9,1.0,0.95) 胜过孤立尖峰 3.0


def test_expand_grid():
    assert expand_grid({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_config_overrides(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("strategy: {name: macd, params: {fast: 8}}\ndata: {symbols: [X]}\n")
    cfg = load_config(f, ["strategy.params.slow=30", "data.symbols=[A, B]"])
    assert cfg["strategy"]["params"] == {"fast": 8, "slow": 30}
    assert cfg["data"]["symbols"] == ["A", "B"]
    assert cfg["data"]["freq"] == "1d"  # 默认值保留


def test_cache_roundtrip(tmp_path):
    src = CachedSource(SyntheticSource(), str(tmp_path), ttl_hours=1)
    a = src.fetch("AAA", "2020-01-01", "2020-12-31")
    b = src.fetch("AAA", "2020-03-01", "2020-06-30")
    assert len(list(tmp_path.glob("*.parquet"))) == 1
    pd.testing.assert_frame_equal(a.loc["2020-03-01":"2020-06-30"], b, check_freq=False)


def test_paper_broker_t1_and_fees(tmp_path):
    rules = get_rules("ashare", slippage=0.0)
    b = PaperBroker(rules, tmp_path / "p.json", 100_000)
    f = b.execute("600000", 1000, 10.0)
    assert f and b.positions()["600000"] == 1000
    assert b.sellable()["600000"] == 0  # T+1
    assert b.cash() == pytest.approx(100_000 - 10_000 - rules.fee(10_000, False))
    b2 = PaperBroker(rules, tmp_path / "p.json")  # 状态持久化
    assert b2.positions() == {"600000": 1000}
    assert b.execute("600000", 100_000, 10.0) is None  # 资金不足


def test_risk_guard(tmp_path):
    rules = get_rules("generic")
    g = RiskGuard(rules, max_daily_loss=0.05, max_order_value=1000, kill_switch_file=str(tmp_path / "STOP"))
    assert g.halt_reason(96, 100) is None
    assert "亏损" in g.halt_reason(94, 100)
    (tmp_path / "STOP").touch()
    assert "熔断" in g.halt_reason(100, 100)
    d = g.filter(np.array([500.0, -500.0]), np.array([0.0, 100.0]), np.array([10.0, 10.0]), halted=True)
    assert d[0] == 0 and d[1] == -100  # 暂停时禁止加仓；单笔上限 1000 元


def test_live_runner_paper_idempotent(tmp_path):
    cfg = load_config(None, [f"live.state_dir={tmp_path}", "data.symbols=[AAA, BBB]",
                             "strategy.name=buy_and_hold", "market=crypto",
                             f"live.kill_switch_file={tmp_path / 'STOP'}"])
    # 合成数据截止到"现在"，让运行器认为有最新 K 线
    runner = LiveRunner(cfg, source=_NowSource())
    s1 = runner.run_once()
    assert len(s1["orders"]) == 2
    s2 = runner.run_once()
    assert s2["orders"] == []
    state = json.loads((tmp_path / f"{cfg['name']}_runner.json").read_text())
    assert state["last_targets"] == {"AAA": 0.5, "BBB": 0.5}


def test_live_runner_liquidates_orphans(tmp_path):
    cfg = load_config(None, [f"live.state_dir={tmp_path}", "data.symbols=[AAA, BBB]",
                             "strategy.name=buy_and_hold", "market=crypto",
                             f"live.kill_switch_file={tmp_path / 'STOP'}"])
    LiveRunner(cfg, source=_NowSource()).run_once()
    cfg["data"]["symbols"] = ["AAA"]  # BBB 被移出股票池
    s = LiveRunner(cfg, source=_NowSource()).run_once()
    sold = [o for o in s["orders"] if o["symbol"] == "BBB"]
    assert sold and sold[0]["qty"] < 0
    assert "BBB" not in PaperBroker(get_rules("crypto"), tmp_path / f"{cfg['name']}_paper.json").positions()


def test_live_runner_capital_limit(tmp_path):
    cfg = load_config(None, [f"live.state_dir={tmp_path}", "data.symbols=[AAA, BBB]",
                             "strategy.name=buy_and_hold", "market=crypto", "live.capital=10000",
                             f"live.kill_switch_file={tmp_path / 'STOP'}"])
    s = LiveRunner(cfg, source=_NowSource()).run_once()
    spent = sum(o["qty"] * o["price"] for o in s["orders"])
    assert 9_900 < spent <= 10_000  # 账户有 100 万现金，但策略只用 1 万
    assert s["equity"] == pytest.approx(10_000, rel=0.01)


class _NowSource(SyntheticSource):
    def fetch(self, symbol, start=None, end=None, freq="1d"):
        end = pd.Timestamp.now().normalize()
        return super().fetch(symbol, end - pd.Timedelta(days=400), end, freq)


def test_load_panel_skips_missing():
    p = load_panel(SyntheticSource(), ["AAA"], "2020-01-01", "2020-02-01")
    assert p.symbols == ["AAA"]
