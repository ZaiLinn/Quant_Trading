import numpy as np
import pandas as pd
import pytest

from conftest import make_panel
from quant import get_rules, get_strategy, run_backtest
from quant.data import SyntheticSource, load_panel
from quant.factor import FACTORS, combine, compute_factor, evaluate, forward_returns, ic_series
from quant.factor.library import parse_spec
from quant.strategy import check_lookahead
from quant.strategy.library import erc_weights


@pytest.fixture(scope="module")
def universe():
    return load_panel(SyntheticSource(), [f"S{i}" for i in range(15)], "2019-01-01", "2022-12-31")


@pytest.mark.parametrize("name", sorted(FACTORS))
def test_factor_causal(name, universe):
    full = compute_factor(universe, name)
    t = 400
    part = compute_factor(universe.iloc(slice(0, t + 1)), name)
    a, b = full.iloc[t].to_numpy(), part.iloc[-1].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_parse_spec():
    assert parse_spec("momentum_60_5") == ("momentum", [60, 5])
    with pytest.raises(ValueError):
        parse_spec("nope_5")


def test_forward_returns_use_next_open():
    p = make_panel({"A": [10, 10, 10, 10]}, opens={"A": [10, 11, 12.1, 13.31]})
    f = forward_returns(p, (1,))[1]
    assert f["A"].iloc[0] == pytest.approx(0.1)  # t=0 的因子 -> t+1 开盘买、t+2 开盘卖
    assert np.isnan(f["A"].iloc[-1])


def test_perfect_factor_has_ic_one(universe):
    fwd = forward_returns(universe, (1,))[1]
    ic = ic_series(fwd, fwd)
    assert ic.mean() == pytest.approx(1.0)


def test_evaluate_report(universe):
    rep = evaluate(universe, "momentum_20", horizons=(1, 5))
    s = rep.summary()
    assert set(rep.ic["horizon"]) == {1, 5}
    assert list(rep.quantile.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert -1 <= s["rank_ic"] <= 1 and 0 <= s["top_turnover"] <= 1


def test_combine_is_zscored(universe):
    z = combine(universe, {"momentum_20": 1.0})
    row = z.iloc[-1].dropna()
    assert abs(row.mean()) < 1e-9


def test_factor_topk_holds_k_and_no_lookahead(universe):
    st = get_strategy("factor_topk", top_k=5, buffer=2, min_bars=20)
    w = st.generate(universe).dropna(how="all")
    assert ((w > 0).sum(axis=1).iloc[5:] == 5).all()
    assert check_lookahead(st, universe, n_checks=4) == []
    r = run_backtest(st, universe, get_rules("ashare"), 1_000_000)
    assert (r.positions > 0).sum(axis=1).max() <= 5


def test_erc_equal_risk_contribution():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(200, 4)) * [0.01, 0.02, 0.005, 0.03]
    cov = np.cov(a, rowvar=False)
    w = erc_weights(cov)
    rc = w * (cov @ w)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(rc / rc.sum(), 0.25, atol=1e-4)


def test_grid_hysteresis():
    # 价格在同一网格线附近来回波动时不应反复交易
    prices = [100, 95, 95.5, 94.8, 95.4, 94.9, 95.3, 90, 95, 100]
    p = make_panel({"A": prices})
    w = get_strategy("grid", levels=10, lower=50, upper=150).generate(p)["A"]
    assert w.diff().fillna(0).ne(0).sum() <= 3
