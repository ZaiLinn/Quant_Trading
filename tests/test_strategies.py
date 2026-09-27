import numpy as np
import pandas as pd
import pytest

from quant import get_rules
from quant.pipeline import apply_risk, build_targets
from quant.strategy import STRATEGIES, Strategy, check_lookahead, get_strategy, rebalance_mask


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_no_lookahead(name, panel):
    strat = get_strategy(name)
    assert check_lookahead(strat, panel, n_checks=6) == []


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_weights_valid(name, panel):
    t = build_targets(get_strategy(name), panel, get_rules("ashare"))
    assert list(t.columns) == panel.symbols
    v = t.to_numpy()
    assert np.isfinite(v[~np.isnan(v)]).all()
    assert (np.nansum(np.abs(v), axis=1) <= 1 + 1e-9).all()
    assert (np.nan_to_num(v) >= 0).all()  # A 股不做空


def test_lookahead_detector_catches_future_function(panel):
    class Cheat(Strategy):
        name = "cheat"
        params = {}

        def generate(self, p):
            return (p.close.shift(-1) > p.close).astype(float) / len(p.symbols)

    assert check_lookahead(Cheat(), panel)


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        get_strategy("sma_cross", fastt=5)


def test_rebalance_mask_first_bar_of_period():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    m = rebalance_mask(idx, "M")
    assert list(idx[m]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]


def test_apply_risk_caps_and_keeps_nan(panel):
    t = pd.DataFrame(0.6, index=panel.index, columns=panel.symbols)
    t.iloc[5] = np.nan
    out = apply_risk(t, panel, get_rules("generic"), max_weight=0.5, weight_step=0.1)
    assert out.iloc[5].isna().all()
    row = out.iloc[10]
    assert row.abs().sum() <= 1 + 1e-9 and (row <= 0.5).all()
    assert np.allclose((row / 0.1).round(), row / 0.1)


def test_vol_target_reduces_exposure(panel):
    t = pd.DataFrame(1 / 3, index=panel.index, columns=panel.symbols)
    out = apply_risk(t, panel, get_rules("generic"), vol_target=0.05)
    assert out.iloc[-100:].abs().sum(axis=1).max() < 1


def test_regime_filter_goes_to_cash_in_downtrend():
    from conftest import make_panel

    prices = list(np.linspace(10, 20, 60)) + list(np.linspace(20, 8, 60))
    p = make_panel({"A": prices, "B": prices})
    t = pd.DataFrame(np.nan, index=p.index, columns=p.symbols)
    t.iloc[0] = 0.5  # 只在第一根给出目标，之后一直"保持"
    out = apply_risk(t, p, get_rules("generic"), regime_ma=20)
    held = out.ffill()
    assert (held.iloc[50] == 0.5).all()      # 上涨阶段满仓
    assert (held.iloc[-1] == 0.0).all()      # 跌破均线后空仓
    assert out.iloc[1:50].isna().all().all()  # 状态未切换时不产生多余调仓
