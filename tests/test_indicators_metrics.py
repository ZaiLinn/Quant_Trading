import numpy as np
import pandas as pd
import pytest

from quant import indicators as ind
from quant.analysis.metrics import cagr, drawdown, max_drawdown_duration, monthly_returns, sharpe
from quant.analysis.robustness import bootstrap, deflated_sharpe, probabilistic_sharpe
from quant.data import normalize


def test_sma_ema_rsi_basic():
    s = pd.Series(np.arange(1, 31, dtype=float))
    assert ind.sma(s, 5).iloc[4] == pytest.approx(3.0)
    assert np.isnan(ind.sma(s, 5).iloc[3])
    assert ind.rsi(s, 14).iloc[-1] == pytest.approx(100.0)
    assert ind.rsi(-s, 14).iloc[-1] == pytest.approx(0.0)
    e = ind.ema(s, 10)
    assert e.iloc[-1] < s.iloc[-1] and e.iloc[-1] > ind.sma(s, 10).iloc[-1] - 5


def test_macd_bollinger_atr_shapes():
    rng = np.random.default_rng(0)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    dif, dea, hist = ind.macd(c)
    assert np.allclose((dif - dea).dropna(), hist.dropna())
    mid, up, lo = ind.bollinger(c)
    assert (up.dropna() >= mid.dropna()).all() and (lo.dropna() <= mid.dropna()).all()
    a = ind.atr(c * 1.01, c * 0.99, c, 14)
    assert (a.dropna() > 0).all()


def test_donchian_excludes_current_bar():
    h = pd.Series([1, 2, 3, 4, 5], dtype=float)
    up, _ = ind.donchian(h, h, 2)
    assert up.iloc[4] == 4  # 前两根(3,4)的最高价，不含当根 5


def test_metrics_known_values():
    idx = pd.bdate_range("2020-01-01", periods=253)
    eq = pd.Series(100 * 1.001 ** np.arange(253), index=idx)
    years = (idx[-1] - idx[0]).days / 365.25
    assert cagr(eq) == pytest.approx((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)
    assert drawdown(eq).min() == 0
    dd_eq = pd.Series([100, 110, 99, 105, 111, 90], index=idx[:6], dtype=float)
    assert drawdown(dd_eq).min() == pytest.approx(90 / 111 - 1)
    assert max_drawdown_duration(dd_eq) == 2
    r = pd.Series([0.01, -0.01] * 50)
    assert sharpe(r, 252) == pytest.approx(0.0, abs=1e-12)


def test_monthly_returns_table():
    idx = pd.bdate_range("2023-03-15", "2024-02-28")
    eq = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    tbl = monthly_returns(eq)
    assert list(tbl.columns) == list(range(1, 13)) + ["year"]
    assert np.isnan(tbl.loc[2023, 1])
    total = (1 + tbl.drop(columns="year").stack()).prod() - 1
    assert total == pytest.approx(0.2, rel=1e-9)


def test_normalize_fixes_bad_rows():
    df = pd.DataFrame({"open": [1, 2, np.nan], "high": [0.5, 3, 4], "low": [1, 1, 1],
                       "close": [1.2, 2.5, 3], "volume": [1, np.nan, 3]},
                      index=pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"]))
    out = normalize(df)
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert out["volume"].notna().all()


def test_psr_dsr_bootstrap():
    rng = np.random.default_rng(1)
    good = pd.Series(rng.normal(0.002, 0.01, 1000))
    assert probabilistic_sharpe(good) > 0.95
    assert deflated_sharpe(good, n_trials=100) < probabilistic_sharpe(good)
    sim = bootstrap(good, 252, n=200)
    assert sim.shape == (200, 3) and (sim["max_drawdown"] <= 0).all()


def test_quality_report_flags_split():
    from conftest import make_panel
    from quant import get_rules
    from quant.data.quality import quality_report

    p = make_panel({"A": [10, 10.1, 10.2, 2.05, 2.06], "B": [5, 5.1, 5.2, 5.3, 5.4]})
    q = quality_report(p, get_rules("ashare_etf")).set_index("symbol")
    assert "异常跳变" in q.loc["A", "issues"]
    assert q.loc["B", "issues"] == ""


def test_symbol_limits_override():
    from quant import get_rules

    r = get_rules("ashare_etf", symbol_limits={"159915": 0.2})
    assert r.limit_for("159915") == 0.2 and r.limit_for("510300") == 0.10
    a = get_rules("ashare")
    assert a.limit_for("300750") == 0.2 and a.limit_for("688981") == 0.2 and a.limit_for("600519") == 0.1
    assert get_rules("crypto").limit_for("BTC/USDT") is None
