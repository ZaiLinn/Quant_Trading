"""逐 K 线事件驱动回测引擎（numpy 实现）。

时序（每根 K 线 t）：
  1. 开盘：执行 t-1 收盘产生的目标权重（含滑点、费用、手数、T+1、涨跌停、停牌、资金约束）；
  2. 盘中：用最高/最低价检查止损、止盈、移动止损（跳空时按开盘价成交）；
  3. 收盘：按收盘价估值，记录权益与持仓。
只在目标权重变化时交易（或持仓偏离超过 drift_threshold），避免无意义的每日再平衡。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..execution import EPS, exec_price, is_margin, plan_orders
from ..market import MarketRules

FILL_COLS = ["date", "symbol", "qty", "price", "value", "fee", "reason"]
TRADE_COLS = ["symbol", "side", "entry_date", "exit_date", "entry_price", "exit_price",
              "max_qty", "pnl", "return", "bars", "fees", "exit_reason"]


@dataclass
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    positions: pd.DataFrame        # 持仓数量
    weights: pd.DataFrame          # 收盘持仓权重
    fills: pd.DataFrame            # 逐笔成交
    trades: pd.DataFrame           # 完整交易回合（开仓到平仓）
    targets: pd.DataFrame          # 策略目标权重
    rules: MarketRules
    initial_cash: float
    benchmark: pd.Series | None = None   # 基准权益曲线（同样初始资金）
    meta: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    def metrics(self) -> dict:
        from ..analysis.metrics import compute_metrics

        return compute_metrics(self)


class _Book:
    """单标的交易回合记账。"""

    __slots__ = ("side", "entry_date", "entry_bar", "cost", "qty", "max_qty", "realized", "fees",
                 "exit_value", "exit_qty")

    def __init__(self, side: int, date, bar: int):
        self.side, self.entry_date, self.entry_bar = side, date, bar
        self.cost = self.qty = self.max_qty = self.realized = self.fees = 0.0
        self.exit_value = self.exit_qty = 0.0


class BacktestEngine:
    def __init__(self, rules: MarketRules, initial_cash: float = 1_000_000,
                 stop_loss: float | None = None, take_profit: float | None = None,
                 trailing_stop: float | None = None, drift_threshold: float | None = None,
                 min_order_value: float = 0.0, max_volume_pct: float | None = None):
        self.rules = rules
        self.initial_cash = float(initial_cash)
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.drift_threshold = drift_threshold
        self.min_order_value = min_order_value
        self.max_volume_pct = max_volume_pct  # 单根成交量参与率上限（以上一根成交量估算，避免用到当根未来数据）

    # ------------------------------------------------------------------ 主循环
    def run(self, panel: Panel, targets: pd.DataFrame,
            benchmark: pd.Series | None = None) -> BacktestResult:
        rules = self.rules
        idx, syms = panel.index, panel.symbols
        T, N = len(idx), len(syms)
        O, H, L, C, V = (panel[f].to_numpy(float) for f in ("open", "high", "low", "close", "volume"))
        Cff = panel.close.ffill().to_numpy(float)
        tgt = targets.reindex(index=idx, columns=syms).to_numpy(float)
        limits = np.array([rules.limit_for(s) or np.nan for s in syms])
        has_limit = np.isfinite(limits).any()
        days = idx.normalize()

        self._syms, self._idx = syms, idx
        self._lots = rules.lots_for(syms)
        self.cash = self.initial_cash
        self.pos = np.zeros(N)
        self.avg = np.zeros(N)
        self.locked = np.zeros(N)
        self.peak = np.full(N, np.nan)
        self.books: list[_Book | None] = [None] * N
        self.fills: list[tuple] = []
        self.trades: list[tuple] = []
        last_tgt = np.full(N, np.nan)
        pending = np.zeros(N, dtype=bool)
        blocked_stop = np.zeros(N, dtype=bool)

        eq_arr = np.zeros(T)
        cash_arr = np.zeros(T)
        pos_arr = np.zeros((T, N))
        busted = False

        for t in range(T):
            if t == 0 or days[t] != days[t - 1]:
                self.locked[:] = 0.0
            prev_c = Cff[t - 1] if t > 0 else np.full(N, np.nan)

            # ---- 1. 开盘执行
            if t > 0 and not busted:
                desired = tgt[t - 1]
                changed = np.isfinite(desired) & (np.isnan(last_tgt) | (np.abs(desired - last_tgt) > EPS))
                last_tgt[changed] = desired[changed]
                blocked_stop[changed] = False
                pending[changed] = False
                mask = changed | pending
                val_px = np.where(np.isfinite(O[t]), O[t], prev_c)
                equity_open = self.cash + np.nansum(self.pos * val_px)
                if self.drift_threshold is not None and equity_open > 0:
                    cur_w = np.where(np.isfinite(val_px), self.pos * val_px, 0.0) / equity_open
                    drift = np.isfinite(last_tgt) & ~blocked_stop & (np.abs(cur_w - last_tgt) > self.drift_threshold)
                    mask |= drift
                if mask.any():
                    tradable = np.isfinite(O[t]) & (V[t] > 0)
                    can_buy = can_sell = None
                    if has_limit:
                        up = prev_c * (1 + limits)
                        dn = prev_c * (1 - limits)
                        # 一字涨停买不进、一字跌停卖不出（开盘即在涨跌停价）
                        can_buy = ~(np.isfinite(up) & (O[t] >= up * (1 - 1e-4)))
                        can_sell = ~(np.isfinite(dn) & (O[t] <= dn * (1 + 1e-4)))
                    sellable = self.pos - self.locked if rules.t_plus_1 else None
                    price = np.where(tradable, O[t], np.nan)
                    max_qty = self.max_volume_pct * V[t - 1] if self.max_volume_pct else None
                    delta, blocked = plan_orders(last_tgt, self.pos, price, equity_open, self.cash,
                                                 rules, mask, can_buy, can_sell, sellable,
                                                 self.min_order_value, max_qty, self._lots)
                    pending = mask & blocked
                    # 先卖后买，释放资金
                    for i in np.argsort(delta):
                        if delta[i] != 0:
                            self._fill(t, i, delta[i], O[t, i], "signal")

            # ---- 2. 盘中止损/止盈
            if (self.stop_loss or self.take_profit or self.trailing_stop) and not busted:
                for i in np.nonzero(self.pos)[0]:
                    if self._check_exit(t, i, O[t, i], H[t, i], L[t, i], prev_c[i], limits[i]):
                        blocked_stop[i] = True
                        pending[i] = False
                held = self.pos != 0
                self.peak = np.where(held & (self.pos > 0), np.fmax(self.peak, H[t]), self.peak)
                self.peak = np.where(held & (self.pos < 0), np.fmin(self.peak, L[t]), self.peak)

            # ---- 3. 收盘估值
            equity = self.cash + np.nansum(self.pos * Cff[t])
            if equity <= 0:
                busted = True  # 爆仓：停止交易，此后权益恒为 0
            eq_arr[t] = 0.0 if busted else equity
            cash_arr[t] = self.cash
            pos_arr[t] = self.pos

        # 期末未平仓的回合按最后收盘价记为未实现
        for i, book in enumerate(self.books):
            if book is not None and self.pos[i] != 0 and np.isfinite(Cff[-1, i]):
                self._close_book(i, T - 1, Cff[-1, i] * abs(self.pos[i]), abs(self.pos[i]),
                                 (Cff[-1, i] - self.avg[i]) * self.pos[i], "open")

        equity_s = pd.Series(eq_arr, index=idx, name="equity")
        positions = pd.DataFrame(pos_arr, index=idx, columns=syms)
        mv = positions * panel.close.ffill().fillna(0.0)
        weights = mv.div(equity_s.replace(0, np.nan), axis=0).fillna(0.0)
        bench = None
        if benchmark is not None:
            b = benchmark.reindex(idx.union(benchmark.index)).ffill().reindex(idx).dropna()
            if len(b):
                bench = (b / b.iloc[0] * self.initial_cash).rename("benchmark")
        return BacktestResult(
            equity=equity_s, cash=pd.Series(cash_arr, index=idx, name="cash"),
            positions=positions, weights=weights,
            fills=pd.DataFrame(self.fills, columns=FILL_COLS),
            trades=pd.DataFrame(self.trades, columns=TRADE_COLS),
            targets=targets, rules=rules, initial_cash=self.initial_cash, benchmark=bench,
        )

    # ------------------------------------------------------------------ 成交与记账
    def _fill(self, t: int, i: int, qty: float, ref_price: float, reason: str) -> None:
        rules = self.rules
        px = exec_price(ref_price, qty, rules)
        if qty > 0 and not is_margin(rules):
            # 现金账户不能透支：按可用资金收缩（最低佣金、取整误差兜底）
            value = qty * px
            while qty > 0 and value + rules.fee(value, False, qty) > self.cash + EPS:
                step = self._lots[i] or qty * 0.01
                qty = rules.round_qty(min(qty - step, self.cash / px * 0.999), self._lots[i])
                qty = max(float(qty), 0.0)
                value = qty * px
            if qty <= 0:
                return
        value = abs(qty) * px
        fee = rules.fee(value, qty < 0, qty)
        self.cash -= qty * px + fee

        old = self.pos[i]
        new = old + qty
        if abs(new) < EPS * max(1.0, abs(old)):
            new = 0.0
        date = self._idx[t]
        closed = 0.0
        if old == 0 or np.sign(old) == np.sign(qty):
            self.avg[i] = (self.avg[i] * abs(old) + px * abs(qty)) / abs(new)
        else:
            closed = min(abs(qty), abs(old))
            realized = (px - self.avg[i]) * closed * np.sign(old)
            book = self.books[i]
            if book is not None:
                book.realized += realized
                book.exit_value += closed * px
                book.exit_qty += closed
        # 回合记账
        if old != 0 and closed > 0:
            book = self.books[i]
            # 手续费按平仓比例计入当前回合，翻仓部分计入新回合
            book.fees += fee * closed / abs(qty)
            if new == 0 or np.sign(new) != np.sign(old):
                self._close_book(i, t, 0.0, 0.0, 0.0, reason)
                self.avg[i] = px if new != 0 else 0.0
        if new != 0 and (old == 0 or np.sign(new) != np.sign(old)):
            self.books[i] = _Book(int(np.sign(new)), date, t)
            self.peak[i] = px
        book = self.books[i]
        if new != 0 and book is not None:
            opened = abs(qty) - closed
            if opened > 0:
                book.cost += opened * px
                book.qty += opened
                book.fees += fee * opened / abs(qty)
            book.max_qty = max(book.max_qty, abs(new))
        self.pos[i] = new
        if qty > 0 and rules.t_plus_1:
            self.locked[i] += qty
        self.fills.append((date, self._syms[i], qty, px, value, fee, reason))

    def _close_book(self, i: int, t: int, extra_value: float, extra_qty: float,
                    extra_pnl: float, reason: str) -> None:
        book = self.books[i]
        if book is None:
            return
        exit_qty = book.exit_qty + extra_qty
        exit_value = book.exit_value + extra_value
        pnl = book.realized + extra_pnl - book.fees
        entry_px = book.cost / book.qty if book.qty else np.nan
        exit_px = exit_value / exit_qty if exit_qty else np.nan
        ret = pnl / book.cost if book.cost else np.nan
        self.trades.append((self._syms[i], "long" if book.side > 0 else "short", book.entry_date,
                            self._idx[t], entry_px, exit_px, book.max_qty, pnl, ret,
                            t - book.entry_bar, book.fees, reason))
        self.books[i] = None

    def _check_exit(self, t: int, i: int, o: float, h: float, l: float, prev_c: float,
                    limit: float) -> bool:
        """盘中止损/止盈，触发则平掉可卖部分。返回是否触发。"""
        if not (np.isfinite(l) and np.isfinite(h)):
            return False
        pos, avg = self.pos[i], self.avg[i]
        qty = pos - self.locked[i] if (self.rules.t_plus_1 and pos > 0) else pos
        if qty == 0:
            return False
        side = np.sign(pos)
        stop = np.nan
        if self.stop_loss:
            stop = avg * (1 - side * self.stop_loss)
        if self.trailing_stop and np.isfinite(self.peak[i]):
            trail = self.peak[i] * (1 - side * self.trailing_stop)
            stop = trail if np.isnan(stop) else (max(stop, trail) if side > 0 else min(stop, trail))
        if np.isfinite(limit) and np.isfinite(prev_c):
            # 全天封死跌停（多头）/ 涨停（空头）无法离场
            if side > 0 and h <= prev_c * (1 - limit) * (1 + 1e-4):
                return False
            if side < 0 and l >= prev_c * (1 + limit) * (1 - 1e-4):
                return False
        o = o if np.isfinite(o) else prev_c
        if np.isfinite(stop) and ((side > 0 and l <= stop) or (side < 0 and h >= stop)):
            price = min(o, stop) if side > 0 else max(o, stop)
            self._fill(t, i, -qty, price, "stop")
            return True
        if self.take_profit:
            tp = avg * (1 + side * self.take_profit)
            if (side > 0 and h >= tp) or (side < 0 and l <= tp):
                price = max(o, tp) if side > 0 else min(o, tp)
                self._fill(t, i, -qty, price, "take_profit")
                return True
        return False
