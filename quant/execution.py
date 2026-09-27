"""目标权重 -> 下单数量。回测引擎、模拟盘、实盘共用，保证下单逻辑只有一份。"""
from __future__ import annotations

import numpy as np

from .market import MarketRules

EPS = 1e-9


def exec_price(price: float, qty: float, rules: MarketRules) -> float:
    """含滑点的成交价：买入更贵、卖出更便宜。"""
    return price * (1 + rules.slippage * np.sign(qty))


def is_margin(rules: MarketRules) -> bool:
    return rules.allow_short or rules.max_leverage > 1


def plan_orders(target_w: np.ndarray, pos: np.ndarray, price: np.ndarray, equity: float,
                cash: float, rules: MarketRules, mask: np.ndarray,
                can_buy: np.ndarray | None = None, can_sell: np.ndarray | None = None,
                sellable: np.ndarray | None = None,
                min_order_value: float = 0.0,
                max_qty: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """计算各标的下单数量（正买负卖）。

    返回 (delta, blocked)：blocked 表示因停牌/涨跌停/T+1/容量限制未能（全部）成交、需要下一根继续的标的。
    max_qty 为单笔数量上限（如上一根成交量 × 参与率），超出部分分多根执行。
    资金不足时按比例缩减加仓单；平仓单优先释放资金。
    """
    n = len(pos)
    delta = np.zeros(n)
    blocked = np.zeros(n, dtype=bool)
    valid_px = np.isfinite(price) & (price > 0)
    blocked |= mask & ~valid_px
    ok = mask & valid_px & np.isfinite(target_w)
    if not ok.any() or equity <= 0:
        return delta, blocked

    px = np.where(valid_px, price, 1.0)
    tw = np.where(ok, target_w, 0.0)
    if not rules.allow_short:
        tw = np.maximum(tw, 0.0)
    raw = np.where(ok, tw * equity / px - pos, 0.0)
    # 目标为 0 时全部平掉（A 股卖出允许零股），否则按手数向零取整
    delta = np.where(ok & (tw == 0), -pos, rules.round_qty(raw))

    if can_buy is not None:
        hit = (delta > 0) & ~can_buy
        blocked |= hit
        delta[hit] = 0.0
    if can_sell is not None:
        hit = (delta < 0) & ~can_sell
        blocked |= hit
        delta[hit] = 0.0
    if sellable is not None:
        # T+1：多头卖出数量不超过可卖数量
        cap = -np.maximum(sellable, 0.0)
        hit = (delta < 0) & (pos > 0) & (delta < cap - EPS)
        blocked |= hit
        delta = np.where(hit, cap, delta)

    if max_qty is not None:
        cap = rules.round_qty(np.where(np.isfinite(max_qty), np.maximum(max_qty, 0.0), np.inf))
        hit = np.abs(delta) > cap + EPS
        blocked |= hit
        delta = np.where(hit, np.sign(delta) * cap, delta)

    if min_order_value > 0:
        small = np.abs(delta) * px < min_order_value
        # 清仓单即使金额很小也要执行，避免残留碎仓
        small &= ~(ok & (tw == 0))
        delta[small] = 0.0

    # 拆分为"减仓部分"与"加仓部分"（多翻空时两者都有）
    reduce = np.where(np.sign(delta) == -np.sign(pos),
                      np.sign(delta) * np.minimum(np.abs(delta), np.abs(pos)), 0.0)
    inc = delta - reduce
    if not np.any(inc):
        return reduce, blocked

    slip = rules.slippage
    if is_margin(rules):
        exposure_after_reduce = np.sum(np.abs(pos + reduce) * px)
        budget = equity * rules.max_leverage - exposure_after_reduce
        need = np.sum(np.abs(inc) * px)
    else:
        red_value = np.abs(reduce) * px * (1 - slip)
        cash_after = cash + red_value.sum() - sum(rules.fee(v, True) for v in red_value if v > 0)
        buy_value = inc * px * (1 + slip)
        budget = cash_after
        need = buy_value.sum() + sum(rules.fee(v, False) for v in buy_value if v > 0)

    if need > budget + EPS:
        scale = max(budget, 0.0) / need
        for _ in range(5):  # 取整与最低佣金可能导致仍超预算，逐步收缩
            inc_s = rules.round_qty(inc * scale)
            if is_margin(rules):
                need_s = np.sum(np.abs(inc_s) * px)
            else:
                bv = inc_s * px * (1 + slip)
                need_s = bv.sum() + sum(rules.fee(v, False) for v in bv if v > 0)
            if need_s <= budget + EPS:
                break
            scale *= 0.98
        else:
            inc_s = np.zeros(n)
        inc = inc_s
        if min_order_value > 0:
            inc[np.abs(inc) * px < min_order_value] = 0.0
    return reduce + inc, blocked
