"""模拟盘/实盘运行器。

与回测严格同构：
- 只用"已收盘"的 K 线计算信号（未完成的当前 K 线被剔除，只用其最新价作为下单参考价）；
- 目标权重来自同一个 build_targets()，下单数量来自同一个 plan_orders()；
- 只在目标变化（或上次因停牌/涨跌停未成交）时交易，重复运行是幂等的。
建议 A 股在开盘后运行（如 9:35，对应回测"次日开盘成交"），加密货币在日线收盘后运行（如 UTC 00:05）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..app import make_rules, make_strategy, warmup_start
from ..data import DataSource, Panel, freq_to_timedelta, load_panel, make_source, resolve_symbols
from ..execution import plan_orders
from ..pipeline import build_targets
from .broker import Broker, CcxtBroker, PaperBroker
from .guard import RiskGuard
from .notify import Notifier

log = logging.getLogger(__name__)

ASHARE_CLOSE = (15, 0)


class LiveRunner:
    def __init__(self, cfg: dict, broker: Broker | None = None, source: DataSource | None = None,
                 notifier: Notifier | None = None):
        self.cfg = cfg
        lv = cfg["live"]
        self.rules = make_rules(cfg)
        self.strategy = make_strategy(cfg)
        self.symbols = resolve_symbols(cfg["data"])
        self.freq = cfg["data"].get("freq", "1d")
        self.risk = cfg.get("risk") or {}
        self.tz = lv.get("timezone") or ("Asia/Shanghai" if self.rules.name.startswith("ashare") else "UTC")
        state_dir = Path(lv.get("state_dir", "./live_state"))
        self.state_path = state_dir / f"{cfg['name']}_runner.json"
        self.state = self._load_state()
        # 实盘需要未收盘的当前 K 线来取最新价；信号计算前由 load_data() 剔除
        self.source = source or make_source({**cfg["data"], "drop_incomplete": False}, cache=False)
        self.broker = broker or self._make_broker(state_dir)
        self.guard = RiskGuard(self.rules, lv.get("max_daily_loss"), lv.get("max_order_value"),
                               lv.get("kill_switch_file"))
        self.notifier = notifier or Notifier(lv.get("webhook"), lv.get("webhook_kind", "generic"))

    # ------------------------------------------------------------------ 状态
    def _make_broker(self, state_dir: Path) -> Broker:
        lv = self.cfg["live"]
        if lv.get("broker", "paper") == "paper":
            return PaperBroker(self.rules, state_dir / f"{self.cfg['name']}_paper.json",
                               self.cfg.get("initial_cash", 1_000_000))
        if lv["broker"] == "ccxt":
            return CcxtBroker(self.cfg["data"].get("exchange", "binance"), self.rules, self.symbols,
                              quote=lv.get("quote", "USDT"), sandbox=lv.get("sandbox", False),
                              dry_run=lv.get("dry_run", True),
                              options=self.cfg["data"].get("exchange_options"))
        raise ValueError(f"未知 broker: {lv['broker']}")

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"last_targets": {}, "pending": [], "day": "", "day_start_equity": None,
                "entry": {}, "peak": {}, "stopped": {}, "equity_log": []}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=self.tz).tz_localize(None)

    # ------------------------------------------------------------------ 数据
    def is_trading_day(self, now: pd.Timestamp) -> bool:
        if not self.rules.name.startswith("ashare"):
            return True
        if now.weekday() >= 5:
            return False
        try:
            from ..data.akshare_source import trade_calendar

            return now.normalize() in trade_calendar()
        except Exception as e:  # noqa: BLE001 日历获取失败时仅按周末判断
            log.warning("交易日历获取失败(%s)，按工作日处理", e)
            return True

    def load_data(self, now: pd.Timestamp) -> tuple[Panel, pd.Series]:
        """返回 (已收盘 K 线面板, 最新参考价)。"""
        bars = self.strategy.warmup() + int(self.risk.get("vol_window", 20)) + 5
        start = warmup_start(now.normalize(), bars, self.freq)
        panel = load_panel(self.source, self.symbols, start, None, self.freq)
        last_price = panel.close.ffill().iloc[-1]
        step = freq_to_timedelta(self.freq)
        last = panel.index[-1]
        if self.rules.name.startswith("ashare") and step >= pd.Timedelta(days=1):
            closed = now.time() >= datetime(2000, 1, 1, *ASHARE_CLOSE).time()
            incomplete = last.normalize() == now.normalize() and not closed
        else:
            incomplete = last + step > now
        if incomplete:
            panel = panel.iloc(slice(0, -1))
        return panel, last_price

    # ------------------------------------------------------------------ 主流程
    def run_once(self, force: bool = False) -> dict:
        """force=True 时忽略交易日历（调试用）。"""
        now = self.now()
        if not force and not self.is_trading_day(now):
            log.info("%s 非交易日，跳过", now.date())
            return {"skipped": "非交易日"}
        panel, price_s = self.load_data(now)
        prev_close = panel.close.ffill().iloc[-1] if len(panel) else pd.Series(dtype=float)
        price_s, prev_close = self._realtime(panel.symbols, price_s, prev_close, now)
        targets = build_targets(self.strategy, panel, self.rules, self.risk)
        desired_s = targets.ffill().iloc[-1]
        pos_d = self.broker.positions()
        # 已不在股票池（如指数成分调整、配置删除标的）但仍有持仓的标的：目标设为 0，清仓
        orphans = [s for s, q in pos_d.items() if q and s not in panel.symbols]
        if orphans:
            log.warning("持仓 %s 不在当前股票池中，将清仓", orphans)
            for s in orphans:
                desired_s[s] = 0.0
                price_s[s] = self._latest_price(s, now)
        syms = list(panel.symbols) + orphans
        desired = desired_s.reindex(syms).to_numpy(float)
        price = price_s.reindex(syms).to_numpy(float)
        pos = np.array([pos_d.get(s, 0.0) for s in syms])
        cash = self.broker.cash()
        equity = cash + np.nansum(pos * price)
        today = now.strftime("%Y-%m-%d")
        if self.state["day"] != today:
            self.state["day"] = today
            self.state["day_start_equity"] = equity
        halt = self.guard.halt_reason(equity, self.state["day_start_equity"])

        last = np.array([self.state["last_targets"].get(s, np.nan) for s in syms], dtype=float)
        changed = np.isfinite(desired) & (np.isnan(last) | (np.abs(desired - last) > 1e-9))
        pending = np.array([s in self.state["pending"] for s in syms])
        mask = changed | pending

        # 离场规则（止损/移动止损），与回测同一套参数；触发后在目标变化前不再入场
        tgt = desired.copy()
        stopped_before, stops = self._check_stops(syms, pos, price, changed)
        for i in stopped_before + stops:
            tgt[i] = 0.0
            mask[i] = mask[i] or pos[i] != 0

        drift = self.risk.get("drift_threshold")
        if drift is not None and equity > 0:
            cur_w = np.where(np.isfinite(price), pos * price, 0.0) / equity
            stopped = np.array([s in self.state["stopped"] for s in syms])
            mask |= np.isfinite(tgt) & ~stopped & (np.abs(cur_w - tgt) > drift)

        can_buy = can_sell = None
        limits = np.array([self.rules.limit_for(s) or np.nan for s in syms])
        if np.isfinite(limits).any() and len(panel) > 0:
            prev = prev_close.reindex(syms).to_numpy(float)
            can_buy = ~(price >= prev * (1 + limits) * (1 - 1e-4))
            can_sell = ~(price <= prev * (1 - limits) * (1 + 1e-4))
        sell_d = self.broker.sellable()
        sellable = np.array([sell_d.get(s, 0.0) for s in syms]) if self.rules.t_plus_1 else None

        max_qty = None
        if self.risk.get("max_volume_pct"):
            vol = panel.volume.iloc[-1].reindex(syms).to_numpy(float)
            max_qty = self.risk["max_volume_pct"] * vol
        delta, blocked = plan_orders(tgt, pos, price, equity, cash, self.rules, mask, can_buy, can_sell,
                                     sellable, float(self.risk.get("min_order_value") or 0.0), max_qty)
        guarded = self.guard.filter(delta, pos, price, halted=halt is not None)
        clipped = np.abs(guarded) < np.abs(delta) - 1e-12  # 被风控截断/取消的部分下次继续
        delta = guarded

        fills, failed = [], np.zeros(len(syms), dtype=bool)
        for i in np.argsort(delta):  # 先卖后买
            if delta[i] == 0:
                continue
            try:
                f = self.broker.execute(syms[i], float(delta[i]), float(price[i]))
                if f:
                    fills.append(f)
                    self._track_entry(syms[i], pos[i], pos[i] + f["qty"], f["price"])
            except Exception as e:  # noqa: BLE001 单个标的失败不影响其他标的
                failed[i] = True
                log.exception("%s 下单失败: %s", syms[i], e)

        for i, s in enumerate(syms):
            if changed[i]:
                self.state["last_targets"][s] = float(desired[i])
                self.state["stopped"].pop(s, None)
        for i in stops:
            self.state["stopped"][syms[i]] = float(desired[i])
        self.state["pending"] = [s for i, s in enumerate(syms)
                                 if (mask[i] and blocked[i]) or failed[i] or clipped[i]]
        pos_after = self.broker.positions()
        eq_after = self.broker.cash() + sum(q * float(price_s.get(s, np.nan)) for s, q in pos_after.items()
                                            if np.isfinite(price_s.get(s, np.nan)))
        self.state["equity_log"] = (self.state["equity_log"] + [[now.isoformat(), eq_after]])[-5000:]
        self._save_state()

        summary = {"time": str(now), "bar": str(panel.index[-1]), "equity": eq_after, "halt": halt,
                   "orders": [{"symbol": f["symbol"], "qty": float(f["qty"]), "price": float(f["price"])}
                              for f in fills],
                   "pending": self.state["pending"]}
        lines = [f"[{self.cfg['name']}] {now:%Y-%m-%d %H:%M} 权益 {eq_after:,.2f}"]
        if halt:
            lines.append(f"风控暂停开仓：{halt}")
        lines += [f"{'买入' if f['qty'] > 0 else '卖出'} {f['symbol']} {abs(f['qty']):g} @ {f['price']:.4f}"
                  for f in fills]
        if self.state["pending"]:
            lines.append(f"待成交（停牌/涨跌停/T+1/失败）：{self.state['pending']}")
        if fills or halt or self.state["pending"]:
            self.notifier.send("\n".join(lines))
        else:
            log.info("%s 无需调仓", lines[0])
        return summary

    def _realtime(self, syms: list[str], price: pd.Series, prev_close: pd.Series,
                  now: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
        """A 股用新浪实时行情覆盖参考价与昨收（日线数据源盘中可能还没有当天的 K 线）。"""
        if not self.rules.name.startswith("ashare") or self.cfg["data"].get("source") != "akshare":
            return price, prev_close
        try:
            from ..data.akshare_source import sina_quotes

            q = sina_quotes(syms, self.cfg["data"].get("asset_type", "auto"))
        except Exception as e:  # noqa: BLE001 实时行情失败时退回日线价格
            log.warning("实时行情获取失败(%s)，使用日线收盘价", e)
            return price, prev_close
        if q.empty:
            return price, prev_close
        today = q[q["time"].dt.normalize() == now.normalize()]
        price, prev_close = price.copy(), prev_close.copy()
        price.update(today["price"])
        prev_close.update(today["prev_close"])
        log.info("实时行情覆盖 %d/%d 个标的", len(today), len(syms))
        return price, prev_close

    def _latest_price(self, symbol: str, now: pd.Timestamp) -> float:
        try:
            df = self.source.fetch(symbol, now - pd.Timedelta(days=15), None, self.freq)
            return float(df["close"].iloc[-1]) if len(df) else np.nan
        except Exception as e:  # noqa: BLE001 取不到价格则本次跳过，下次重试
            log.warning("%s 最新价获取失败: %s", symbol, e)
            return np.nan

    def _track_entry(self, sym: str, old: float, new: float, px: float) -> None:
        if new == 0:
            self.state["entry"].pop(sym, None)
            self.state["peak"].pop(sym, None)
        elif old == 0 or np.sign(old) != np.sign(new):
            self.state["entry"][sym] = px
            self.state["peak"][sym] = px

    def _check_stops(self, syms, pos, price, changed) -> tuple[list[int], list[int]]:
        """返回 (此前已止损且目标未变的标的, 本次新触发离场的标的)。"""
        sl, tp, trail = (self.risk.get(k) for k in ("stop_loss", "take_profit", "trailing_stop"))
        before, hits = [], []
        for i, s in enumerate(syms):
            if s in self.state["stopped"] and not changed[i]:
                before.append(i)
                continue
            if not (sl or tp or trail) or pos[i] == 0 or not np.isfinite(price[i]):
                continue
            side = np.sign(pos[i])
            entry = self.state["entry"].get(s)
            if entry is None:
                continue
            peak = self.state["peak"].get(s, entry)
            peak = max(peak, price[i]) if side > 0 else min(peak, price[i])
            self.state["peak"][s] = peak
            r = (price[i] / entry - 1) * side
            if (sl and r <= -sl) or (tp and r >= tp) or \
                    (trail and (price[i] / peak - 1) * side <= -trail):
                log.info("%s 触发离场规则（收益 %.2f%%）", s, r * 100)
                hits.append(i)
        return before, hits

    def run_forever(self) -> None:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger

        cron = self.cfg["live"].get("cron")
        if not cron:
            raise ValueError("live.cron 未配置，例如 '35 9 * * mon-fri'")
        sched = BlockingScheduler(timezone=self.tz)

        def job():
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001 调度任务异常不能让进程退出
                log.exception("运行失败")
                self.notifier.send(f"[{self.cfg['name']}] 运行失败: {e}")

        sched.add_job(job, CronTrigger.from_crontab(cron, timezone=self.tz), max_instances=1,
                      coalesce=True, misfire_grace_time=300)
        log.info("已启动调度 cron='%s' tz=%s，Ctrl+C 退出", cron, self.tz)
        try:
            sched.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("已停止")
