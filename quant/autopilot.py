"""自动驾驶：模拟盘 / 实盘自动运行 + 定期自动重新优化策略。

流程
  1. 交易任务（live.cron）：用"当前生效"的策略与参数运行一次 LiveRunner；
  2. 优化任务（autopilot.reopt_cron，默认每月）：
     - 取最近 train_bars + holdout_bars 根 K 线；前段为训练集，最后 holdout_bars 根为验证集；
     - 每个候选策略在训练集上网格搜索，按"参数平台得分"选参（不是取孤立最高点）；
     - 候选方案与当前方案都在验证集（训练时没见过的最近数据）上回测；
     - 只有验证集得分比当前方案高出 min_improvement，且交易次数足够，才切换；否则保持不变；
  3. 健康检查：运行以来回撤超过 max_live_drawdown 时创建熔断文件（此后只减仓不开仓）并通知。

自动优化本身也会过拟合，所以这里刻意保守：切换门槛、验证集、平台选参、切换冷却期、完整决策日志。
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .app import data_cfg, make_rules, warmup_start
from .backtest import run_backtest
from .data import DataSource, load_panel, make_source, resolve_symbols
from .live.notify import Notifier
from .live.runner import DEFAULT_TZ, LiveRunner
from .optimize import expand_grid, grid_search
from .strategy import get_strategy

log = logging.getLogger(__name__)

DEFAULTS = {
    "candidates": None,        # [{name, params, grid}]；缺省为 strategy + optimize.grid
    "train_bars": 756,         # 训练集长度（日线约 3 年）
    "holdout_bars": 126,       # 验证集长度（日线约半年）
    "objective": None,         # 缺省用 optimize.objective
    "min_improvement": 0.2,    # 验证集得分至少提高多少才切换
    "min_trades": 3,           # 验证集最少成交笔数（太少说明结果不可信）
    "cooldown_days": 20,       # 两次切换之间至少间隔的天数
    "reopt_cron": "0 18 1 * *",  # 每月 1 日 18:00（交易所当地时间）
    "max_live_drawdown": 0.25,  # 运行以来回撤超过该值即熔断
    "jobs": 1,
}


class Autopilot:
    def __init__(self, cfg: dict, source: DataSource | None = None, notifier: Notifier | None = None):
        self.cfg = cfg
        self.ap = {**DEFAULTS, **(cfg.get("autopilot") or {})}
        self.rules = make_rules(cfg)
        lv = cfg["live"]
        self.tz = lv.get("timezone") or DEFAULT_TZ.get(self.rules.name.split("_")[0], "UTC")
        self.state_path = Path(lv.get("state_dir", "./live_state")) / f"{cfg['name']}_autopilot.json"
        self.state = self._load()
        self.source = source
        self.notifier = notifier or Notifier(lv.get("webhook"), lv.get("webhook_kind", "generic"))

    # ------------------------------------------------------------------ 状态
    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        s = self.cfg["strategy"]
        return {"active": {"name": s["name"], "params": s.get("params") or {}},
                "last_reopt": None, "last_switch": None, "history": []}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, default=_json), encoding="utf-8")
        tmp.replace(self.state_path)

    def active_cfg(self) -> dict:
        """把当前生效的策略写回配置，交给 LiveRunner 使用。"""
        cfg = copy.deepcopy(self.cfg)
        cfg["strategy"] = {"name": self.state["active"]["name"], "params": self.state["active"]["params"]}
        return cfg

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=self.tz).tz_localize(None)

    def candidates(self) -> list[dict]:
        c = self.ap["candidates"]
        if c:
            return [{"name": x["name"], "params": x.get("params") or {}, "grid": x.get("grid") or {}} for x in c]
        s = self.cfg["strategy"]
        return [{"name": s["name"], "params": s.get("params") or {}, "grid": self.cfg["optimize"].get("grid") or {}}]

    # ------------------------------------------------------------------ 重新优化
    def _warmup(self) -> int:
        w = [get_strategy(c["name"], **{**c["params"], **p}).warmup()
             for c in self.candidates() for p in expand_grid(c["grid"])]
        a = self.state["active"]
        w.append(get_strategy(a["name"], **a["params"]).warmup())
        return max(w)

    def _panel(self, now: pd.Timestamp, bars: int):
        d = self.cfg["data"]
        src = self.source or make_source(data_cfg(self.cfg))
        start = warmup_start(now.normalize(), bars, d.get("freq", "1d"))
        return load_panel(src, resolve_symbols(d), start, None, d.get("freq", "1d"))

    def _holdout_score(self, name: str, params: dict, panel, split: int, warm: int, objective: str) -> dict:
        strat = get_strategy(name, **params)
        sub = panel.iloc(slice(max(0, split - warm), len(panel)))
        r = run_backtest(strat, sub, self.rules, self.cfg["initial_cash"], self.cfg.get("risk"),
                         start=panel.index[split])
        m = r.metrics()
        score = m.get(objective, np.nan)
        return {"score": float(score) if np.isfinite(score) else -np.inf, "trades": m["trades"],
                "fills": int(len(r.fills)),
                "return": m["total_return"], "max_drawdown": m["max_drawdown"]}

    def decide(self, panel, active: dict, now: pd.Timestamp, last_switch: str | None = None) -> dict:
        """只用 panel（截至 now 的数据）做一次优化决策，不修改状态。"""
        ap = self.ap
        objective = ap["objective"] or self.cfg["optimize"].get("objective", "sharpe")
        warm = self._warmup()
        T = len(panel)
        hold = ap["holdout_bars"]
        if T < hold + warm + 60:
            raise RuntimeError(f"数据只有 {T} 根，不足以做训练 + 验证（需要 > {hold + warm + 60}）")
        split = T - hold
        tr_start = max(warm, split - ap["train_bars"])
        train_panel = panel.iloc(slice(tr_start - warm, split))

        rows = []
        for c in self.candidates():
            if c["grid"]:
                res = grid_search(c["name"], train_panel, self.rules, c["grid"], c["params"], self.cfg.get("risk"),
                                  self.cfg["initial_cash"], objective, panel.index[tr_start],
                                  self.cfg["optimize"].get("min_trades", 0), ap["jobs"])
                ok = res[np.isfinite(res["smooth"])]
                pick = ok.loc[ok["smooth"].idxmax()] if len(ok) else res.iloc[0]
                chosen = {k: next(v for v in c["grid"][k] if v == pick[k]) for k in c["grid"]}
                train_score = float(pick["score"])
            else:
                chosen, train_score = {}, np.nan
            params = {**c["params"], **chosen}
            h = self._holdout_score(c["name"], params, panel, split, warm, objective)
            rows.append({"name": c["name"], "params": params, "train_score": train_score, **h})

        cur = self._holdout_score(active["name"], active["params"], panel, split, warm, objective)
        # 按成交笔数判断活跃度（风险平价等持续持仓的策略完整买卖回合很少，但会定期调仓）
        valid = [r for r in rows if r["fills"] >= ap["min_trades"] and np.isfinite(r["score"])]
        best = max(valid, key=lambda r: r["score"]) if valid else None
        same = best is not None and best["name"] == active["name"] and best["params"] == active["params"]
        cooling = last_switch and (now - pd.Timestamp(last_switch)).days < ap["cooldown_days"]
        switch = False
        if best is None:
            reason = "没有满足最少成交笔数的候选方案"
        elif same:
            reason = "当前方案仍是最优"
        elif best["score"] < cur["score"] + ap["min_improvement"]:
            reason = f"验证集提升 {best['score'] - cur['score']:.3f} 未达到门槛 {ap['min_improvement']}"
        elif cooling:
            reason = f"距上次切换不足 {ap['cooldown_days']} 天"
        else:
            switch = True
            reason = f"验证集 {objective} {cur['score']:.3f} -> {best['score']:.3f}"
        return {"time": now.isoformat(), "objective": objective,
                "holdout": [str(panel.index[split].date()), str(panel.index[-1].date())],
                "current": {**active, **cur}, "candidates": rows, "switched": switch, "reason": reason,
                "new": {"name": best["name"], "params": best["params"]} if switch else None}

    def reoptimize(self, now: pd.Timestamp | None = None) -> dict:
        now = now or self.now()
        ap = self.ap
        panel = self._panel(now, ap["train_bars"] + ap["holdout_bars"] + self._warmup())
        act = self.state["active"]
        decision = self.decide(panel, act, now, self.state.get("last_switch"))
        if decision["switched"]:
            self.state["active"] = decision["new"]
            self.state["last_switch"] = now.isoformat()
        self.state["last_reopt"] = now.isoformat()
        self.state["history"] = (self.state["history"] + [decision])[-100:]
        self._save()

        obj, cur = decision["objective"], decision["current"]
        lines = [f"[{self.cfg['name']}] 自动优化（验证集 {decision['holdout'][0]} ~ {decision['holdout'][1]}）",
                 f"当前 {act['name']} {act['params']}：{obj}={cur['score']:.3f}"]
        lines += [f"候选 {r['name']} {r['params']}：{obj}={r['score']:.3f}（训练集 {r['train_score']:.3f}，"
                  f"成交 {r['fills']} 笔）" for r in decision["candidates"]]
        lines.append(("已切换：" if decision["switched"] else "保持不变：") + decision["reason"])
        self.notifier.send("\n".join(lines))
        return decision

    def simulate(self, panel, start, every: int = 21):
        """历史回放整个自动驾驶流程：每 every 根 K 线只用当时之前的数据做一次决策，
        并用决策结果生成下一段的目标仓位；所有段拼成一条目标序列后只跑一次撮合，
        持仓在段与段之间连续（不会每段清仓重建）。返回 (回测结果, 决策记录)。
        用来回答"自动优化到底有没有用"——与固定参数回测对比即可。"""
        from .backtest import BacktestEngine
        from .backtest.runner import ENGINE_KEYS
        from .pipeline import build_targets

        warm = self._warmup()
        need = self.ap["train_bars"] + self.ap["holdout_bars"] + warm
        idx = panel.index
        t0 = t = max(int(idx.searchsorted(pd.Timestamp(start))) if start is not None else 0, need)
        s = self.cfg["strategy"]
        active = {"name": s["name"], "params": s.get("params") or {}}
        last_switch, pieces, log_rows = None, [], []
        risk = self.cfg.get("risk") or {}
        while t < len(idx) - 1:
            now = idx[t - 1]
            d = self.decide(panel.iloc(slice(0, t)), active, now, last_switch)
            if d["switched"]:
                active, last_switch = d["new"], now.isoformat()
            end = min(t + every, len(idx))
            # 只用截至本段结束的数据计算目标（策略本身无未来函数，结果与实时逐日计算一致）
            tg = build_targets(get_strategy(active["name"], **active["params"]),
                               panel.iloc(slice(0, end)), self.rules, risk)
            seg = tg.iloc[t - 1:end].copy()  # 含前一根：决策当天收盘的目标在本段第一根开盘执行
            seg.iloc[0] = tg.ffill().iloc[t - 1]  # 切换/延续时立即采用新方案的当前目标
            pieces.append(seg)
            log_rows.append({"date": idx[t].date(), "strategy": active["name"], "params": active["params"],
                             "switched": d["switched"], "reason": d["reason"]})
            t = end
        targets = pd.concat(pieces)
        targets = targets[~targets.index.duplicated(keep="last")]
        engine = BacktestEngine(self.rules, self.cfg["initial_cash"],
                                **{k: risk[k] for k in ENGINE_KEYS if risk.get(k) is not None})
        sub = panel.iloc(slice(t0 - 1, len(idx)))
        result = engine.run(sub, targets.reindex(sub.index))
        return result, pd.DataFrame(log_rows)

    def status(self) -> dict:
        return self.state

    def due(self, now: pd.Timestamp) -> bool:
        """启动时从未优化过、或距上次优化超过 35 天（错过了定时任务）则需要优化。"""
        last = self.state.get("last_reopt")
        return last is None or (now - pd.Timestamp(last)).days > 35

    # ------------------------------------------------------------------ 交易与健康检查
    def trade(self, force: bool = False, source: DataSource | None = None) -> dict:
        runner = LiveRunner(self.active_cfg(), source=source)
        summary = runner.run_once(force=force)
        self.health_check(runner)
        return summary

    def health_check(self, runner: LiveRunner) -> str | None:
        eq = [v for _, v in runner.state.get("equity_log", [])]
        if len(eq) < 2:
            return None
        s = pd.Series(eq, dtype=float)
        dd = float((s / s.cummax() - 1).min())
        limit = self.ap["max_live_drawdown"]
        if limit and dd <= -limit:
            kill = Path(self.cfg["live"].get("kill_switch_file") or "./live_state/STOP")
            if not kill.exists():
                kill.parent.mkdir(parents=True, exist_ok=True)
                kill.write_text(f"autopilot: 运行以来回撤 {dd:.2%} 超过 {limit:.0%}\n", encoding="utf-8")
                self.notifier.send(f"[{self.cfg['name']}] 回撤 {dd:.2%} 超过上限 {limit:.0%}，已熔断（只减仓）。"
                                   f"检查策略后删除 {kill} 恢复。")
            return f"回撤 {dd:.2%}"
        return None

    def run_forever(self) -> None:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger

        cron = self.cfg["live"].get("cron")
        if not cron:
            raise ValueError("live.cron 未配置，例如 '35 9 * * mon-fri'")
        if self.due(self.now()):
            self._safe(self.reoptimize, "自动优化")
        sched = BlockingScheduler(timezone=self.tz)
        sched.add_job(lambda: self._safe(self.trade, "交易"), CronTrigger.from_crontab(cron, timezone=self.tz),
                      max_instances=1, coalesce=True, misfire_grace_time=300)
        sched.add_job(lambda: self._safe(self.reoptimize, "自动优化"),
                      CronTrigger.from_crontab(self.ap["reopt_cron"], timezone=self.tz),
                      max_instances=1, coalesce=True, misfire_grace_time=3600)
        log.info("自动驾驶已启动：交易 '%s'，优化 '%s'（%s），Ctrl+C 退出", cron, self.ap["reopt_cron"], self.tz)
        try:
            sched.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("已停止")

    def _safe(self, fn, what: str):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 定时任务异常不能让进程退出
            log.exception("%s失败", what)
            self.notifier.send(f"[{self.cfg['name']}] {what}失败: {e}")


def _json(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return str(x)
