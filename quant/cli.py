"""命令行入口：python -m quant <命令> -c 配置文件 [--set key=value ...]"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
from tabulate import tabulate

from . import app
from .config import load_config

log = logging.getLogger("quant")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("urllib3", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_strategies(args, cfg) -> None:
    from .strategy import STRATEGIES

    rows = [(name, cls.description, ", ".join(f"{k}={v}" for k, v in cls.params.items()))
            for name, cls in sorted(STRATEGIES.items())]
    print(tabulate(rows, headers=["策略", "说明", "默认参数"], tablefmt="simple", maxcolwidths=[None, 40, 50]))


def cmd_download(args, cfg) -> None:
    d = cfg["data"]
    src = app.make_source(d)
    for sym in d["symbols"]:
        df = src.fetch(sym, d.get("start"), d.get("end"), d.get("freq", "1d"))
        print(f"{sym}: {len(df)} 行 {df.index.min()} ~ {df.index.max()}" if len(df) else f"{sym}: 无数据")


def cmd_backtest(args, cfg) -> None:
    from .analysis import html_report, print_summary
    from .backtest import run_backtest

    s = app.prepare(cfg)
    print(f"策略 {s.strategy}  市场 {s.rules.name}  {s.panel}")
    res = run_backtest(s.strategy, s.panel, s.rules, cfg["initial_cash"], cfg.get("risk"),
                       s.benchmark, start=s.trade_start)
    m = res.metrics()
    print_summary(m, repr(s.strategy))
    if args.mc:
        _robustness(res, args.mc)
    if args.no_save:
        return
    out = app.run_dir(cfg, "backtest")
    res.equity.to_frame().join(res.cash).to_csv(out / "equity.csv")
    res.fills.to_csv(out / "fills.csv", index=False)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.weights.to_csv(out / "weights.csv")
    app.save_json({"metrics": m, "config": cfg}, out / "metrics.json")
    path = html_report(res, m, out / "report.html", f"{cfg['name']} · {s.strategy}")
    print(f"\n结果已保存: {out}\n报告: {path.resolve()}")


def _robustness(res, n: int) -> None:
    from .analysis.robustness import bootstrap, bootstrap_summary, probabilistic_sharpe
    from .pipeline import bars_per_year

    ppy = bars_per_year(res.equity.index, res.rules)
    sim = bootstrap(res.returns, ppy, n=n)
    print(f"\n分块自助法 {n} 次模拟（收益序列重采样，衡量结果对运气的依赖）：")
    print(tabulate(bootstrap_summary(sim).reset_index().to_dict("records"), headers="keys",
                   tablefmt="simple", floatfmt=".3f"))
    print(f"P(年化收益<0) = {(sim['cagr'] < 0).mean():.1%}   "
          f"PSR(夏普>0 的概率) = {probabilistic_sharpe(res.returns):.1%}")


def cmd_optimize(args, cfg) -> None:
    import numpy as np

    from .analysis.robustness import deflated_sharpe
    from .backtest import run_backtest
    from .optimize import grid_search
    from .pipeline import bars_per_year

    o = cfg["optimize"]
    if not o.get("grid"):
        sys.exit("配置 optimize.grid 为空")
    warm = max(app.make_strategy(cfg, **p).warmup() for p in _grid_items(o["grid"]))
    s = app.prepare(cfg, extra_warmup=warm)
    res = grid_search(cfg["strategy"]["name"], s.panel, s.rules, o["grid"], cfg["strategy"].get("params"),
                      cfg.get("risk"), cfg["initial_cash"], o.get("objective", "sharpe"), s.trade_start,
                      o.get("min_trades", 0), args.jobs or o.get("jobs", 1))
    print(tabulate(res.head(args.top).to_dict("records"), headers="keys", tablefmt="simple", floatfmt=".4f"))
    # 多重检验校正：试了 N 组参数后，最优夏普仍显著的概率
    valid = res[np.isfinite(res["score"])]
    if len(valid):
        best = {k: next(v for v in o["grid"][k] if v == valid.iloc[0][k]) for k in o["grid"]}
        bt = run_backtest(app.make_strategy(cfg, **best), s.panel, s.rules, cfg["initial_cash"],
                          cfg.get("risk"), start=s.trade_start)
        ppy = bars_per_year(bt.equity.index, s.rules)
        dsr = deflated_sharpe(bt.returns, len(valid), list(valid["sharpe"].dropna() / np.sqrt(ppy)))
        print(f"\n最优参数 {best}：Deflated Sharpe = {dsr:.1%}（试参 {len(valid)} 组；>95% 才较可信）")
        center = res.loc[res["smooth"].idxmax()]
        center_p = {k: next(v for v in o["grid"][k] if v == center[k]) for k in o["grid"]}
        print(f"平台区中心 {center_p}：平台得分 {center['smooth']:.4f}，"
              f"自身得分 {center['score']:.4f}（参数多维时更推荐用它）")
    out = app.run_dir(cfg, "optimize")
    res.to_csv(out / "grid.csv", index=False)
    if len(o["grid"]) == 2:
        from .analysis.robustness import grid_heatmap_html

        grid_heatmap_html(res, list(o["grid"]), out / "heatmap.html", o.get("objective", "sharpe"))
    print(f"\n全部结果: {out / 'grid.csv'}")
    print("提示：优先选择'参数邻域整体表现都不错'的平台区，而不是孤立的最高点；再用 walkforward 检验。")


def _grid_items(grid):
    from .optimize import expand_grid

    return expand_grid(grid)


def cmd_compare(args, cfg) -> None:
    """同一数据上对比多个策略（默认参数或配置 compare 列表中的参数）。"""
    from .analysis.report import date_labels, downsample_index, page, series_values
    from .backtest import run_backtest
    from .strategy import get_strategy

    bench = None

    items = cfg.get("compare") or [{"name": n} for n in (args.strategies or "").split(",") if n]
    if not items:
        sys.exit("请用 --strategies a,b,c 或配置 compare: [{name:..., params:{...}}] 指定策略")
    strats = [get_strategy(it["name"], **(it.get("params") or {})) for it in items]
    s = app.prepare(cfg, extra_warmup=max(st.warmup() for st in strats))
    rows, curves = [], []
    for st in strats:
        r = run_backtest(st, s.panel, s.rules, cfg["initial_cash"], cfg.get("risk"), s.benchmark, start=s.trade_start)
        m = r.metrics()
        rows.append({"strategy": repr(st), **{k: m.get(k) for k in
                     ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "trades", "turnover")}})
        curves.append((st.name, r.equity))
        bench = r.benchmark
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(tabulate(df.to_dict("records"), headers="keys", tablefmt="simple", floatfmt=".3f"))
    out = app.run_dir(cfg, "compare")
    df.to_csv(out / "compare.csv", index=False)
    eq = pd.concat({n: e for n, e in curves}, axis=1)
    if bench is not None:
        eq["基准"] = bench.reindex(eq.index)
    keep = downsample_index(eq.index)
    series = [{"name": c, "values": series_values(eq[c][keep]), "color": f"--c{i + 1}"}
              for i, c in enumerate(eq.columns[:8])]
    data = {"title": f"{cfg['name']} · 策略对比", "dates": date_labels(eq.index[keep]), "series": series,
            "table": [[r["strategy"], f"{r['cagr']:.2%}", f"{r['sharpe']:.2f}", f"{r['max_drawdown']:.2%}",
                       f"{r['calmar']:.2f}", int(r["trades"])] for r in df.to_dict("records")]}
    path = page(data["title"], _COMPARE_BODY, _COMPARE_SCRIPT, data, out / "compare.html")
    print(f"\n报告: {path.resolve()}")


_COMPARE_BODY = """
  <h1 id="title"></h1>
  <section class="card"><h2>权益曲线</h2><div class="legend" id="lg"></div><div class="chart" id="eq"></div></section>
  <section class="card"><h2>绩效对比（按夏普排序）</h2><div class="scroll"><table id="tb"></table></div></section>
"""
_COMPARE_SCRIPT = """
document.getElementById("title").textContent = D.title;
renderLegend(document.getElementById("lg"), D.series);
lineChart(document.getElementById("eq"), D.dates, D.series, {height: 360});
renderTable(document.getElementById("tb"), ["策略", "年化", "夏普", "最大回撤", "卡玛", "交易次数"], D.table);
"""


def cmd_walkforward(args, cfg) -> None:
    from .analysis import html_report, print_summary
    from .optimize import walk_forward

    o, w = cfg["optimize"], cfg["walkforward"]
    if not o.get("grid"):
        sys.exit("配置 optimize.grid 为空")
    s = app.prepare(cfg)
    res = walk_forward(cfg["strategy"]["name"], s.panel, s.rules, o["grid"], w["train"], w["test"],
                       cfg["strategy"].get("params"), cfg.get("risk"), cfg["initial_cash"],
                       o.get("objective", "sharpe"), w.get("anchored", False), o.get("min_trades", 0),
                       args.jobs or o.get("jobs", 1), w.get("select", "best"))
    print(tabulate(res.windows, headers="keys", tablefmt="simple", floatfmt=".3f", showindex=False))
    if s.benchmark is not None:
        res.oos.benchmark = (s.benchmark.reindex(res.oos.equity.index.union(s.benchmark.index)).ffill()
                             .reindex(res.oos.equity.index))
        res.oos.benchmark = res.oos.benchmark / res.oos.benchmark.iloc[0] * cfg["initial_cash"]
    m = res.oos.metrics()
    print(f"\n样本外效率（OOS/IS 得分）: {res.efficiency:.2f}  （<0.5 通常意味着明显过拟合）")
    print_summary(m, "样本外拼接")
    out = app.run_dir(cfg, "walkforward")
    res.windows.to_csv(out / "windows.csv", index=False)
    app.save_json({"metrics": m, "efficiency": res.efficiency, "config": cfg}, out / "metrics.json")
    path = html_report(res.oos, m, out / "report.html", f"{cfg['name']} · Walk-Forward 样本外")
    print(f"\n报告: {path.resolve()}")


def cmd_check(args, cfg) -> None:
    from .data.quality import quality_report
    from .strategy import check_lookahead

    s = app.prepare(cfg)
    q = quality_report(s.panel, s.rules)
    print("数据质量：")
    print(tabulate(q.to_dict("records"), headers="keys", tablefmt="simple", floatfmt=".3f"))
    if q["issues"].fillna("").astype(bool).any():
        print("提示：异常跳变多为复权口径问题或数据错误，建议核对后再回测。\n")
    problems = check_lookahead(s.strategy, s.panel, n_checks=args.n)
    if problems:
        print(f"发现未来函数（{len(problems)} 处）：")
        for p in problems:
            print("  ", p)
        sys.exit(1)
    print(f"{s.strategy}: 截断测试 {args.n} 次通过，未发现未来函数")


def cmd_factor(args, cfg) -> None:
    from .factor import FACTORS, evaluate, factor_html
    from .factor.library import factor_warmup
    from .pipeline import bars_per_year

    f = cfg["factor"]
    specs = f.get("specs") or sorted(FACTORS)
    s = app.prepare(cfg, extra_warmup=max(factor_warmup(x) for x in specs))
    ppy = int(bars_per_year(s.panel.index, s.rules))
    print(f"{s.panel}，检验 {len(specs)} 个因子")
    out = app.run_dir(cfg, "factor")
    rows = []
    for spec in specs:
        rep = evaluate(s.panel, spec, tuple(f.get("horizons", (1, 5, 20))), int(f.get("quantiles", 5)),
                       start=s.trade_start, periods_per_year=ppy)
        rows.append(rep.summary(ppy))
        factor_html(rep, out / f"factor_{spec}.html", ppy)
    df = pd.DataFrame(rows).sort_values("icir_ann", key=abs, ascending=False)
    df.to_csv(out / "factors.csv", index=False)
    print(tabulate(df.to_dict("records"), headers="keys", tablefmt="simple", floatfmt=".4f"))
    print(f"\n报告目录: {out.resolve()}")
    print("解读：|年化ICIR|>0.5 且 |t|>3、分组单调性接近 ±1 的因子更可靠；负 IC 因子组合时取负权重。")


def cmd_live(args, cfg) -> None:
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    from .live import LiveRunner

    # 实盘日志同时写文件（10MB × 5 份滚动），便于事后排查
    log_dir = Path(cfg["live"].get("state_dir", "./live_state"))
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(log_dir / f"{cfg['name']}.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    runner = LiveRunner(cfg)
    if args.once:
        print(runner.run_once(force=args.force))
    else:
        runner.run_forever()


def cmd_status(args, cfg) -> None:
    import json
    from pathlib import Path

    d = Path(cfg["live"].get("state_dir", "./live_state"))
    for name in (f"{cfg['name']}_paper.json", f"{cfg['name']}_runner.json"):
        p = d / name
        if not p.exists():
            continue
        st = json.loads(p.read_text(encoding="utf-8"))
        if "cash" in st:
            print(f"现金: {st['cash']:,.2f}")
            print(tabulate([(k, v, st['avg_cost'].get(k)) for k, v in st["positions"].items() if v],
                           headers=["标的", "数量", "成本"], tablefmt="simple"))
            print("最近成交:")
            print(tabulate(st["fills"][-10:], headers="keys", tablefmt="simple"))
        else:
            log_ = st.get("equity_log", [])
            if log_:
                eq = pd.Series({pd.Timestamp(t): v for t, v in log_})
                dd = (eq / eq.cummax() - 1).min()
                print(f"\n权益记录 {len(eq)} 条，最新 {eq.iloc[-1]:,.2f}，最高 {eq.max():,.2f}，最大回撤 {dd:.2%}")
                if args.report and len(eq) > 1:
                    _live_report(cfg, eq, d)
            print(f"目标权重: {st.get('last_targets')}  待成交: {st.get('pending')}")


def _live_report(cfg, eq: pd.Series, state_dir) -> None:
    import json

    from .analysis.report import date_labels, page, series_values

    fills = []
    paper = state_dir / f"{cfg['name']}_paper.json"
    if paper.exists():
        fills = json.loads(paper.read_text(encoding="utf-8"))["fills"][-200:][::-1]
    data = {"title": f"{cfg['name']} · 模拟盘/实盘记录", "dates": date_labels(pd.DatetimeIndex(eq.index)),
            "eq": series_values(eq), "fills": [[f["time"], f["symbol"], f"{f['qty']:g}", f"{f['price']:.4f}",
                                                f"{f['fee']:.2f}"] for f in fills]}
    body = """
  <h1 id="title"></h1>
  <section class="card"><h2>权益</h2><div class="chart" id="eq"></div></section>
  <section class="card"><h2>最近成交</h2><div class="scroll"><table id="fl"></table></div></section>
"""
    script = """
document.getElementById("title").textContent = D.title;
lineChart(document.getElementById("eq"), D.dates, [{name: "权益", values: D.eq, color: "--s1"}], {height: 280});
renderTable(document.getElementById("fl"), ["时间", "标的", "数量", "价格", "费用"], D.fills);
"""
    path = page(data["title"], body, script, data, state_dir / f"{cfg['name']}_report.html")
    print(f"报告: {path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quant", description="量化交易框架")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("-c", "--config", help="YAML 配置文件")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖配置，如 --set strategy.params.fast=5")
        sp.set_defaults(fn=fn)
        return sp

    add("strategies", cmd_strategies, "列出内置策略")
    add("download", cmd_download, "下载并缓存行情")
    sp = add("backtest", cmd_backtest, "回测")
    sp.add_argument("--no-save", action="store_true")
    sp.add_argument("--mc", type=int, default=0, metavar="N", help="分块自助法模拟 N 次，评估稳健性")
    sp = add("compare", cmd_compare, "多策略对比")
    sp.add_argument("--strategies", help="逗号分隔的策略名，使用默认参数")
    sp = add("optimize", cmd_optimize, "网格搜索参数")
    sp.add_argument("--top", type=int, default=15)
    sp.add_argument("--jobs", type=int, default=None, help="并行进程数，-1 为全部 CPU")
    sp = add("walkforward", cmd_walkforward, "滚动前向验证")
    sp.add_argument("--jobs", type=int, default=None)
    add("check", cmd_check, "未来函数检测").add_argument("-n", type=int, default=10)
    add("factor", cmd_factor, "单因子检验（IC / 分层收益）")
    sp = add("live", cmd_live, "模拟盘/实盘")
    sp.add_argument("--once", action="store_true", help="只运行一次")
    sp.add_argument("--force", action="store_true", help="忽略交易日历（调试用）")
    add("status", cmd_status, "查看模拟盘/实盘状态").add_argument("--report", action="store_true",
                                                             help="生成权益与成交 HTML 报告")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config, args.set)
    args.fn(args, cfg)


if __name__ == "__main__":
    main()
