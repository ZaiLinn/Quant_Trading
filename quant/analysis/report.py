"""报告：控制台摘要 + 自包含 HTML（内联 SVG 图表，离线可看，支持深色模式、悬停与键盘读数）。

page() 是通用页面骨架，回测报告与因子报告共用同一套样式和图表组件。
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from .metrics import drawdown, format_metrics, monthly_returns

MAX_POINTS = 2500


def print_summary(metrics: dict, title: str = "") -> str:
    text = tabulate(format_metrics(metrics), headers=[title or "指标", "数值"], tablefmt="simple")
    print(text)
    return text


def clean(x):
    """转成 JSON 友好的 Python 类型；NaN/inf -> None。"""
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d %H:%M") if (x.hour or x.minute) else x.strftime("%Y-%m-%d")
    return x


def downsample_index(index: pd.Index) -> np.ndarray:
    """点数过多时等间隔抽样（保留最后一个点），返回布尔掩码。"""
    n = len(index)
    keep = np.zeros(n, dtype=bool)
    step = max(1, int(np.ceil(n / MAX_POINTS)))
    keep[::step] = True
    if n:
        keep[-1] = True
    return keep


def date_labels(index: pd.DatetimeIndex) -> list[str]:
    intraday = len(index) > 2 and (index[1] - index[0]) < pd.Timedelta(hours=20)
    return [d.strftime("%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d") for d in index]


def series_values(s: pd.Series) -> list:
    return [clean(v) for v in s.to_numpy()]


def page(title: str, body: str, script: str, data: dict, path: str | Path) -> Path:
    payload = json.dumps(data, ensure_ascii=False, default=clean)
    payload = payload.replace("</", "<\\/")  # 防止数据中的字符串提前闭合 <script>
    doc = (_PAGE.replace("__TITLE__", html.escape(title)).replace("__CSS__", _CSS)
           .replace("__BODY__", body).replace("__DATA__", payload)
           .replace("__LIB__", _LIB_JS).replace("__SCRIPT__", script))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path


# ---------------------------------------------------------------------- 回测报告


def build_payload(result, metrics: dict, title: str) -> dict:
    eq = result.equity
    keep = downsample_index(eq.index)
    eq_d = eq[keep]
    bench = result.benchmark.reindex(eq.index).ffill()[keep] if result.benchmark is not None else None
    mr = monthly_returns(eq)
    trades = result.trades
    by_sym = []
    if len(trades):
        g = trades.groupby("symbol")
        contrib = pd.DataFrame({"pnl": g["pnl"].sum(), "n": g.size(),
                                "win": g["pnl"].apply(lambda x: (x > 0).mean()),
                                "fees": g["fees"].sum()}).sort_values("pnl", ascending=False)
        by_sym = [[s, clean(r.pnl), int(r.n), clean(r.win), clean(r.fees)] for s, r in contrib.iterrows()]
    return {
        "title": title,
        "dates": date_labels(eq_d.index),
        "equity": series_values(eq_d),
        "bench": series_values(bench) if bench is not None else None,
        "drawdown": series_values(drawdown(eq)[keep]),
        "excess": series_values((eq / result.benchmark.reindex(eq.index).ffill() - 1)[keep])
        if result.benchmark is not None else None,
        "exposure": series_values(result.weights.abs().sum(axis=1)[keep]),
        "metrics": [[k, v] for k, v in format_metrics(metrics)],
        "tiles": {k: clean(metrics.get(k)) for k in
                  ("cagr", "sharpe", "max_drawdown", "total_return", "bench_return", "trades", "win_rate")},
        "monthly": {"years": [int(y) for y in mr.index],
                    "rows": [[clean(v) for v in row] for row in mr.to_numpy()]},
        "trades": [[clean(v) for v in row] for row in trades.tail(200).iloc[::-1].itertuples(index=False)],
        "trade_cols": list(trades.columns),
        "n_trades_total": int(len(trades)),
        "by_symbol": by_sym,
    }


def html_report(result, metrics: dict, path: str | Path, title: str = "回测报告") -> Path:
    return page(title, _BT_BODY, _BT_SCRIPT, build_payload(result, metrics, title), path)


_BT_BODY = r"""
  <h1 id="title"></h1>
  <div class="sub" id="period"></div>
  <div class="tiles" id="tiles"></div>
  <section class="card">
    <h2>权益曲线</h2>
    <div class="legend" id="eq-legend"></div>
    <div class="chart" id="eq-chart"></div>
  </section>
  <section class="card" id="ex-card">
    <h2>相对基准超额收益（策略净值 / 基准净值 − 1）</h2>
    <div class="chart" id="ex-chart"></div>
  </section>
  <section class="card">
    <h2>策略回撤</h2>
    <div class="chart" id="dd-chart"></div>
  </section>
  <section class="card">
    <h2>仓位（总敞口 / 权益）</h2>
    <div class="chart" id="exp-chart"></div>
  </section>
  <section class="card">
    <h2>月度收益（%）</h2>
    <div class="scroll"><table class="heat" id="heat"></table></div>
    <div class="note">红色为盈利、蓝色为亏损，颜色深浅表示幅度；最后一列为全年收益。</div>
  </section>
  <div class="grid2">
    <section class="card">
      <h2>绩效指标</h2>
      <table id="metrics"></table>
    </section>
    <div>
      <section class="card">
        <h2>分标的贡献</h2>
        <div class="scroll"><table id="bysym"></table></div>
      </section>
      <section class="card">
        <h2 id="trades-title">交易记录</h2>
        <div class="scroll"><table id="trades"></table></div>
      </section>
    </div>
  </div>
"""

_BT_SCRIPT = r"""
document.getElementById("title").textContent = D.title;
document.getElementById("period").textContent = `${D.dates[0]} ~ ${D.dates[D.dates.length - 1]}`;
const T = D.tiles;
const tiles = [["年化收益", pct(T.cagr)], ["夏普比率", T.sharpe == null ? "-" : T.sharpe.toFixed(2)],
  ["最大回撤", pct(T.max_drawdown)], ["总收益", pct(T.total_return)]];
if (T.bench_return != null) tiles.push(["基准总收益", pct(T.bench_return)]);
tiles.push(["交易次数 / 胜率", `${T.trades ?? 0} / ${pct(T.win_rate)}`]);
renderTiles(document.getElementById("tiles"), tiles);

const eqSeries = [{name: "策略", values: D.equity, color: "--s1"}];
if (D.bench) eqSeries.push({name: "基准", values: D.bench, color: "--s2"});
renderLegend(document.getElementById("eq-legend"), eqSeries);
lineChart(document.getElementById("eq-chart"), D.dates, eqSeries, {height: 320});
if (D.excess) {
  lineChart(document.getElementById("ex-chart"), D.dates, [{name: "超额", values: D.excess, color: "--s1"}],
    {height: 180, fmt: v => (v * 100).toFixed(1) + "%", area: true, zeroBase: true});
} else document.getElementById("ex-card").remove();
lineChart(document.getElementById("dd-chart"), D.dates, [{name: "回撤", values: D.drawdown, color: "--s1"}],
  {height: 180, fmt: v => (v * 100).toFixed(1) + "%", area: true, zeroBase: true});
lineChart(document.getElementById("exp-chart"), D.dates, [{name: "敞口", values: D.exposure, color: "--s1"}],
  {height: 140, fmt: v => (v * 100).toFixed(0) + "%", area: true, zeroBase: true});

heatTable(document.getElementById("heat"), D.monthly.years,
  [...Array.from({length: 12}, (_, i) => (i + 1) + "月"), "全年"], D.monthly.rows);

renderTable(document.getElementById("metrics"), null, D.metrics);
renderTable(document.getElementById("bysym"), ["标的", "累计盈亏", "交易次数", "胜率", "费用"],
  D.by_symbol.map(r => [r[0], num(r[1]), r[2], pct(r[3]), num(r[4])]));

const TL = {symbol: "标的", side: "方向", entry_date: "开仓", exit_date: "平仓", entry_price: "开仓均价",
  exit_price: "平仓均价", max_qty: "最大数量", pnl: "盈亏", return: "收益率", bars: "持仓(根)",
  fees: "费用", exit_reason: "离场原因"};
document.getElementById("trades-title").textContent =
  `交易记录（共 ${D.n_trades_total} 笔${D.n_trades_total > D.trades.length ? "，显示最近 " + D.trades.length + " 笔" : ""}）`;
renderTable(document.getElementById("trades"), D.trade_cols.map(c => TL[c] || c),
  D.trades.map(row => row.map((v, i) => {
    const c = D.trade_cols[i];
    if (c === "return") return pct(v);
    if (["entry_price", "exit_price", "pnl", "fees", "max_qty"].includes(c)) return num(v);
    return v ?? "-";
  })));
"""

# ---------------------------------------------------------------------- 公共骨架

_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<main>__BODY__</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("payload").textContent);
__LIB__
__SCRIPT__
})();
</script>
</body>
</html>
"""

_THEME_DARK = """
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --pos: #e66767; --neg: #3987e5; --mid: #383835;
    --q1: #184f95; --q2: #256abf; --q3: #3987e5; --q4: #6da7ec; --q5: #b7d3f6;
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #008300;
    --c7: #9085e9; --c8: #e66767;
"""

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834;
  --pos: #e34948; --neg: #256abf; --mid: #f0efec;
  --q1: #86b6ef; --q2: #5598e7; --q3: #2a78d6; --q4: #1c5cab; --q5: #0d366b;
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100; --c5: #e87ba4; --c6: #008300;
  --c7: #4a3aa7; --c8: #e34948;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {""" + _THEME_DARK + """} }
:root[data-theme="dark"] {""" + _THEME_DARK + """}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "PingFang SC", "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 15px; margin: 0 0 12px; font-weight: 600; }
.sub { color: var(--ink-2); margin-bottom: 20px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px; margin-bottom: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .label { color: var(--ink-2); font-size: 12px; }
.tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
.legend { display: flex; gap: 16px; color: var(--ink-2); font-size: 12px; margin-bottom: 8px; flex-wrap: wrap; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { display: inline-block; width: 16px; height: 2px; border-radius: 1px; }
.chart { position: relative; width: 100%; }
.chart svg { display: block; width: 100%; overflow: visible; }
.chart svg:focus { outline: 2px solid var(--s1); outline-offset: 4px; border-radius: 4px; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12);
  white-space: nowrap; display: none; z-index: 2; }
.tip .d { color: var(--muted); margin-bottom: 4px; }
.tip .r { display: flex; align-items: center; gap: 6px; }
.tip .r i { width: 12px; height: 2px; display: inline-block; }
.tip .r b { font-variant-numeric: tabular-nums; }
.tip .r span { color: var(--ink-2); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { padding: 6px 8px; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 500; border-bottom: 1px solid var(--grid); }
td { border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
.heat td { border: 2px solid var(--surface); border-radius: 4px; text-align: center; min-width: 52px; }
.heat td.y { background: transparent; text-align: left; color: var(--ink-2); }
.grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); gap: 16px; }
@media (max-width: 760px) { .grid2 { grid-template-columns: minmax(0, 1fr); } }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
"""

_LIB_JS = r"""
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = v => v == null ? "-" : (v * 100).toFixed(2) + "%";
const num = v => v == null ? "-" : v.toLocaleString(undefined, {maximumFractionDigits: 2});
const el = (tag, attrs = {}, text) => {
  const e = tag.startsWith("svg:") ? document.createElementNS("http://www.w3.org/2000/svg", tag.slice(4))
                                   : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
};
document.title = D.title || document.title;

function renderTiles(box, tiles) {
  for (const [k, v] of tiles) {
    const t = el("div", {class: "tile"});
    t.append(el("div", {class: "label"}, k), el("div", {class: "value"}, v));
    box.append(t);
  }
}
function renderLegend(box, series) {
  if (series.length < 2) { box.remove(); return; }
  for (const s of series) {
    const sp = el("span"), key = el("i");
    key.style.background = `var(${s.color})`;
    sp.append(key, document.createTextNode(s.name));
    box.append(sp);
  }
}
function renderTable(tbl, head, rows) {
  if (head) { const tr = el("tr"); head.forEach(h => tr.append(el("th", {}, h))); tbl.append(tr); }
  for (const row of rows) { const tr = el("tr"); row.forEach(v => tr.append(el("td", {}, v ?? "-"))); tbl.append(tr); }
}
function niceTicks(lo, hi, n = 5) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= raw);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}
// 折线/面积图：单一 y 轴；十字准线吸附最近日期，提示框列出所有序列；方向键可逐点查看
function lineChart(host, dates, series, {height = 300, fmt = num, area = false, zeroBase = false} = {}) {
  const tip = el("div", {class: "tip"});
  let idx = dates.length - 1;
  function draw() {
    host.querySelectorAll("svg").forEach(s => s.remove());
    const W = host.clientWidth || 800, H = height, m = {l: 64, r: 16, t: 8, b: 26};
    const n = dates.length, iw = W - m.l - m.r, ih = H - m.t - m.b;
    const vals = series.flatMap(s => s.values.filter(v => v != null));
    if (!vals.length) return;
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (zeroBase) { hi = Math.max(hi, 0); lo = Math.min(lo, 0); }
    const ticks = niceTicks(lo, hi);
    lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
    const x = i => m.l + (n <= 1 ? 0 : i / (n - 1) * iw);
    const y = v => m.t + (hi === lo ? ih / 2 : (hi - v) / (hi - lo) * ih);
    const svg = el("svg:svg", {viewBox: `0 0 ${W} ${H}`, height: H, tabindex: 0, role: "img",
      "aria-label": series.map(s => s.name).join("、") + " 走势图，左右方向键查看数值"});
    for (const tv of ticks) {
      svg.append(el("svg:line", {x1: m.l, x2: W - m.r, y1: y(tv), y2: y(tv), style: "stroke: var(--grid)", "stroke-width": 1}));
      svg.append(el("svg:text", {x: m.l - 8, y: y(tv) + 4, "text-anchor": "end", "font-size": 11,
        style: "fill: var(--muted); font-variant-numeric: tabular-nums"}, fmt(tv)));
    }
    svg.append(el("svg:line", {x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, style: "stroke: var(--axis)", "stroke-width": 1}));
    const nx = Math.max(2, Math.floor(iw / 110));
    for (let k = 0; k < nx; k++) {
      const i = Math.round(k / (nx - 1) * (n - 1));
      svg.append(el("svg:text", {x: x(i), y: H - 6, "font-size": 11, style: "fill: var(--muted)",
        "text-anchor": k === 0 ? "start" : k === nx - 1 ? "end" : "middle"}, dates[i].slice(0, 10)));
    }
    for (const s of series) {
      let d = "", pen = false;
      s.values.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        d += (pen ? "L" : "M") + x(i).toFixed(1) + "," + y(v).toFixed(1); pen = true;
      });
      if (area) {
        const base = y(Math.min(Math.max(0, lo), hi));
        svg.append(el("svg:path", {d: `${d}L${x(n - 1)},${base}L${x(0)},${base}Z`, style: `fill: var(${s.color})`, "fill-opacity": 0.1}));
      }
      svg.append(el("svg:path", {d, fill: "none", style: `stroke: var(${s.color})`, "stroke-width": 2,
        "stroke-linejoin": "round", "stroke-linecap": "round"}));
    }
    const cross = el("svg:line", {y1: m.t, y2: m.t + ih, style: "stroke: var(--axis)", "stroke-width": 1, visibility: "hidden"});
    svg.append(cross);
    const dots = series.map(s => {
      const c = el("svg:circle", {r: 4, style: `fill: var(${s.color}); stroke: var(--surface)`, "stroke-width": 2, visibility: "hidden"});
      svg.append(c); return c;
    });
    const hit = el("svg:rect", {x: m.l, y: m.t, width: iw, height: ih, fill: "transparent"});
    svg.append(hit);
    function show(i) {
      idx = Math.max(0, Math.min(n - 1, i));
      cross.setAttribute("x1", x(idx)); cross.setAttribute("x2", x(idx)); cross.setAttribute("visibility", "visible");
      tip.replaceChildren(el("div", {class: "d"}, dates[idx]));
      series.forEach((s, k) => {
        const v = s.values[idx];
        if (v == null) { dots[k].setAttribute("visibility", "hidden"); return; }
        dots[k].setAttribute("cx", x(idx)); dots[k].setAttribute("cy", y(v)); dots[k].setAttribute("visibility", "visible");
        const row = el("div", {class: "r"}), key = el("i");
        key.style.background = `var(${s.color})`;
        row.append(key, el("b", {}, fmt(v)), el("span", {}, s.name));
        tip.append(row);
      });
      tip.style.display = "block";
      const px = x(idx) / W * host.clientWidth;
      tip.style.left = (px + 12 + tip.offsetWidth > host.clientWidth ? px - 12 - tip.offsetWidth : px + 12) + "px";
      tip.style.top = "8px";
    }
    function hide() {
      cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden"));
      tip.style.display = "none";
    }
    hit.addEventListener("pointermove", e => {
      const r = svg.getBoundingClientRect();
      show(Math.round(((e.clientX - r.left) / r.width * W - m.l) / iw * (n - 1)));
    });
    hit.addEventListener("pointerleave", hide);
    svg.addEventListener("focus", () => show(idx));
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", e => {
      const step = Math.max(1, Math.round(n / 100));
      if (e.key === "ArrowLeft") { show(idx - step); e.preventDefault(); }
      if (e.key === "ArrowRight") { show(idx + step); e.preventDefault(); }
    });
    host.append(svg);
  }
  host.append(tip);
  draw();
  new ResizeObserver(() => draw()).observe(host);
}
// 发散色热力表（红=正 / 蓝=负，灰色中点），格内直接标数值（百分比）
function heatTable(tbl, rowNames, colNames, rows, {digits = 1, scale = 100} = {}) {
  const hex = c => { c = c.replace("#", ""); return [0, 2, 4].map(i => parseInt(c.slice(i, i + 2), 16)); };
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  function draw() {
    tbl.replaceChildren();
    const head = el("tr");
    head.append(el("th", {}, ""));
    colNames.forEach(c => head.append(el("th", {}, c)));
    tbl.append(head);
    const all = rows.flat().filter(v => v != null).map(Math.abs).sort((a, b) => a - b);
    const cap = Math.max(all[Math.floor(all.length * 0.9)] || 0, 1e-9);
    const mid = hex(css("--mid")), pos = hex(css("--pos")), neg = hex(css("--neg"));
    rowNames.forEach((rn, r) => {
      const tr = el("tr");
      tr.append(el("td", {class: "y"}, rn));
      rows[r].forEach((v, c) => {
        const td = el("td", {}, v == null ? "" : (v * scale).toFixed(digits));
        if (v != null) {
          const rgb = mix(mid, v >= 0 ? pos : neg, Math.min(Math.abs(v) / cap, 1));
          td.style.background = `rgb(${rgb})`;
          const lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
          td.style.color = lum > 0.55 ? "#0b0b0b" : "#ffffff";
          td.title = `${rn} ${colNames[c]}：${(v * scale).toFixed(digits + 1)}`;
          if (c === colNames.length - 1) td.style.fontWeight = "600";
        }
        tr.append(td);
      });
      tbl.append(tr);
    });
  }
  draw();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
  new MutationObserver(draw).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
}
"""
