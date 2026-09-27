"""稳健性检验：防止把运气当能力。

- deflated_sharpe：多次试参后夏普比率的显著性（Bailey & López de Prado, 2014）；
- bootstrap：分块自助法重采样收益，估计收益/回撤的分布区间；
- grid_heatmap_html：二维参数平面热力图，用于寻找"平台区"而不是孤立尖峰。
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
EULER = 0.5772156649


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR：真实（非年化）夏普大于 sr_benchmark 的概率，考虑了偏度与峰度。"""
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std()
    skew = r.skew()
    kurt = r.kurt() + 3  # pandas 给的是超额峰度
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(_N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sharpes: list[float] | None = None) -> float:
    """DSR：在做了 n_trials 次参数尝试后，最优结果仍显著优于 0 的概率。

    trial_sharpes 为各次尝试的（非年化）夏普，用于估计其方差；缺省时用 1/样本数 近似。
    DSR > 0.95 才算比较可信。
    """
    r = returns.dropna()
    if n_trials <= 1:
        return probabilistic_sharpe(r)
    var = np.var(trial_sharpes, ddof=1) if trial_sharpes is not None and len(trial_sharpes) > 1 \
        else 1.0 / max(len(r), 1)
    sd = math.sqrt(max(var, 1e-12))
    sr0 = sd * ((1 - EULER) * _N.inv_cdf(1 - 1 / n_trials) + EULER * _N.inv_cdf(1 - 1 / (n_trials * math.e)))
    return probabilistic_sharpe(r, sr0)


def bootstrap(returns: pd.Series, periods_per_year: float, n: int = 1000, block: int = 20,
              seed: int = 0) -> pd.DataFrame:
    """分块自助法（保留短期自相关）。返回每次模拟的 年化收益 / 最大回撤 / 夏普。"""
    r = returns.dropna().to_numpy()
    T = len(r)
    if T < block * 2:
        raise ValueError("样本太短，无法做分块自助")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    out = np.empty((n, 3))
    for k in range(n):
        starts = rng.integers(0, T - block + 1, n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:T]
        eq = np.cumprod(1 + sample)
        years = T / periods_per_year
        cagr = eq[-1] ** (1 / years) - 1 if eq[-1] > 0 else -1.0
        mdd = (eq / np.maximum.accumulate(eq) - 1).min()
        sd = sample.std()
        out[k] = (cagr, mdd, sample.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan)
    return pd.DataFrame(out, columns=["cagr", "max_drawdown", "sharpe"])


def bootstrap_summary(sim: pd.DataFrame) -> pd.DataFrame:
    return sim.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T.rename(
        columns=lambda q: f"P{int(q * 100)}")


def grid_heatmap_html(res: pd.DataFrame, params: list[str], path: str | Path,
                      objective: str = "score") -> Path:
    """二维参数网格热力表（单色顺序色阶，格内标数值，无需悬停即可读数）。"""
    a, b = params
    tbl = res.pivot_table(index=a, columns=b, values="score", aggfunc="max")
    vals = tbl.replace([np.inf, -np.inf], np.nan)
    data = {"rows": [str(i) for i in tbl.index], "cols": [str(c) for c in tbl.columns],
            "vals": [[None if pd.isna(v) else float(v) for v in row] for row in vals.to_numpy()],
            "a": a, "b": b, "objective": objective}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    doc = _HEAT.replace("__DATA__", payload).replace("__TITLE__", html.escape(f"参数热力图 · {objective}"))
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path


_HEAT = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --border:rgba(11,11,11,.1);
  --lo:#cde2fb; --hi:#104281; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19;
  --ink:#fff; --ink-2:#c3c2b7; --border:rgba(255,255,255,.1); --lo:#184f95; --hi:#cde2fb; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7;
  --border:rgba(255,255,255,.1); --lo:#184f95; --hi:#cde2fb; }
body { margin:0; background:var(--page); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"PingFang SC",sans-serif; }
main { max-width:1000px; margin:0 auto; padding:24px 16px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; overflow-x:auto; }
table { border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }
th { color:var(--ink-2); font-weight:500; padding:4px 8px; }
td { min-width:56px; height:32px; text-align:center; border:2px solid var(--surface); border-radius:4px; }
.note { color:var(--ink-2); font-size:12px; margin-top:8px; }
</style></head><body><main>
<h1 style="font-size:18px" id="t"></h1>
<div class="card"><table id="h"></table></div>
<div class="note">颜色越深得分越高。寻找大片深色的"平台区"，其中心参数更可能在样本外保持稳定。</div>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("payload").textContent);
document.getElementById("t").textContent = `${D.objective}：行 ${D.a} × 列 ${D.b}`;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hex = c => [0, 2, 4].map(i => parseInt(c.replace("#", "").slice(i, i + 2), 16));
function draw() {
  const t = document.getElementById("h"); t.replaceChildren();
  const flat = D.vals.flat().filter(v => v != null);
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const cLo = hex(css("--lo")), cHi = hex(css("--hi"));
  const head = document.createElement("tr");
  const corner = document.createElement("th"); corner.textContent = `${D.a} \\ ${D.b}`; head.append(corner);
  D.cols.forEach(c => { const th = document.createElement("th"); th.textContent = c; head.append(th); });
  t.append(head);
  D.rows.forEach((r, i) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th"); th.textContent = r; tr.append(th);
    D.vals[i].forEach((v, j) => {
      const td = document.createElement("td");
      if (v != null) {
        const k = hi > lo ? (v - lo) / (hi - lo) : 1;
        const rgb = cLo.map((x, n) => Math.round(x + (cHi[n] - x) * k));
        td.style.background = `rgb(${rgb})`;
        const lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
        td.style.color = lum > 0.55 ? "#0b0b0b" : "#ffffff";
        td.textContent = v.toFixed(2);
        td.title = `${D.a}=${r}, ${D.b}=${D.cols[j]}: ${v.toFixed(4)}`;
      } else td.textContent = "-";
      tr.append(td);
    });
    t.append(tr);
  });
}
draw();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
})();
</script></body></html>
"""
