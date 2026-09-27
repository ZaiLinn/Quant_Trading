# Quant Trading

一个精简、可读、可落地的 Python 量化交易框架：数据获取 → 策略研究 → 因子检验 → 回测 → 参数优化与防过拟合 → 模拟盘 / 实盘，一套代码贯穿始终。

支持 **A 股**（股票 / ETF / 指数，akshare 免费数据）、**美股 / 港股**（免费日线 + Interactive Brokers 数据与实盘）与 **加密货币**（ccxt，100+ 交易所）。核心代码约 4000 行（含报告模板），无重型依赖（不需要 TA-Lib、scipy、数据库）。

## 特性

- **数据层**：akshare（A 股东方财富优先，限流时自动切换腾讯 / 新浪，成交量统一为"股"；美股 / 港股日线）、IBKR 历史数据（含分钟线）、ccxt、本地 CSV、合成数据；Parquet 缓存；多标的自动对齐（停牌为 NaN）；指数成分股股票池（沪深 300 / 中证 500 / 上证 50 等）；数据质量检查（异常跳变、OHLC 不一致、长期不变价）。
- **回测引擎**：逐 K 线事件驱动，t 收盘出信号、t+1 开盘成交，从机制上杜绝未来函数；完整 A 股规则（100 股一手、T+1、按板块区分的涨跌停、印花税、过户费、最低 5 元佣金）；美股按股收佣（IBKR 费率）、港股逐只每手股数与双边印花税；停牌 / 涨停买不进的订单自动顺延；成交量参与率上限（大单拆到多根 K 线执行，估算策略容量）；盘中止损、止盈、移动止损（跳空按开盘价成交）；做空与杠杆（永续合约）；逐笔成交与完整交易回合记账。
- **策略库（12 个）**：双均线、MACD、RSI 回归、布林回归、KDJ、海龟通道突破（ATR 定仓）、ETF 动量轮动、风险平价（等风险贡献）、网格交易（带滞回）、多因子 TopK 选股、买入持有，以及多策略组合 `blend`（按资金比例同账户运行多个策略）。
- **因子研究**：12 个内置因子；RankIC / ICIR / t 值、分层收益、多空组合、换手与自相关；多因子 z-score 合成。
- **组合风控**：波动率目标、单标的上限、总杠杆上限、大盘择时（均线过滤）、权重步长（过滤无效换手）、目标漂移再平衡。
- **防过拟合**：网格搜索（多进程）、参数平台得分（邻域均值减标准差）与热力图、Walk-Forward 滚动前向验证（可按平台得分选参）、Deflated Sharpe Ratio、分块自助法蒙特卡洛、未来函数截断检测。
- **报告**：自包含 HTML（离线可看、支持深色模式与悬停读数）：权益 / 超额收益 / 回撤 / 仓位曲线、月度收益热力表、分标的贡献、交易明细。
- **模拟盘 / 实盘**：本地模拟盘、IBKR（美股 / 港股）、ccxt（加密货币）；与回测共用目标权重与下单逻辑；只用已收盘 K 线计算信号；A 股用新浪实时行情作为下单参考价；幂等（重复运行不重复下单）；股票池变动后自动清掉池外持仓；熔断文件、单日亏损上限、单笔金额上限；APScheduler 定时；日志滚动写入文件；Webhook 通知（飞书 / 钉钉 / Slack）；ccxt 实盘默认 dry-run，API Key 只从环境变量读取。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

quant strategies                                         # 列出策略
quant backtest -c configs/demo_synthetic.yaml            # 离线演示（合成数据）
quant backtest -c configs/ashare_etf_rotation.yaml       # A 股 ETF 动量轮动
quant compare  -c configs/ashare_etf_rotation.yaml --strategies momentum_rotation,risk_parity,buy_and_hold
quant backtest -c configs/crypto_trend.yaml --mc 1000    # 加密货币海龟 + 蒙特卡洛稳健性
```

回测结果保存在 `runs/<名称>_<类型>_<时间>/`：`report.html`、`metrics.json`、`equity.csv`、`trades.csv`、`fills.csv`、`weights.csv`。

## 命令一览

| 命令 | 作用 |
|---|---|
| `quant strategies` | 列出内置策略与默认参数 |
| `quant download -c 配置` | 下载并缓存行情 |
| `quant backtest -c 配置 [--mc N] [--no-save]` | 回测，生成 HTML 报告；`--mc` 做分块自助法模拟 |
| `quant compare -c 配置 --strategies a,b,c` | 同一数据上对比多个策略 |
| `quant optimize -c 配置 [--jobs -1]` | 网格搜索参数，输出 Deflated Sharpe 与二维参数热力图 |
| `quant walkforward -c 配置` | 滚动前向验证，拼接样本外曲线，给出样本外效率 |
| `quant check -c 配置` | 数据质量检查 + 未来函数截断检测 |
| `quant factor -c 配置` | 单因子检验（IC、分层收益），每个因子一份 HTML |
| `quant live -c 配置 [--once] [--force]` | 模拟盘 / 实盘：`--once` 运行一次，否则按 cron 定时 |
| `quant status -c 配置 [--report]` | 查看模拟盘持仓、成交、权益记录，`--report` 生成 HTML |

任意配置项都可以在命令行覆盖：`--set strategy.params.fast=5 --set data.start=2020-01-01`。

## 配置文件

```yaml
name: etf_rotation
market: ashare_etf            # generic | ashare | ashare_etf | crypto | crypto_perp
market_overrides:             # 按自己的实际费率覆盖
  commission: 0.0001
  symbol_limits: {"159915": 0.2}   # 个别标的涨跌停幅度（如创业板 ETF 为 20%）
initial_cash: 200000
benchmark: "510300"           # 或 {symbol: "000300", asset_type: index}
data:
  source: akshare             # akshare | ccxt | csv | synthetic
  asset_type: etf             # auto | stock | etf | index
  adjust: qfq
  symbols: ["510300", "510500", "159915", "518880"]
  # universe: csi300          # 或直接用指数成分股作为股票池
  start: "2016-01-01"         # 开始交易日期，指标预热数据会自动向前多取
  freq: 1d                    # 1d / 1w / 1h / 15m ...
strategy:
  name: momentum_rotation
  params: {lookback: 20, top_k: 2, rebalance: W}
risk:
  stop_loss: 0.1              # 盘中止损（相对持仓成本）
  trailing_stop: 0.15         # 移动止损（相对持仓以来最高价）
  take_profit: null
  max_weight: 0.5             # 单标的权重上限
  vol_target: 0.15            # 组合年化波动率目标
  weight_step: 0.05           # 目标权重按 5% 取整，过滤小幅调仓
  regime_ma: 200              # 大盘择时：股票池等权指数跌破 200 日均线时降仓
  regime_scale: 0.0           # 降仓后的仓位比例（0 = 空仓）
  drift_threshold: null       # 实际权重偏离目标超过该值时再平衡
  min_order_value: 0
  max_volume_pct: 0.05        # 单根 K 线最多成交上一根成交量的 5%，超出部分顺延
optimize: {objective: sharpe, min_trades: 10, grid: {lookback: [10, 20, 60], top_k: [1, 2]}}
walkforward: {train: 756, test: 252, select: smooth}   # smooth = 按参数平台得分选参
live: {broker: paper, cron: "35 9 * * mon-fri", max_daily_loss: 0.05}
```

完整默认值见 `quant/config.py`，示例见 `configs/`：

| 配置 | 内容 |
|---|---|
| `demo_synthetic.yaml` | 离线演示：合成数据双均线 |
| `ashare_etf_rotation.yaml` | 6 只 ETF 周度动量轮动 |
| `ashare_etf_allweather.yaml` | ETF 风险平价（类全天候） |
| `ashare_etf_blend.yaml` | 多策略组合：50% 动量轮动 + 50% 风险平价 |
| `ashare_stock_trend.yaml` | 个股双均线 + 移动止损，基准沪深 300 |
| `ashare_factor_csi300.yaml` | 沪深 300 多因子选股与因子检验 |
| `us_etf_rotation.yaml` | 美股 ETF 动量轮动，IBKR 实盘配置（默认 dry-run） |
| `hk_stock_trend.yaml` | 港股趋势，逐只每手股数，IBKR 实盘配置 |
| `crypto_trend.yaml` | 币安现货海龟突破 + 波动率目标（含实盘配置） |
| `crypto_grid.yaml` | BTC 小时线网格 |

## 写自己的策略

策略只需实现 `generate(panel)`，返回"目标权重"表（行 = 时间，列 = 标的）：权重 = 目标市值 / 权益，负数为做空，`NaN` 表示保持不动。成交、费用、手数、T+1、涨跌停由引擎统一处理，同一份代码直接用于实盘。

```python
from quant import indicators as ind
from quant.strategy import Strategy, register, hold_until, normalize_weights

@register
class VolumeBreakout(Strategy):
    name = "volume_breakout"
    params = {"n": 20, "k": 1.5, "exit": 10}

    def generate(self, panel):
        c, v = panel.close, panel.volume
        high_n = c.rolling(self.n).max().shift(1)          # 不含当根
        entries = (c > high_n) & (v > self.k * ind.sma(v, 20))
        exits = c < ind.sma(c, self.exit)
        return normalize_weights(hold_until(entries, exits))
```

完整可运行示例：`python examples/custom_strategy.py`。写完后务必跑 `check_lookahead(strategy, panel)`（或 `quant check`），它会把 `shift(-1)`、居中窗口、全样本标准化等未来函数抓出来。

常用工具：`hold_until(entries, exits)` 把进出场信号变成持仓状态；`normalize_weights(signal)` 等权归一；`on_rebalance(weights, "W")` 只在每周首个交易日调仓。

## 因子研究

```bash
quant factor -c configs/ashare_factor_csi300.yaml                  # 沪深 300，首次下载约数分钟
quant factor -c configs/ashare_factor_csi300.yaml --set data.universe=sse50
```

前瞻收益按"t+1 开盘买入、t+1+h 开盘卖出"计算，与回测成交时点一致。输出每个因子的 RankIC 均值、年化 ICIR、t 值（持有期重叠已折算）、分层单调性、多空年化、头部换手。经验上 |年化 ICIR| > 0.5、|t| > 3、分层单调的因子才值得进入组合；组合用 `factor_topk` 策略：

```yaml
strategy:
  name: factor_topk
  params:
    factors: {momentum_120_20: 1.0, volatility_20: -1.0}   # 负权重 = 因子值越小越好
    top_k: 20
    buffer: 10          # 已持有标的排名仍在前 30 则不换，降低换手
    rebalance: W
```

## 防过拟合流程

1. `quant optimize`：看参数热力图与输出的"平台区中心"（邻域得分均值减标准差最高的参数），而不是孤立的最高点；关注 Deflated Sharpe（试参次数越多，要求越高）。
2. `quant walkforward`：每个窗口只用训练段选参、在紧随其后的样本外段检验；样本外效率（OOS/IS）低于 0.5 通常意味着明显过拟合。`walkforward.select: smooth` 按平台得分选参，通常更稳。
3. `quant backtest --mc 1000`：分块自助法看年化收益与回撤的分布区间，以及亏损概率。
4. `quant check`：确认没有未来函数。
5. 先模拟盘运行一段时间，对比实际成交与回测。

## 模拟盘与实盘

```bash
quant live -c configs/ashare_etf_rotation.yaml --once     # 运行一次（非交易日自动跳过，--force 忽略）
quant live -c configs/ashare_etf_rotation.yaml            # 按 live.cron 定时运行
quant status -c configs/ashare_etf_rotation.yaml
```

- **时点**：回测是"t 收盘出信号、t+1 开盘成交"，因此 A 股建议开盘后运行（如 9:35），加密货币在日线收盘后运行（如 UTC 00:05）。运行器会剔除未收盘的当前 K 线，只用它的最新价作为下单参考价；A 股优先使用新浪实时行情的最新价与昨收（涨跌停判断）。
- **日志**：`live_state/<名称>.log`（10MB × 5 份滚动）。
- **状态**：保存在 `live_state/`（原子写入）。目标权重未变化时不交易，重复运行安全；停牌 / 涨跌停 / T+1 / 下单失败 / 被风控截断的订单下次继续尝试。
- **风控**：创建 `live_state/STOP` 文件即熔断（只允许减仓）；当日亏损超过 `max_daily_loss` 暂停开仓；`max_order_value` 限制单笔金额。
- **加密货币实盘**：`live.broker: ccxt`。API Key 通过环境变量提供：`QUANT_BINANCE_API_KEY`、`QUANT_BINANCE_SECRET`（其他交易所同理，部分需要 `QUANT_<EXCHANGE>_PASSWORD`）。默认 `dry_run: true` 只打印不下单，可先用 `sandbox: true` 在测试网验证；确认后再改为 `dry_run: false`。
- **IBKR（美股 / 港股）**：见下一节。
- **A 股实盘**：内置模拟盘。接入券商只需继承 `quant.live.Broker` 实现 `positions / cash / sellable / execute` 四个方法，建议基于 QMT（xtquant）、掘金等官方量化接口；不建议使用模拟点击交易客户端的方案。
- **通知**：设置 `live.webhook` 或环境变量 `QUANT_WEBHOOK_URL`，`live.webhook_kind` 取 `feishu` / `dingtalk` / `slack` / `generic`。

## Interactive Brokers（IBKR）

1. 安装依赖：`pip install ib_async`（已在 requirements.txt 中）。
2. 启动 TWS 或 IB Gateway，登录**模拟账户**；在 Configure → API → Settings 中勾选 "Enable ActiveX and Socket Clients"，记下端口（TWS 模拟 7497 / 实盘 7496；Gateway 模拟 4002 / 实盘 4001）。
3. 配置（以 `configs/us_etf_rotation.yaml` 为例）：

```yaml
market: us                  # us | hk
ibkr:
  port: 7497
  client_id: 17
  currency: USD             # 港股用 HKD
  order_type: ADAPTIVE      # MKT | ADAPTIVE（IB 自适应算法单）
live:
  broker: ibkr
  dry_run: true             # 只读连接、只打印订单；确认无误后再改为 false
  capital: 20000            # 只用账户中的一部分资金跑本策略
  cron: "35 9 * * mon-fri"  # 交易所当地时间（美股 America/New_York、港股 Asia/Hong_Kong）
```

4. 回测用免费日线（`data.source: akshare`，`asset_type: us / hk`）；有 IBKR 行情权限时可改为 `data.source: ibkr`（日线为 IB 复权数据，支持 1m–1h 分钟线，自动分段下载）。
5. `quant live -c configs/us_etf_rotation.yaml --once` 运行一次，观察日志与 dry-run 订单；稳定后再定时运行、关闭 dry-run。

标的写法：`AAPL`（默认 SMART / USD）、`00700`（currency 为 HKD 时自动转为 SEHK 的 `700`）、`SYMBOL:EXCHANGE:CURRENCY[:PRIMARY]` 可逐个指定。港股每手股数各不相同，写在 `market_overrides.symbol_lots` 中。

安全设计：只管理配置中的标的，账户里手动持有的其他股票不会被卖出；股票池变化时只清掉本策略曾经持有的标的；开市判断基于合约交易时段，自动跳过节假日；未完全成交的订单撤单后下次继续。

## 架构

```
quant/
├── market.py          市场规则与费用（回测 / 模拟盘 / 实盘共用）
├── execution.py       目标权重 -> 下单数量（手数、T+1、涨跌停、资金约束），全项目唯一一份
├── pipeline.py        策略输出 -> 风控调整后的目标权重（波动率目标、上限、步长）
├── indicators.py      技术指标（纯 pandas）
├── data/              数据源、缓存、多标的对齐面板
├── strategy/          策略基类、内置策略、未来函数检测
├── factor/            因子库、因子检验、多因子选股策略
├── backtest/          事件驱动回测引擎
├── analysis/          绩效指标、HTML 报告、稳健性检验
├── optimize.py        网格搜索、Walk-Forward
├── live/              Broker（模拟盘 / ccxt）、风控、通知、运行器
├── ibkr.py            Interactive Brokers 数据源与 Broker
├── config.py / app.py 配置加载与装配
└── cli.py             命令行
```

数据流：`DataSource -> Panel -> Strategy.generate -> apply_risk -> 目标权重 -> plan_orders -> 引擎 / Broker`。回测与实盘只在最后一步不同。

## 借鉴与取舍

参考了主流开源量化项目（见 [awesome-quant](https://github.com/wilsonfreitas/awesome-quant)、[thuquant/awesome-quant](https://github.com/thuquant/awesome-quant)），取其精华、去其糟粕：

| 项目 | 借鉴 | 舍弃 |
|---|---|---|
| backtrader | 事件驱动、下一根开盘成交、分析器思想 | 元类与 lines 魔法、难以调试的内部状态 |
| vectorbt / backtesting.py | 向量化信号生成、参数网格与热力图 | 向量化撮合（难以正确处理 T+1、涨跌停、资金约束）、"快引擎 + 慢引擎"两套结果不一致 |
| zipline-reloaded | `order_target_percent` 式目标仓位接口、基准对比 | 繁重的数据 bundle 与安装 |
| qlib / alphalens | IC / ICIR / 分层收益、截面标准化、TopkDropout 降换手 | 重量级数据格式与机器学习流水线 |
| vn.py / hikyuu | 统一 Broker 网关、事前风控、策略部件化（信号 / 资金管理 / 止损分离） | GUI、C++ 依赖 |
| freqtrade | 配置驱动、dry-run、剔除未收盘 K 线、Key 不进配置、熔断保护 | 庞大的插件体系 |
| rqalpha / QUANTAXIS | A 股规则：T+1、涨跌停、印花税、最低佣金、停牌 | 与特定平台耦合的数据与账户体系 |
| pyfolio / quantstats / empyrical | 绩效指标与报告版式 | 已停更依赖，改为自实现 |
| López de Prado | Deflated Sharpe、Walk-Forward、分块自助法 | — |

另外修正了开源代码中常见的几个坑：复权数据增量缓存会混入不同复权基准（改为过期整段重拉）；"周期末调仓"隐含下一根日期信息（改为周期首根）；网格线来回穿越时原价反复买卖（加一格滞回）；新浪接口多线程并发会导致 V8 崩溃（加锁串行）；东财成交量单位是"手"、新浪 / 腾讯是"股"（统一换算）；新浪美股"前复权"对分红用累计减法，长历史下早期价格被大幅压低、收益率失真（改为用原始价 + 因子表重建乘法复权）；不复权 ETF 数据把份额拆分当成暴跌（优先腾讯复权数据，并用数据质量检查兜底）。

## 注意事项

- 回测不等于未来收益。使用指数"当前"成分股回测历史存在幸存者偏差；免费数据可能有错漏，回测前先跑 `quant check` 看数据质量；东财与腾讯都不可用时 ETF 会退回新浪不复权数据（份额拆分会表现为"暴跌"，`quant check` 能发现）。
- 默认费率只是常见值，请按自己的券商 / 交易所费率在 `market_overrides` 中覆盖。印花税 2023-08-28 起为 0.05%，更早期间的成本会略被低估。
- 日线回测无法刻画盘中成交细节；止损按最高 / 最低价判断，同一根 K 线内的先后顺序未知。
- 实盘有真实资金风险。先模拟盘，再小资金，确认日志与回测一致后再逐步放大。本项目不构成投资建议。

## 测试

```bash
pytest            # 70+ 个测试：引擎记账与费用、A 股规则、未来函数、指标、因子、优化、模拟盘
make test         # 同上；另有 make demo / backtest / optimize / factor / paper / status（CFG=配置文件）
```

服务器部署模拟盘 / 实盘可用 `Dockerfile`（见文件头注释）；`.github/workflows/tests.yml` 为 GitHub Actions 测试流程。
