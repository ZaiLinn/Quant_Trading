"""Interactive Brokers 接入（ib_async）：历史行情数据源 + 下单 Broker。

前提：本机运行 TWS 或 IB Gateway，并在其设置中启用 API（Configure → API → Enable ActiveX and Socket Clients）。
默认端口：TWS 模拟账户 7497 / 实盘 7496；IB Gateway 模拟 4002 / 实盘 4001。

标的写法（config 的 data.symbols）：
  "AAPL"                  -> 使用 ibkr.exchange / ibkr.currency 默认值（SMART / USD）
  "00700:SEHK:HKD"        -> 港股（前导 0 会自动去掉，IB 代码为 700）
  "AAPL:SMART:USD:NASDAQ" -> 第 4 段为 primaryExchange，用于消除歧义

安全设计：
- 只管理配置中的标的；账户里其他持仓（手动交易的）不会被动；
- live.capital 限定策略可用资金，避免动用整个账户；
- 默认 dry_run（只读连接、只打印订单）；确认无误后再把 live.dry_run 改为 false。
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
import pandas as pd

from .data.base import DataSource, freq_to_timedelta, normalize, to_timestamp
from .live.broker import Broker
from .market import MarketRules

log = logging.getLogger(__name__)

DEFAULTS = {"host": "127.0.0.1", "port": 7497, "client_id": 17, "exchange": "SMART", "currency": "USD",
            "account": "", "timeout": 30, "order_type": "MKT", "fill_timeout": 60}
_BAR = {"1m": "1 min", "5m": "5 mins", "15m": "15 mins", "30m": "30 mins", "1h": "1 hour",
        "1d": "1 day", "1w": "1 week"}
# 单次请求可覆盖的时长（IB 历史数据限制），分钟线按此分段向前翻页
_CHUNK = {"1m": "1 D", "5m": "1 W", "15m": "1 W", "30m": "1 M", "1h": "1 M"}
_TZ = {"USD": "America/New_York", "HKD": "Asia/Hong_Kong"}
_CONN: dict[tuple, object] = {}


def settings(cfg: dict | None) -> dict:
    return {**DEFAULTS, **(cfg or {})}


def connect(cfg: dict | None, readonly: bool = False):
    """按 (host, port, client_id) 复用连接。"""
    from ib_async import IB

    c = settings(cfg)
    key = (c["host"], int(c["port"]), int(c["client_id"]))
    ib = _CONN.get(key)
    if ib is not None and ib.isConnected():
        return ib
    ib = IB()
    try:
        ib.connect(c["host"], int(c["port"]), clientId=int(c["client_id"]), timeout=c["timeout"],
                   readonly=readonly, account=c["account"])
    except (ConnectionRefusedError, OSError, TimeoutError) as e:
        raise ConnectionError(f"无法连接 IBKR {c['host']}:{c['port']}（{type(e).__name__}）。请确认 TWS/IB Gateway "
                              "已启动、已启用 API、端口正确（TWS 模拟 7497 / Gateway 模拟 4002）") from e
    _CONN[key] = ib
    return ib


def parse_symbol(symbol: str, cfg: dict | None = None) -> dict:
    c = settings(cfg)
    parts = symbol.split(":")
    sym = parts[0]
    exch = parts[1] if len(parts) > 1 and parts[1] else c["exchange"]
    cur = parts[2] if len(parts) > 2 and parts[2] else c["currency"]
    primary = parts[3] if len(parts) > 3 else ""
    if (exch == "SEHK" or cur == "HKD") and sym.isdigit():
        sym = str(int(sym))  # IB 港股代码不带前导 0
        exch = "SEHK" if exch == "SMART" else exch
    return {"symbol": sym, "exchange": exch, "currency": cur, "primaryExchange": primary}


def make_contract(symbol: str, cfg: dict | None = None):
    from ib_async import Stock

    return Stock(**parse_symbol(symbol, cfg))


class IbkrSource(DataSource):
    """IBKR 历史 K 线。日线/周线默认用 ADJUSTED_LAST（已复权，含拆分与分红），分钟线用 TRADES。"""

    name = "ibkr"

    def __init__(self, cfg: dict | None = None, adjust: bool = True, use_rth: bool = True):
        self.cfg = settings(cfg)
        self.adjust = adjust
        self.use_rth = use_rth

    def cache_key(self, symbol: str, freq: str) -> str:
        p = parse_symbol(symbol, self.cfg)
        return f"ibkr_{p['symbol']}_{p['exchange']}_{p['currency']}_{freq}_{'adj' if self.adjust else 'raw'}"

    def fetch(self, symbol: str, start=None, end=None, freq: str = "1d") -> pd.DataFrame:
        from ib_async import util

        if freq not in _BAR:
            raise ValueError(f"IBKR 不支持周期 {freq}，可选 {sorted(_BAR)}")
        ib = connect(self.cfg, readonly=True)
        contract = make_contract(symbol, self.cfg)
        ib.qualifyContracts(contract)
        tz = _TZ.get(parse_symbol(symbol, self.cfg)["currency"], "UTC")
        now = pd.Timestamp.now(tz=tz).tz_localize(None)
        start_ts = to_timestamp(start) or now - pd.Timedelta(days=365 * 5)
        end_ts = to_timestamp(end)
        frames = []
        if freq in _CHUNK:
            cursor = "" if end_ts is None else end_ts.tz_localize(tz).tz_convert("UTC").strftime("%Y%m%d-%H:%M:%S")
            for _ in range(500):  # 分段向前翻页直到覆盖 start
                bars = ib.reqHistoricalData(contract, cursor, _CHUNK[freq], _BAR[freq], "TRADES",
                                            self.use_rth, formatDate=2)
                if not bars:
                    break
                df = util.df(bars)
                frames.append(df)
                first = pd.Timestamp(df["date"].iloc[0])
                if first.tz_convert(tz).tz_localize(None) <= start_ts:
                    break
                cursor = first.tz_convert("UTC").strftime("%Y%m%d-%H:%M:%S")
                ib.sleep(0.5)  # 避免触发 IB 的历史数据频率限制
        else:
            days = (now - start_ts).days + 5
            duration = f"{math.ceil(days / 365)} Y" if days > 365 else f"{max(days, 2)} D"
            # ADJUSTED_LAST 只能取到最新（endDateTime 必须为空），之后再按 end 截取
            what = "ADJUSTED_LAST" if self.adjust else "TRADES"
            bars = ib.reqHistoricalData(contract, "", duration, _BAR[freq], what, self.use_rth, formatDate=1)
            if bars:
                frames.append(util.df(bars))
        if not frames:
            return normalize(pd.DataFrame())
        df = pd.concat(frames).drop_duplicates("date")
        idx = pd.to_datetime(df["date"])
        if idx.dt.tz is not None:
            idx = idx.dt.tz_convert(tz).dt.tz_localize(None)  # 分钟线统一为交易所当地时间
        df.index = pd.DatetimeIndex(idx)
        out = normalize(df).loc[start_ts:end_ts]
        step = freq_to_timedelta(freq)
        if len(out) and freq in _CHUNK and out.index[-1] + step > now:
            out = out.iloc[:-1]  # 剔除未走完的分钟 K 线
        return out


class IbkrBroker(Broker):
    """IBKR 下单。order_type: MKT（市价）| ADAPTIVE（IB 自适应算法市价单，通常滑点更小）。"""

    def __init__(self, rules: MarketRules, symbols: list[str], cfg: dict | None = None, dry_run: bool = True):
        self.rules = rules
        self.cfg = settings(cfg)
        self.dry_run = dry_run
        self.ib = connect(self.cfg, readonly=dry_run)
        self.symbols = list(symbols)
        self.contracts = {s: make_contract(s, self.cfg) for s in self.symbols}
        self.ib.qualifyContracts(*self.contracts.values())
        self._by_con = {c.conId: s for s, c in self.contracts.items()}

    def _key(self, contract) -> str:
        if contract.conId in self._by_con:
            return self._by_con[contract.conId]
        c = self.cfg
        if contract.currency == c["currency"] and contract.exchange in ("", c["exchange"]):
            return contract.symbol
        return f"{contract.symbol}:{contract.exchange or contract.primaryExchange}:{contract.currency}"

    def positions(self) -> dict[str, float]:
        out = {}
        for p in self.ib.positions(self.cfg["account"]):
            if p.contract.secType == "STK" and p.position:
                out[self._key(p.contract)] = float(p.position)
        return out

    def cash(self) -> float:
        cur = self.cfg["currency"]
        vals = self.ib.accountValues(self.cfg["account"])
        for tag in ("CashBalance", "TotalCashValue"):
            for v in vals:
                if v.tag == tag and v.currency == cur:
                    return float(v.value)
        raise RuntimeError(f"账户中没有 {cur} 现金记录，请检查 ibkr.currency / ibkr.account")

    def market_open_today(self, now: pd.Timestamp) -> bool:
        """根据合约的 liquidHours 判断今天是否开市（自动识别节假日）。"""
        contract = next(iter(self.contracts.values()))
        details = self.ib.reqContractDetails(contract)
        if not details:
            return now.weekday() < 5
        today = now.strftime("%Y%m%d")
        for seg in details[0].liquidHours.split(";"):
            if seg.startswith(today):
                return "CLOSED" not in seg
        return now.weekday() < 5

    def execute(self, symbol: str, qty: float, price: float) -> dict | None:
        from ib_async import MarketOrder, TagValue

        amount = abs(float(qty))
        if amount <= 0:
            return None
        action = "BUY" if qty > 0 else "SELL"
        if self.dry_run:
            log.info("[DRY-RUN] %s %s %g @~%.4f", action, symbol, amount, price)
            return {"symbol": symbol, "qty": float(qty), "price": price, "fee": 0.0, "dry_run": True}
        order = MarketOrder(action, amount)
        if self.cfg["order_type"].upper() == "ADAPTIVE":
            order.algoStrategy = "Adaptive"
            order.algoParams = [TagValue("adaptivePriority", "Normal")]
        order.account = self.cfg["account"]
        trade = self.ib.placeOrder(self.contracts[symbol], order)
        deadline = time.monotonic() + float(self.cfg["fill_timeout"])
        while not trade.isDone() and time.monotonic() < deadline:
            self.ib.sleep(0.5)
        if not trade.isDone():
            log.warning("%s 订单 %s 超时未完全成交，撤单", symbol, trade.order.orderId)
            self.ib.cancelOrder(trade.order)
            self.ib.sleep(2)
        filled = float(trade.orderStatus.filled or 0.0)
        if filled <= 0:
            raise RuntimeError(f"{symbol} 订单未成交（状态 {trade.orderStatus.status}）")
        avg = float(trade.orderStatus.avgFillPrice or price)
        fee = sum(f.commissionReport.commission for f in trade.fills
                  if f.commissionReport and np.isfinite(f.commissionReport.commission))
        log.info("成交 %s %s %g @ %.4f 佣金 %.2f", action, symbol, filled, avg, fee)
        return {"symbol": symbol, "qty": float(np.sign(qty) * filled), "price": avg, "fee": fee,
                "id": trade.order.orderId}
