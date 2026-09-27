"""quant：精简的量化交易框架（数据 / 策略 / 回测 / 分析 / 优化 / 模拟盘与实盘）。"""
from .backtest import BacktestEngine, BacktestResult, run_backtest
from .data import Panel, load_panel, make_source
from .market import MarketRules, get_rules
from .strategy import Strategy, get_strategy, register

__version__ = "0.1.0"
__all__ = ["BacktestEngine", "BacktestResult", "run_backtest", "Panel", "load_panel", "make_source",
           "MarketRules", "get_rules", "Strategy", "get_strategy", "register"]
