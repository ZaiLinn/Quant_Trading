PY ?= .venv/bin/python
CFG ?= configs/demo_synthetic.yaml

.PHONY: install test demo backtest optimize walkforward factor check paper status clean

install:
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

test:
	$(PY) -m pytest

demo:
	$(PY) -m quant backtest -c configs/demo_synthetic.yaml

backtest:
	$(PY) -m quant backtest -c $(CFG)

optimize:
	$(PY) -m quant optimize -c $(CFG) --jobs -1

walkforward:
	$(PY) -m quant walkforward -c $(CFG) --jobs -1

factor:
	$(PY) -m quant factor -c $(CFG)

check:
	$(PY) -m quant check -c $(CFG)

paper:
	$(PY) -m quant live -c $(CFG)

status:
	$(PY) -m quant status -c $(CFG) --report

clean:
	rm -rf runs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
