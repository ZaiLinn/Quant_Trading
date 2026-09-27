# 量化交易运行镜像（回测 / 模拟盘 / 实盘 / 自动驾驶）
#   docker compose up -d            # 推荐：见 docker-compose.yml 与 DEPLOY.md
#   docker build -t quant . && docker run --rm quant strategies
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install --no-deps -e . && mkdir -p live_state data_cache runs

ENTRYPOINT ["quant"]
CMD ["--help"]
