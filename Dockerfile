# 用于在服务器上长期运行模拟盘/实盘：
#   docker build -t quant .
#   docker run -d --name etf -v $PWD/live_state:/app/live_state -v $PWD/configs:/app/configs \
#     -e QUANT_WEBHOOK_URL=... quant live -c configs/ashare_etf_rotation.yaml
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["quant"]
CMD ["--help"]
