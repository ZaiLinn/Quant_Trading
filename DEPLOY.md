# 部署到 VPS

让模拟盘 / 实盘 / 自动驾驶 7×24 小时运行。推荐 Docker Compose，一条命令启动、崩溃自动重启、重启服务器后自动恢复。

## 1. 选 VPS

| 项目 | 建议 |
|---|---|
| 配置 | 1–2 vCPU、2 GB 内存、20 GB 硬盘足够（跑 IB Gateway 建议 2 GB 以上内存） |
| 系统 | Ubuntu 22.04 / 24.04 |
| 地区 | 美股 / IBKR：任意地区均可，美东延迟最低；A 股数据（新浪 / 腾讯 / 东财）：香港、新加坡、日本访问较稳定，欧美机房可能慢或被限；币安：**不要选美国机房**（币安限制美国 IP） |

## 2. 初始化

SSH 登录 VPS 后：

```bash
curl -fsSL https://raw.githubusercontent.com/ZaiLinn/Quant_Trading/main/deploy/setup_ubuntu.sh | bash
# 重新登录 SSH 让 docker 权限生效
cd ~/Quant_Trading
nano .env            # 填写 webhook（强烈建议，出问题能第一时间收到通知）、IBKR 账号等
```

脚本会安装 Docker、开启只放行 SSH 的防火墙、拉取代码并从 `.env.example` 创建 `.env`（权限 600）。

## 3. 运行

先试跑一次（`--force` 忽略交易日历，周末也能验证流程）：

```bash
docker compose run --rm autopilot autopilot -c configs/us_autopilot.yaml --once --force
```

没有报错就常驻运行：

```bash
docker compose up -d                   # 本地模拟盘（不需要 IBKR）
docker compose --profile ibkr up -d    # 连接 IB Gateway 模拟账户（需在 .env 填写 TWS_USERID / TWS_PASSWORD）
```

默认服务 `autopilot` 运行 `configs/us_autopilot.yaml`。换策略：修改 `docker-compose.yml` 中的 `command`，或复制一个服务运行多个配置（每个配置的 `name` 必须不同，状态文件按 `name` 区分）。

## 4. 日常运维

```bash
docker compose ps                                        # 运行状态
docker compose logs -f autopilot                         # 实时日志（也写入 live_state/<name>.log）
docker compose run --rm autopilot status -c configs/us_autopilot.yaml --report      # 持仓、成交、权益报告
docker compose run --rm autopilot autopilot -c configs/us_autopilot.yaml --status   # 自动优化决策记录
touch live_state/STOP                                    # 紧急熔断：只减仓不开仓；rm 恢复
docker compose restart autopilot                         # 修改配置后重启生效
git pull && docker compose up -d --build                 # 更新代码
docker compose down                                      # 停止
```

查看 HTML 报告：`scp` 下载 `live_state/*_report.html` 或 `runs/` 目录到本地打开。

**备份** `live_state/` 目录（模拟盘账户、运行状态、决策记录）：

```bash
tar czf live_state_$(date +%F).tgz live_state/
```

## 5. IBKR 注意事项

- 网关镜像 `ghcr.io/gnzsnz/ib-gateway` 内置 IBC 自动登录；策略容器通过 `ib-gateway:4004`（模拟）/ `4003`（实盘）连接，已在 `docker-compose.yml` 中配置好。
- **先用模拟账户**：`.env` 中 `TRADING_MODE=paper`，配置里保持 `live.dry_run: true` 运行几天，确认订单合理后再关闭 dry-run。
- **实盘账户需要 2FA**：首次登录与每周重新认证时，手机 IBKR Mobile 会收到确认请求；已设置每日自动重启（`AUTO_RESTART_TIME`）以避免每天都要确认。确认超时会自动重试。
- **API 端口只绑定 127.0.0.1**，绝不对公网开放。本地电脑想连 VPS 上的网关：`ssh -L 4002:127.0.0.1:4002 user@vps`，然后本地用端口 4002。
- 用 `live.capital` 限定策略可用资金；账户里手动持有的其他股票不会被策略动到。

## 6. 不用 Docker（systemd）

```bash
sudo apt install -y python3-venv git
git clone https://github.com/ZaiLinn/Quant_Trading.git && cd Quant_Trading
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
sudo cp deploy/quant-autopilot.service /etc/systemd/system/   # 按需修改其中的用户名、路径、配置文件
sudo systemctl daemon-reload && sudo systemctl enable --now quant-autopilot
journalctl -u quant-autopilot -f
```

IBKR 用 systemd 方案时需要自行安装 IB Gateway + IBC，Docker 方案更省事。

## 7. 安全清单

- `.env` 权限 600，不提交到 Git（已在 `.gitignore`）；API Key 只从环境变量读取。
- 交易所 API Key 只开"交易"权限，**关闭提现权限**，并绑定 VPS 的 IP 白名单。
- SSH 使用密钥登录、禁用密码登录；防火墙只放行 SSH。
- 配置 webhook 通知：下单、熔断、运行失败、自动优化结果都会推送。
