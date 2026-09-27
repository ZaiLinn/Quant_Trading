#!/usr/bin/env bash
# Ubuntu 22.04 / 24.04 VPS 初始化：安装 Docker、拉取代码、准备目录。
# 用法（以有 sudo 权限的普通用户运行）：
#   curl -fsSL https://raw.githubusercontent.com/ZaiLinn/Quant_Trading/main/deploy/setup_ubuntu.sh | bash
set -euo pipefail

REPO="${REPO:-https://github.com/ZaiLinn/Quant_Trading.git}"
DIR="${DIR:-$HOME/Quant_Trading}"

if ! command -v docker >/dev/null 2>&1; then
  echo ">> 安装 Docker"
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

echo ">> 基础防火墙：只放行 SSH（IB Gateway 端口只绑定本机，不需要放行）"
if command -v ufw >/dev/null 2>&1; then
  sudo ufw allow OpenSSH >/dev/null
  sudo ufw --force enable >/dev/null
fi

if [ ! -d "$DIR/.git" ]; then
  git clone "$REPO" "$DIR"
fi
cd "$DIR"
mkdir -p live_state data_cache runs
[ -f .env ] || { cp .env.example .env; chmod 600 .env; }

echo
echo "完成。下一步："
echo "  1. 重新登录 SSH（使 docker 组生效）"
echo "  2. cd $DIR && nano .env                         # 填写 webhook / IBKR 账号"
echo "  3. docker compose run --rm autopilot autopilot -c configs/us_autopilot.yaml --once --force   # 试运行"
echo "  4. docker compose up -d                          # 常驻运行（本地模拟盘）"
echo "     docker compose --profile ibkr up -d           # 或：连 IB Gateway 模拟账户"
