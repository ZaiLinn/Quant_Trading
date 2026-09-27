"""消息通知：日志 + 可选 webhook（Slack / 飞书 / 钉钉 / 通用 JSON）。"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, url: str | None = None, kind: str = "generic"):
        self.url = url or os.environ.get("QUANT_WEBHOOK_URL")
        self.kind = kind

    def _body(self, text: str) -> dict:
        if self.kind == "feishu":
            return {"msg_type": "text", "content": {"text": text}}
        if self.kind == "dingtalk":
            return {"msgtype": "text", "text": {"content": text}}
        return {"text": text}

    def send(self, text: str) -> None:
        log.info("通知: %s", text)
        if not self.url:
            return
        try:
            requests.post(self.url, json=self._body(text), timeout=10)
        except requests.RequestException as e:
            log.warning("webhook 发送失败: %s", e)
