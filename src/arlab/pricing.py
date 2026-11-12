"""成本核算：把 token 用量换算成钱。

为什么必须在评测里报钱：token 数对工程师有意义，对拍板的人没有意义。
「每任务 2.7 万 token」没人听得懂，「每次问答 0.003 美元」才能进预算表。

单价从网关的 /v1/models 接口读 `pricing` 字段，不写死在本仓库——
写死了换模型就错，而且各地的计价会变。读不到时退回「未计价」并明确标注，
宁可显示 N/A，也不要给一个假数字。
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CACHE = Path(os.environ.get("ARLAB_PRICING_CACHE", str(Path.home() / ".hermes" / ".arlab-pricing.json")))


@dataclass
class Pricing:
    prompt_per_token: float = 0.0
    completion_per_token: float = 0.0
    currency: str = "USD"

    @property
    def known(self) -> bool:
        return self.prompt_per_token > 0 or self.completion_per_token > 0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.prompt_per_token
                + completion_tokens * self.completion_per_token)


def fetch_pricing(base_url: str, api_key: str = "", timeout: float = 8.0) -> dict[str, dict]:
    """从网关读全部模型的单价。返回 {model_id: {prompt, completion}}。"""
    req = urllib.request.Request(f"{base_url.rstrip('/')}/models")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    out: dict[str, dict] = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    for m in data.get("data") or []:
        pricing = m.get("pricing") or {}
        if "prompt" in pricing or "completion" in pricing:
            out[m.get("id", "")] = {
                "prompt": float(pricing.get("prompt") or 0),
                "completion": float(pricing.get("completion") or 0),
            }
    return out


def load(model: str, base_url: str, api_key: str = "") -> Pricing:
    """取某模型的单价，带本地缓存；网络失败就用缓存；都没有就返回空计价。"""
    table: dict[str, dict] = {}
    if CACHE.exists():
        try:
            table = json.loads(CACHE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            table = {}

    if model not in table:
        try:
            fresh = fetch_pricing(base_url, api_key)
            if fresh:
                table.update(fresh)
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass  # 拿不到就用手上有的，没有就空计价

    row = table.get(model) or {}
    return Pricing(
        prompt_per_token=float(row.get("prompt") or 0),
        completion_per_token=float(row.get("completion") or 0),
    )


def human(amount: float, currency: str = "USD") -> str:
    """钱要按量级选精度，否则会显示成 0.00 或 0.0000001。"""
    if amount <= 0:
        return "—"
    symbol = "$" if currency == "USD" else ("¥" if currency == "CNY" else "")
    if amount < 0.01:
        return f"{symbol}{amount:.5f}"
    if amount < 1:
        return f"{symbol}{amount:.4f}"
    return f"{symbol}{amount:.2f}"
