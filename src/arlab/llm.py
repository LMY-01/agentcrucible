"""LLM 客户端：OpenAI 兼容 chat/completions，支持 tool calling。

零第三方依赖（只用标准库 urllib），任何 OpenAI 兼容端点都能通过环境变量换上来。

环境变量：
    ARLAB_BASE_URL   端点，如 http://172.28.208.1:3425/v1
    ARLAB_API_KEY    密钥
    ARLAB_MODEL      模型 id，默认 bailian/qwen3.8-flash
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_MODEL = os.environ.get("ARLAB_MODEL", "bailian/qwen3.8-flash")
_TIMEOUT = float(os.environ.get("ARLAB_TIMEOUT", "120"))
_RETRIES = int(os.environ.get("ARLAB_RETRIES", "3"))


def _gateway_ip() -> str | None:
    """WSL(NAT) 下 Windows 主机就是默认网关；拿不到就返回 None。"""
    try:
        with open("/proc/net/route", encoding="utf-8") as fh:
            for line in fh.readlines()[1:]:
                cols = line.split()
                if len(cols) > 2 and cols[1] == "00000000":
                    hexip = cols[2]
                    return ".".join(str(int(hexip[i : i + 2], 16)) for i in (6, 4, 2, 0))
    except OSError:
        pass
    return None


def resolve_base_url() -> str:
    """决定端点地址：环境变量 > 本机 localhost > WSL 默认网关。"""
    env = os.environ.get("ARLAB_BASE_URL")
    if env:
        return env.rstrip("/")

    candidates = ["http://127.0.0.1:3425/v1"]
    gw = _gateway_ip()
    if gw:
        candidates.append(f"http://{gw}:3425/v1")

    for url in candidates:
        host_port = url.split("//", 1)[1].split("/", 1)[0]
        host, port = host_port.rsplit(":", 1)
        try:
            with socket.create_connection((host, int(port)), timeout=1.5):
                return url
        except OSError:
            continue
    return candidates[-1]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Reply:
    """一次模型调用的结果。"""

    message: dict
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    raw_finish_reason: str | None = None


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.base_url = (base_url or resolve_base_url()).rstrip("/")
        self.api_key = api_key or os.environ.get("ARLAB_API_KEY", "")
        self.model = model

    # ---- 底层 HTTP -------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "arlab/0.1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(1, _RETRIES + 1):
            req = urllib.request.Request(
                f"{self.base_url}{path}", data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:400]
                last_err = LLMError(f"HTTP {exc.code}: {body}")
                # 4xx 除了 429 都不重试，重试也没用
                if 400 <= exc.code < 500 and exc.code != 429:
                    raise last_err from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_err = LLMError(f"网络错误: {exc}")
            if attempt < _RETRIES:
                time.sleep(1.5 * attempt)
        raise last_err or LLMError("未知失败")

    # ---- 对外 ------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Reply:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if max_tokens:
            payload["max_tokens"] = max_tokens

        started = time.time()
        data = self._post("/chat/completions", payload)
        latency = int((time.time() - started) * 1000)

        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"响应里没有 choices: {json.dumps(data)[:300]}")

        usage_raw = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        )
        return Reply(
            message=choices[0].get("message") or {},
            usage=usage,
            latency_ms=latency,
            raw_finish_reason=choices[0].get("finish_reason"),
        )
