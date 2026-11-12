"""执行轨迹落盘。

一次 run 一个目录，一行一个事件（jsonl），不做缓冲——进程被杀也能留下现场。
这是后面做评测归因和造 SFT/DPO 偏好对的原料。
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class Trace:
    def __init__(self, run_dir: str, meta: dict[str, Any] | None = None) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "trace.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")
        self.step = 0
        if meta is not None:
            self.emit("run_start", **meta)

    def emit(self, event: str, **fields: Any) -> dict:
        rec = {"ts": _now_iso(), "event": event, **fields}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        return rec

    def llm_call(self, *, step: int, model: str, usage, latency_ms: int,
                 finish_reason: str | None, tools_offered: list[str]) -> None:
        self.emit(
            "llm",
            step=step,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            tools_offered=tools_offered,
        )

    def tool_call(self, *, step: int, name: str, args: Any, ok: bool,
                  duration_ms: int, result_chars: int, error: str | None = None) -> None:
        self.emit(
            "tool",
            step=step,
            name=name,
            args=args,
            ok=ok,
            duration_ms=duration_ms,
            result_chars=result_chars,
            error=error,
        )

    def close(self, **summary: Any) -> None:
        self.emit("run_end", **summary)
        self._fh.close()

    # ---- 统计 ------------------------------------------------------
    def summarize(self) -> dict:
        """读回 jsonl，算一份指标。"""
        steps = 0
        tool_total = tool_failed = 0
        prompt_tokens = completion_tokens = 0
        llm_latency = tool_latency = 0
        tool_names: dict[str, int] = {}

        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["event"] == "llm":
                    steps += 1
                    prompt_tokens += rec.get("prompt_tokens", 0)
                    completion_tokens += rec.get("completion_tokens", 0)
                    llm_latency += rec.get("latency_ms", 0)
                elif rec["event"] == "tool":
                    tool_total += 1
                    if not rec.get("ok"):
                        tool_failed += 1
                    tool_latency += rec.get("duration_ms", 0)
                    tool_names[rec["name"]] = tool_names.get(rec["name"], 0) + 1

        return {
            "steps": steps,
            "tool_calls": tool_total,
            "tool_failed": tool_failed,
            "invalid_tool_rate": round(tool_failed / tool_total, 3) if tool_total else 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "llm_latency_ms": llm_latency,
            "tool_latency_ms": tool_latency,
            "tool_histogram": tool_names,
        }


def new_run_dir(root: str, tag: str) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:40]
    return os.path.join(root, f"{stamp}__{safe}")
