"""运行预算：步数 / token / 时间 / 连续失败。

设计原则：超限不抛异常崩掉，而是让 loop 拿到一个明确的 exhaustion 原因，
注入一条「该收尾了」的指令，让 agent 有机会降级输出。硬崩的运行时在线上
会丢结果，这是和玩具实现的区别。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    max_steps: int = 12
    max_tokens: int = 120_000
    max_seconds: float = 180.0
    max_consecutive_tool_failures: int = 3

    steps: int = 0
    tokens: int = 0
    consecutive_failures: int = 0
    _started: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def charge(self, total_tokens: int) -> None:
        self.tokens += total_tokens
        self.steps += 1

    def note_tool_result(self, ok: bool) -> None:
        self.consecutive_failures = 0 if ok else self.consecutive_failures + 1

    def exhausted(self) -> str | None:
        """返回超限原因；还没超返回 None。按代价从小到大检查。"""
        if self.steps >= self.max_steps:
            return "steps"
        if self.tokens >= self.max_tokens:
            return "tokens"
        if self.elapsed >= self.max_seconds:
            return "time"
        if self.consecutive_failures >= self.max_consecutive_tool_failures:
            return "tool_failures"
        return None

    def snapshot(self) -> dict:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "elapsed_s": round(self.elapsed, 2),
            "consecutive_failures": self.consecutive_failures,
            "limits": {
                "max_steps": self.max_steps,
                "max_tokens": self.max_tokens,
                "max_seconds": self.max_seconds,
            },
        }
