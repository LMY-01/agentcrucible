"""故障注入：在指定工具的第 N 次调用上强制失败。

为什么必须有这个：错误恢复是整条流水线里最难自动验证的一环——
模型在一切正常时根本不会犯错，所以你永远等不到一个失败样本。
实测统计：5 条任务跑三轮共 50 次工具调用，零失败，`错误恢复` 阶段 15 次全是 N/A。

靠等不行，只能造。注入的失败和真实失败在 agent 眼里没有区别：
它拿到的是 ToolResult(ok=False, error="injected_fault")，和网络超时、权限不足走同一条路径。

注入点放在 Registry 层而不是工具内部，因为这样对工具实现零侵入——
接任意工具集都能注入，不用每个 handler 自己写 if。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 注入故障的错误码。单独一个值，方便在 trace 里区分「真挂」和「假挂」
FAULT = "injected_fault"

DEFAULT_MESSAGES = {
    "exception": "服务不可用（连接超时），本次调用失败。",
    "timeout": "请求超时：上游在 30 秒内没有响应。",
    "empty": "调用成功但返回了空结果（上游索引为空）。",
    "denied": "权限不足：该接口当前不可用。",
}


@dataclass
class FaultInjector:
    """第 at_call 次调用 tool 时返回失败。

    sticky=True 表示从第 at_call 次起**一直**失败——用来模拟「接口挂了」，
    而不是「偶发抖动」。两者的正确应对不同：后者应该重试，前者必须降级报告。
    """

    tool: str
    at_call: int = 1
    mode: str = "exception"
    message: str = ""
    sticky: bool = False
    seen: int = field(default=0, init=False)

    @classmethod
    def from_spec(cls, spec: dict | None) -> "FaultInjector | None":
        if not spec:
            return None
        return cls(
            tool=spec.get("tool", ""),
            at_call=int(spec.get("at_call", 1)),
            mode=spec.get("mode", "exception"),
            message=spec.get("message", ""),
            sticky=bool(spec.get("sticky", False)),
        )

    def intercept(self, name: str) -> str | None:
        """命中返回错误文本，未命中返回 None。只对指定工具计数。"""
        if not self.tool or name != self.tool:
            return None
        self.seen += 1
        fired = self.seen >= self.at_call if self.sticky else self.seen == self.at_call
        if not fired:
            return None
        return self.message or DEFAULT_MESSAGES.get(self.mode, DEFAULT_MESSAGES["exception"])

    def describe(self) -> str:
        span = f"第 {self.at_call} 次起全部" if self.sticky else f"第 {self.at_call} 次"
        return f"{self.tool} {span} → {self.mode}"
