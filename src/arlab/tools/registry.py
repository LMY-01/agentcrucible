"""工具注册表：JSON Schema 声明 + 轻量校验 + 权限分级。

两个刻意的设计：

1. **校验失败不抛异常**，而是返回一段给模型看的错误文本。这样模型的非法调用
   会被 trace 记成 invalid_tool_call，变成可测量的指标（无效工具调用率），
   而不是让整个 run 崩掉。
2. **权限分级**。read 默认放行，write / net 要显式开启。线上 agent 出事故
   基本都是写文件和发网络请求，光靠 prompt 约束不住。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..faults import FAULT

READ, WRITE, NET = "read", "write", "net"


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: str | None = None
    duration_ms: int = 0


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., str]
    permission: str = READ
    # 终止型工具：调用即结束本轮，参数本身就是最终交付物。
    # 用工具而不是 prompt 约束输出格式——schema 校验是硬的，提示词是软的。
    terminal: bool = False
    # 交付门禁：terminal 工具生效前跑一遍，返回 None 放行，返回字符串就拒绝并让模型改。
    # 和事后判分的区别是「提交不进去」而不是「提交了扣分」。
    validator: Callable[[dict], str | None] | None = None

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def validate(args: dict, schema: dict) -> list[str]:
    """极简 JSON Schema 校验：type / required / enum / additionalProperties。

    够用且不引入 jsonschema 依赖；报错信息直接给模型看，所以写成中文短句。
    """
    errors: list[str] = []
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for key in required:
        if key not in args:
            errors.append(f"缺少必填参数 {key!r}")

    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            if schema.get("additionalProperties") is False:
                errors.append(f"不允许的参数 {key!r}")
            continue
        expected = spec.get("type")
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        if expected in type_map and not isinstance(value, type_map[expected]):
            errors.append(f"参数 {key!r} 应为 {expected}，实际给了 {type(value).__name__}")
        # Python 里 bool 是 int 的子类，JSON Schema 里 boolean 不是 integer，得单独挡
        elif expected in ("integer", "number") and isinstance(value, bool):
            errors.append(f"参数 {key!r} 应为 {expected}，实际给了 boolean")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"参数 {key!r} 必须是 {spec['enum']} 之一，实际给了 {value!r}")
    return errors


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)
    allow: set[str] = field(default_factory=lambda: {READ})
    injector: Any = None  # FaultInjector；故障注入挂在调用入口，对工具实现零侵入

    def register(self, tool: Tool) -> "Registry":
        if tool.name in self.tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self.tools[tool.name] = tool
        return self

    def names(self) -> list[str]:
        return sorted(self.tools)

    def has_terminal(self) -> bool:
        """是否存在「调用即交付」的终止型工具。"""
        return any(t.terminal for t in self.tools.values())

    def terminal_schemas(self) -> list[dict]:
        """只留终止型工具。预算耗尽收尾时用——关掉检索，但不关交付通道。"""
        return [t.openai_schema() for name, t in sorted(self.tools.items()) if t.terminal]

    def schemas(self) -> list[dict]:
        return [self.tools[n].openai_schema() for n in self.names()]

    def call(self, name: str, raw_args: str | None) -> ToolResult:
        """执行一次工具调用。任何失败都变成 ok=False 的结果，不向上抛。"""
        started = time.time()

        # 故障注入：命中就当作上游失败，走真实的失败路径
        if self.injector is not None:
            injected = self.injector.intercept(name)
            if injected:
                return ToolResult(
                    False, f"错误：{injected}", error=FAULT,
                    duration_ms=int((time.time() - started) * 1000),
                )

        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                False,
                f"错误：不存在工具 {name!r}。可用工具：{', '.join(self.names())}",
                error="unknown_tool",
                duration_ms=int((time.time() - started) * 1000),
            )

        if tool.permission not in self.allow:
            return ToolResult(
                False,
                f"错误：工具 {name!r} 需要 {tool.permission} 权限，当前未开启。",
                error="permission_denied",
                duration_ms=int((time.time() - started) * 1000),
            )

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            return ToolResult(
                False,
                f"错误：参数不是合法 JSON（{exc}）。请重新给出完整的 JSON 对象。",
                error="bad_json",
                duration_ms=int((time.time() - started) * 1000),
            )
        if not isinstance(args, dict):
            return ToolResult(False, "错误：参数必须是 JSON 对象。", error="bad_json")

        errs = validate(args, tool.parameters)
        if errs:
            return ToolResult(
                False,
                "错误：参数校验未通过 —— " + "；".join(errs),
                error="schema",
                duration_ms=int((time.time() - started) * 1000),
            )

        try:
            content = tool.handler(**args)
            return ToolResult(True, str(content), duration_ms=int((time.time() - started) * 1000))
        except Exception as exc:  # 工具内部异常不该杀掉 run
            return ToolResult(
                False,
                f"错误：{type(exc).__name__}: {exc}",
                error="exception",
                duration_ms=int((time.time() - started) * 1000),
            )
