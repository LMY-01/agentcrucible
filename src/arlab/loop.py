"""Agent 主循环。

一次 run 的生命周期：
    system prompt（含技能目录） + 用户任务
      → LLM 决定调工具 → 校验 → 执行 → 结果回填
      → 直到模型不再调工具、或预算耗尽触发强制收尾

三个和玩具实现的区别：
  * 预算耗尽不是抛异常，而是关掉工具、注入收尾指令，逼出一次可用答案；
  * 工具失败被记成指标而不是中断；
  * 每一步都落 trace，人能复盘模型为什么走错。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .budget import Budget
from .llm import LLM, Usage
from .skills import SkillLoader
from .tools.registry import Registry, ToolResult
from .trace import Trace

MAX_TOOL_RESULT_CHARS = 4000

SYSTEM_TEMPLATE = """你是一个会使用工具的智能体，运行在一个受控的 Agent Runtime 里。

工作方式：
1. 先用工具获取事实，不要在没查之前凭记忆下结论。
2. 一次可以并行调用多个工具；拿到结果后判断是否还缺信息。
3. 信息足够后，直接给出最终答案，不要再调工具。
4. 工具报错时读懂错误信息再决定下一步，不要用同样的参数重试。
5. 无法完成时，明确说出卡在哪一步、缺什么，不要编造。

【可用技能目录】
以下是可按需加载的操作手册。只有当简述不足以完成任务时，才调 load_skill 加载全文。
{skill_catalog}

【运行限制】
单轮最多 {max_steps} 步，超时 {max_seconds:.0f} 秒。预算耗尽时必须基于已有信息收尾。
"""

FINISH_NUDGE = (
    "预算已耗尽（原因：{reason}）。检索类工具已关闭，但交付通道仍然开放。"
    "请立刻调用 submit_answer 提交你已有的结论；确实缺的部分，在 answer 里直接说明缺什么，不要编造。"
    "直接输出文本不会被采纳。"
)


@dataclass
class AgentResult:
    answer: str
    status: str  # ok | budget_exhausted | error
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    reason: str | None = None


class Agent:
    def __init__(
        self,
        llm: LLM,
        registry: Registry,
        *,
        skills: SkillLoader | None = None,
        budget: Budget | None = None,
        trace: Trace | None = None,
        extra_system: str = "",
        pricing=None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.skills = skills
        self.budget = budget or Budget()
        self.trace = trace
        self.extra_system = extra_system
        self.pricing = pricing  # Pricing 或 None；用于把 token 折成钱

    # ---- prompt ----------------------------------------------------
    def _system_prompt(self) -> str:
        base = SYSTEM_TEMPLATE.format(
            skill_catalog=self.skills.catalog() if self.skills else "(当前没有可用技能)",
            max_steps=self.budget.max_steps,
            max_seconds=self.budget.max_seconds,
        )
        if self.extra_system:
            base += "\n" + self.extra_system
        return base

    # ---- 单步工具执行 ----------------------------------------------
    def _exec_tool_call(self, call: dict, step: int) -> ToolResult:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        result = self.registry.call(name, fn.get("arguments"))
        if self.trace:
            try:
                args: Any = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = fn.get("arguments")
            self.trace.tool_call(
                step=step,
                name=name,
                args=args,
                ok=result.ok,
                duration_ms=result.duration_ms,
                result_chars=len(result.content),
                error=result.error,
            )
        self.budget.note_tool_result(result.ok)
        return result

    # ---- 主循环 ----------------------------------------------------
    def run(self, task: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task},
        ]
        usage = Usage()
        nudged = False
        pushed_submit = False

        while True:
            reason = self.budget.exhausted()
            tools = self.registry.schemas()

            if reason:
                if nudged:
                    return AgentResult(
                        answer="(预算耗尽且未能收尾)",
                        status="budget_exhausted",
                        steps=self.budget.steps,
                        usage=usage,
                        reason=reason,
                    )
                messages.append({"role": "user", "content": FINISH_NUDGE.format(reason=reason)})
                # 收尾时关掉检索类工具（继续调用只会更贵），但保留交付通道。
                # 之前这里是 tools=[]，等于把 submit_answer 也关了，
                # 导致模型只能裸输出文本，格式必然失败 —— 实测 g01 连续三轮栽在这上面。
                tools = self.registry.terminal_schemas()
                nudged = True

            reply = self.llm.chat(messages, tools=tools)
            usage = usage + reply.usage
            self.budget.charge(reply.usage.total)
            step = self.budget.steps

            if self.trace:
                self.trace.llm_call(
                    step=step,
                    model=self.llm.model,
                    usage=reply.usage,
                    latency_ms=reply.latency_ms,
                    finish_reason=reply.raw_finish_reason,
                    tools_offered=[t["function"]["name"] for t in tools],
                    ttft_ms=reply.ttft_ms,
                    reasoning_tokens=reply.reasoning_tokens,
                    cost=(self.pricing.cost(reply.usage.prompt_tokens,
                                            reply.usage.completion_tokens)
                          if self.pricing else 0.0),
                )

            msg = reply.message
            tool_calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()

            if not tool_calls:
                # 有终止型工具、模型却直接吐文本 —— 它绕过了交付通道。
                # 推一次回去逼它走工具，但不无限推（只推一次，且预算用完时不推）。
                if (self.registry.has_terminal() and content
                        and not pushed_submit and not nudged):
                    pushed_submit = True
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": ("你没有调用 submit_answer，直接输出的文本不会被采纳。"
                                    "请立即调用 submit_answer 工具提交答案，参数为 "
                                    "answer / citations / refused。"),
                    })
                    continue

                status = "budget_exhausted" if nudged else "ok"
                return AgentResult(
                    answer=content or "(模型返回空)",
                    status=status,
                    steps=step,
                    usage=usage,
                    reason=reason,
                )

            # 有工具调用：assistant 消息必须原样回填，否则下一轮会丢上下文
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                result = self._exec_tool_call(call, step)

                # 终止型工具：参数通过 schema 校验后直接作为最终交付物。
                # 格式由 runtime 保证，不依赖模型记不记得住提示词。
                fn = call.get("function") or {}
                spec = self.registry.tools.get(fn.get("name") or "")
                if spec is not None and spec.terminal and result.ok:
                    return AgentResult(
                        answer=fn.get("arguments") or "{}",
                        status="ok",
                        steps=step,
                        usage=usage,
                    )

                body = result.content
                if len(body) > MAX_TOOL_RESULT_CHARS:
                    body = body[:MAX_TOOL_RESULT_CHARS] + "\n...[结果过长已截断]"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": body,
                    }
                )
