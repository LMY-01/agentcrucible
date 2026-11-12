"""LLM-as-a-Judge：读失败现场，判断「为什么错」并给可执行的修复建议。

和评测判分是两件事，必须分开：

  * 判分（evaluate.py）用确定性断言，答案是死的，谁跑都一样
  * 归因（本文件）用模型读 trace，产出的是假设，可能错，所以要标 confidence

把两者混在一起会毁掉评测：让模型既当运动员又当裁判，分数就没法比了。
这里只在**已经判失败**的任务上做归因，且结果不影响分数，只进报告。

输出是固定枚举的 error_type，这样才能跨 run 聚合出「这批任务主要栽在哪类问题上」。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLM, LLMError

# 固定枚举：改这里等于改统计口径，别随手加
ERROR_TYPES = {
    "reasoning_error": "推理或计算出错",
    "output_format": "输出格式不符合要求",
    "tool_misuse": "工具选择或参数使用错误",
    "prompt_ambiguity": "任务描述本身没说清，模型理解偏了",
    "context_missing": "没获取到完成任务所需的信息就下结论",
    "hallucination": "编造了资料里没有的事实或数字",
    "skill_miss": "该加载的技能手册没加载",
    "other": "其他",
}

SYSTEM = """你是一个 Agent 评测的诊断员。你会看到：任务描述、运行时的工具调用轨迹、模型的最终答案、
标准答案、以及确定性判分给出的扣分项。你的唯一工作是判断**失败的根本原因**并给出修复建议。

规则：
1. 只依据给你的材料下判断，不要猜运行时之外的东西。
2. 扣分项是事实，不要质疑它；你要解释它为什么发生。
3. 修复建议必须落到具体可改的东西上：改提示词的哪一句、改哪个工具的什么行为、改任务的什么描述。
   禁止写"加强模型能力""优化提示词"这类没法执行的话。
4. 如果证据不足以下结论，confidence 填 low，不要硬编。

输出严格为 JSON，不要 markdown 代码块，不要解释文字：
{
  "error_type": "从这些里选一个: """ + " | ".join(ERROR_TYPES) + """,
  "root_cause": "一句话说清直接原因，必须引用轨迹里的具体证据",
  "evidence": "轨迹里的关键那一步，例如 step2 read_file(data.csv) 返回 exception",
  "suggestion": "一条可执行的修复动作，指明改哪里",
  "confidence": "high | medium | low"
}"""


@dataclass
class Diagnosis:
    error_type: str
    root_cause: str
    evidence: str
    suggestion: str
    confidence: str
    raw: str = ""
    ok: bool = True
    error: str = ""

    @property
    def label(self) -> str:
        return ERROR_TYPES.get(self.error_type, self.error_type)


def _trace_digest(events: list[dict], limit: int = 14) -> str:
    """把 trace 压成给人（和模型）看的时间线，只保留有信息量的行。"""
    lines: list[str] = []
    for ev in events:
        if ev["event"] == "llm":
            lines.append(f"step{ev.get('step')} [LLM] {ev.get('completion_tokens')} tokens, "
                         f"{ev.get('latency_ms')}ms, finish={ev.get('finish_reason')}")
        elif ev["event"] == "tool":
            status = "ok" if ev.get("ok") else f"FAIL({ev.get('error')})"
            args = json.dumps(ev.get("args"), ensure_ascii=False)[:110]
            lines.append(f"step{ev.get('step')} [TOOL] {ev.get('name')} {args} -> {status}")
        elif ev["event"] == "run_end":
            lines.append(f"end status={ev.get('status')}")
    if len(lines) > limit:
        lines = lines[: limit - 2] + [f"... 省略 {len(lines) - limit} 行 ..."] + lines[-2:]
    return "\n".join(lines)


def _build_user_prompt(task: dict, answer: str, failures: list[str],
                       events: list[dict]) -> str:
    seeds = "\n".join(
        f"--- {name} ---\n{content}" for name, content in (task.get("seed_files") or {}).items()
    ) or "(工作区没有预置文件)"
    return f"""【任务】
{task.get('prompt', '')}

【工作区预置材料】
{seeds}

【工具调用轨迹】
{_trace_digest(events)}

【模型最终答案】
{answer}

【标准答案】
{task.get('reference_answer', '(未提供)')}
{f"补充说明：{task['reference_note']}" if task.get("reference_note") else ""}

【判分扣分项】
{chr(10).join("- " + f for f in failures) if failures else "- (无)"}

按 JSON 输出诊断。"""


def diagnose(
    llm: LLM,
    task: dict,
    answer: str,
    failures: list[str],
    events: list[dict],
) -> Diagnosis:
    """对一次失败做归因。任何异常都降级成 ok=False，绝不让报告生成失败。"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _build_user_prompt(task, answer, failures, events)},
    ]
    try:
        reply = llm.chat(messages, temperature=0.0, max_tokens=700)
    except LLMError as exc:
        return Diagnosis("other", "诊断调用失败", "", "", "low", ok=False, error=str(exc))

    raw = (reply.message.get("content") or "").strip()
    text = raw
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return Diagnosis("other", "诊断结果不是合法 JSON", "", "", "low",
                             raw=raw, ok=False, error="bad_json")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return Diagnosis("other", "诊断结果不是合法 JSON", "", "", "low",
                             raw=raw, ok=False, error="bad_json")

    etype = str(data.get("error_type") or "other")
    if etype not in ERROR_TYPES:
        etype = "other"
    return Diagnosis(
        error_type=etype,
        root_cause=str(data.get("root_cause") or "")[:400],
        evidence=str(data.get("evidence") or "")[:300],
        suggestion=str(data.get("suggestion") or "")[:400],
        confidence=str(data.get("confidence") or "low"),
        raw=raw,
    )
