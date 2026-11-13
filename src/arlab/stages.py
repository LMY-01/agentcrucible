"""分阶段评测：把 trace 当成一份过程证据，逐环节判分。

端到端评测只能回答「这一次过没过」，分阶段评测才能回答「哪一环掉了、为什么掉」。
这里每个阶段都是**从 trace 事件流派生**的确定性判据，不调模型、不做主观判断，
改一版 prompt 重跑，涨跌就是真涨跌。

八个阶段按一次 run 的时间顺序排列，构成一条漏斗：

    explore → tool_valid → args → recovery → skill_recall → efficiency → termination → answer

四级判定：
  pass  达标
  warn  能用但浪费了（例如并行探查同时盲猜文件名，白掉一次工具调用）
  fail  没做到
  na    该任务不考察这个阶段，不计入分母

warn 单列是因为它的处理方式和 fail 完全不同：fail 要改设计，warn 通常只要改提示词。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .jsonx import extract_json

# 阶段顺序 = 漏斗顺序，改顺序等于改叙事
STAGES = [
    ("explore", "探查"),
    ("tool_path", "工具路径"),
    ("tool_valid", "工具有效性"),
    ("args", "参数合法性"),
    ("recovery", "错误恢复"),
    ("skill_recall", "技能召回"),
    ("efficiency", "执行效率"),
    ("termination", "终止决策"),
    ("format", "输出格式"),
    ("answer", "答案正确"),
]

# 各阶段和错误码的对应关系：错误归到哪个阶段是固定契约，不能各处理解不同
ERROR_STAGE = {
    "unknown_tool": "tool_valid",
    "permission_denied": "tool_valid",
    "bad_json": "args",
    "schema": "args",
}

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "na"

@dataclass
class Verdict:
    stage: str
    name: str
    level: str  # pass | warn | fail | na
    detail: str
    evidence: str = ""  # 一句话现场，方便直接贴进报告

    @property
    def passed(self) -> bool | None:
        return {PASS: True, FAIL: False, WARN: False, NA: None}[self.level]

    @property
    def label(self) -> str:
        return {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", NA: "N/A"}[self.level]


def load_events(trace_path: str | Path) -> list[dict]:
    events = []
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _tool_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e["event"] == "tool"]


def _load_skill_calls(tools: list[dict]) -> list[str]:
    out = []
    for e in tools:
        if e.get("name") == "load_skill" and isinstance(e.get("args"), dict):
            name = e["args"].get("name")
            if isinstance(name, str):
                out.append(name)
    return out


def _same_call(a: dict, b: dict) -> bool:
    """两次调用是否是同一组参数。用来抓「同参数重试」这种坏习惯。"""
    return a.get("name") == b.get("name") and a.get("args") == b.get("args")


def _missed_read(tools: list[dict]) -> list[dict]:
    """读文件失败 = 对路径做了假设。这是探查阶段唯一的观测点。"""
    return [t for t in tools if t.get("name") == "read_file" and not t.get("ok")]


def judge(events: list[dict], task: dict, ctx: dict | None = None) -> list[Verdict]:
    """ctx 传运行时事实（status / steps / tokens / available_tools / fault_injected）。

    没有它的时候这些阶段只能靠猜；有了才能区分「做对了」和「碰巧看起来对」，
    以及「没加载技能」到底是失误还是本次根本没提供技能。
    """
    ctx = ctx or {}
    tools = _tool_events(events)
    end = next((e for e in events if e["event"] == "run_end"), {})
    limits = task.get("stage_limits") or {}
    verdicts: list[Verdict] = []

    # ---- 1. explore：有没有在动手前先确认工作区里到底有什么 ----
    if task.get("require_explore") is False:
        verdicts.append(Verdict("explore", "探查", NA, "任务给出的路径确定，不考察探查行为"))
    elif not tools:
        verdicts.append(Verdict("explore", "探查", FAIL, "全程没有调用任何工具",
                                evidence="没有任何 tool 事件"))
    else:
        misses = _missed_read(tools)
        if not misses:
            verdicts.append(Verdict("explore", "探查", PASS, "读取的路径都真实存在，没有盲猜"))
        else:
            first = misses[0]
            # 同一步里并行发了 list_dir，说明它知道自己不确定，只是懒得分两轮
            explored_before = any(
                t.get("name") == "list_dir" and t.get("step", 99) <= first.get("step", 0)
                for t in tools
            )
            arg = json.dumps(first.get("args"), ensure_ascii=False)
            if explored_before:
                verdicts.append(Verdict(
                    "explore", "探查", WARN,
                    "并行探查：同一步既列了目录、又按猜测的文件名去读，白掉一次工具调用",
                    evidence=f"step{first['step']} list_dir 与 read_file({arg}) 同时发出",
                ))
            else:
                verdicts.append(Verdict(
                    "explore", "探查", FAIL,
                    "完全盲猜：第一次读文件就假设了文件名，靠报错才纠正",
                    evidence=f"step{first['step']} read_file({arg}) -> {first.get('error')}",
                ))

    # ---- 工具路径合规：该调的调了没，不该调的有没有调 ----
    called = [t.get("name") for t in tools]
    required = task.get("required_tools") or []
    forbidden = set(task.get("forbidden_tools") or [])
    problems: list[str] = []
    missing = [r for r in required if r not in called]
    if missing:
        problems.append(f"未调用必选工具 {missing}")
    hit_forbidden = sorted({n for n in called if n in forbidden})
    if hit_forbidden:
        problems.append(f"调用了禁止工具 {hit_forbidden}")

    # 降级检测：某次工具失败之后，才出现的禁止工具 —— 这就是「挂了就换别的凑」
    fail_idx = next((i for i, t in enumerate(tools) if not t.get("ok")), None)
    if fail_idx is not None and forbidden:
        after = [t.get("name") for t in tools[fail_idx + 1 :]]
        downgraded = sorted({n for n in after if n in forbidden})
        if downgraded:
            problems.append(f"工具失败后降级到 {downgraded}（明令禁止）")

    if not required and not forbidden:
        verdicts.append(Verdict("tool_path", "工具路径", NA, "任务未声明工具路径约束"))
    elif problems:
        verdicts.append(Verdict("tool_path", "工具路径", FAIL, "；".join(problems),
                                evidence=f"实际调用链: {called}"))
    else:
        verdicts.append(Verdict("tool_path", "工具路径", PASS,
                                "必选工具都调了，未使用禁止工具"))

    # ---- 2. tool_valid / 3. args：按错误码归类 ----
    bad_by_stage: dict[str, list[dict]] = {"tool_valid": [], "args": []}
    for t in tools:
        stage = ERROR_STAGE.get(t.get("error") or "")
        if stage:
            bad_by_stage[stage].append(t)

    for stage_key, stage_name, hint in (
        ("tool_valid", "工具有效性", "调用了不存在的工具或缺权限"),
        ("args", "参数合法性", "参数不是合法 JSON 或不符合 schema"),
    ):
        bad = bad_by_stage[stage_key]
        if not bad:
            verdicts.append(Verdict(stage_key, stage_name, PASS, f"无{hint}",
                                    evidence=f"{len(tools)} 次调用全部合法"))
        else:
            sample = bad[0]
            verdicts.append(Verdict(
                stage_key, stage_name, FAIL,
                f"{len(bad)} 次{hint}",
                evidence=f"{sample.get('name')}({json.dumps(sample.get('args'), ensure_ascii=False)[:60]}) -> {sample.get('error')}",
            ))

    # ---- 4. recovery：失败之后是否换了策略 ----
    fails = [t for t in tools if not t.get("ok")]
    if not fails:
        verdicts.append(Verdict("recovery", "错误恢复", NA, "全程无工具失败，无从考察"))
    else:
        repeated = []
        for idx, f in enumerate(tools):
            if f.get("ok"):
                continue
            nxt = tools[idx + 1] if idx + 1 < len(tools) else None
            if nxt is not None and _same_call(f, nxt):
                repeated.append(nxt)
        if repeated:
            verdicts.append(Verdict("recovery", "错误恢复", FAIL,
                                    f"有 {len(repeated)} 次用同样的参数重试",
                                    evidence=f"重复调用 {repeated[0].get('name')}"))
        else:
            verdicts.append(Verdict("recovery", "错误恢复", PASS,
                                    f"{len(fails)} 次失败后都换了策略，没有同参数重试",
                                    evidence="失败后下一步参数均不同"))

    # ---- 5. skill_recall：该想起的手册想起来了吗 ----
    expected = task.get("expected_skills") or []
    loaded = _load_skill_calls(tools)
    available = ctx.get("available_tools")
    if not expected:
        verdicts.append(Verdict("skill_recall", "技能召回", NA, "任务未声明期望技能"))
    elif available is not None and "load_skill" not in available:
        # A/B 实验里可能整组都不提供技能，这不是「忘了加载」
        verdicts.append(Verdict("skill_recall", "技能召回", NA,
                                "本次运行未提供技能加载能力（技能目录为空）"))
    else:
        hit = [s for s in expected if s in loaded]
        if hit:
            verdicts.append(Verdict("skill_recall", "技能召回", PASS,
                                    f"加载了 {', '.join(hit)}",
                                    evidence=f"load_skill 调用: {loaded}"))
        else:
            verdicts.append(Verdict("skill_recall", "技能召回", FAIL,
                                    f"应加载 {expected}，实际一次都没加载",
                                    evidence=f"load_skill 调用: {loaded or '无'}"))

    # ---- 6. efficiency：步数和 token 是否在预算内 ----
    max_steps = int(limits.get("steps", 6))
    max_tokens = int(limits.get("tokens", 12_000))
    steps = int(end.get("steps") or 0)
    tokens = int(end.get("total_tokens") or 0)
    over_steps = steps > max_steps
    over_tokens = tokens > max_tokens
    if over_steps or over_tokens:
        reasons = []
        if over_steps:
            reasons.append(f"步数 {steps} > {max_steps}")
        if over_tokens:
            reasons.append(f"token {tokens} > {max_tokens}")
        verdicts.append(Verdict("efficiency", "执行效率", FAIL, "，".join(reasons),
                                evidence=f"预算上限 {max_steps} 步 / {max_tokens} token"))
    else:
        # 用到上限 80% 以上算 warn：这次过得去，但没有余量
        tight = steps / max_steps >= 0.8 or tokens / max_tokens >= 0.8
        verdicts.append(Verdict(
            "efficiency", "执行效率", WARN if tight else PASS,
            f"{steps} 步 / {tokens} token" + ("，已接近上限没有余量" if tight else "，余量充足"),
            evidence=f"上限 {max_steps} 步 / {max_tokens} token",
        ))

    # ---- 7. termination：是正常收尾还是被预算逼停 ----
    status = end.get("status") or "unknown"
    if status == "ok":
        verdicts.append(Verdict("termination", "终止决策", PASS,
                                "模型自行判断信息足够并停止", evidence="status=ok"))
    elif status == "budget_exhausted":
        verdicts.append(Verdict("termination", "终止决策", FAIL,
                                f"预算耗尽被强制收尾（{end.get('reason')}）",
                                evidence=f"status=budget_exhausted reason={end.get('reason')}"))
    else:
        verdicts.append(Verdict("termination", "终止决策", FAIL, f"异常结束: {status}",
                                evidence=f"status={status}"))

    # ---- 8. format：交付物结构是否合规 ----
    expects_json = bool(task.get("expects_json"))
    raw_answer = end.get("answer") or ""
    parsed = extract_json(raw_answer) if expects_json else None
    if not expects_json:
        verdicts.append(Verdict("format", "输出格式", NA, "任务未要求结构化输出"))
    elif parsed is not None:
        verdicts.append(Verdict("format", "输出格式", PASS, "答案是合法 JSON"))
    else:
        verdicts.append(Verdict(
            "format", "输出格式", FAIL,
            "要求 JSON 但答案不是可解析的 JSON——内容可能完全正确，下游却拿不到",
            evidence="开头: " + " ".join(raw_answer.strip().split())[:70],
        ))

    # ---- 9. answer：内容正确性 ----
    failures = end.get("failures") or []
    if end.get("ok"):
        verdicts.append(Verdict("answer", "答案正确", PASS, "全部断言通过"))
    elif expects_json and parsed is None:
        # 格式不合规时内容没法机械判定；已经记在 format 阶段，不再重复扣分
        verdicts.append(Verdict("answer", "答案正确", NA,
                                "格式不合规导致内容无法判定（已记在输出格式阶段）"))
    else:
        verdicts.append(Verdict("answer", "答案正确", FAIL,
                                "；".join(failures) or "断言未通过",
                                evidence=(end.get("answer") or "")[:80]))
    return verdicts


def aggregate(all_verdicts: list[list[Verdict]]) -> dict[str, dict[str, Any]]:
    """按阶段汇总：pass / warn / fail / na，并给出该阶段的一句话诊断。"""
    out: dict[str, dict[str, Any]] = {}
    for stage, name in STAGES:
        counts = {PASS: 0, WARN: 0, FAIL: 0, NA: 0}
        notes: list[str] = []
        for row in all_verdicts:
            v = next(x for x in row if x.stage == stage)
            counts[v.level] += 1
            if v.level in (WARN, FAIL):
                notes.append(f"{v.detail}")
        applicable = counts[PASS] + counts[WARN] + counts[FAIL]
        out[stage] = {
            "name": name,
            "pass": counts[PASS],
            "warn": counts[WARN],
            "fail": counts[FAIL],
            "skipped": counts[NA],
            "applicable": applicable,
            "rate": round(counts[PASS] / applicable, 3) if applicable else None,
            "notes": notes,
        }
    return out


def to_dict(verdicts: list[Verdict]) -> list[dict]:
    return [asdict(v) for v in verdicts]
