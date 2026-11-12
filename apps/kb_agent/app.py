"""kb-agent：在 arlab 运行时上落地的垂直应用 —— 技术文档检索问答。

这一层只做三件事：
  1. 换工具集（kb_search / kb_read / cite_check 替代 read_file / run_python）
  2. 换技能集（retrieval-qa / refusal-policy 替代 csv-profiling）
  3. 换评测口径（引用可溯源、要点覆盖、拒答正确，替代 json_field 比对）

运行时一行没改。这就是「通用运行时 + 垂直应用」的实际形态。

用法：
    ./run.sh ask "LoRA 的 A 和 B 矩阵怎么初始化？"
    ./run.sh eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from arlab import Agent, Budget, LLM, Registry, SkillLoader, Trace, new_run_dir
from arlab.evaluate import _extract_json
from arlab.tools.builtin import build_tools

from .kb_tools import build_kb_tools, _store

APP_ROOT = Path(__file__).resolve().parent
SKILLS_DIR = APP_ROOT / "skills"
GOLD_PATH = APP_ROOT / "eval" / "gold.json"
RUNS_DIR = APP_ROOT / "runs"

DOMAIN_PROMPT = """你是一个技术文档问答助手，只能依据知识库里的资料回答。

硬性要求：
1. 每条结论必须有引用，引用必须从 kb_read 的输出里逐字复制，不许改写。
2. 提交前对每条引用跑一次 cite_check，未通过就回去重抄。
3. 资料里没有就拒答，不要用"一般来说""通常"包装猜测。

输出格式（接口契约，必须严格遵守，不要改成别的结构）：
{"answer": "结论，分点写并用 [1][2] 标出处", "citations": [{"id": 1, "chunk_id": "从 kb_search 拿到的 id", "quote": "原文逐字片段"}], "refused": false}

字段含义和检索策略见 retrieval-qa 技能；什么时候该拒答见 refusal-policy 技能。"""


# ---------------------------------------------------------------- 判据
# 垂直层专用谓词。它们都只依赖「引用能不能在原文找到」，
# 不需要领域专家判断答案对不对 —— 这是检索类应用能自动化评测的关键。

def _citations(answer: str) -> tuple[dict | None, list[dict]]:
    data = _extract_json(answer)
    if not data:
        return None, []
    cites = data.get("citations")
    return data, cites if isinstance(cites, list) else []


def pred_citations_grounded(answer: str, spec: dict) -> str | None:
    """每条引文必须在其 chunk 的原文里逐字存在，且至少一条。"""
    data, cites = _citations(answer)
    if data is None:
        return "答案不是可解析的 JSON"
    if not cites:
        return "一条引用都没给"
    store = _store()
    bad: list[str] = []
    for c in cites:
        if not isinstance(c, dict):
            bad.append("citation 不是对象")
            continue
        quote = (c.get("quote") or "").strip()
        cid = (c.get("chunk_id") or "").strip()
        if not quote or not cid:
            bad.append("citation 缺 quote 或 chunk_id")
            continue
        try:
            text = store.chunk(cid).text
        except KeyError:
            bad.append(f"chunk_id 不存在: {cid}")
            continue
        if quote not in text:
            bad.append(f"引文不在原文里: {quote[:36]!r}")
    if bad:
        return f"{len(bad)}/{len(cites)} 条引用无法溯源：" + "；".join(bad[:3])
    return None


def pred_answer_covers(answer: str, spec: dict) -> str | None:
    """要点覆盖：must 里至少命中 min_hit 个。"""
    data, _ = _citations(answer)
    if data is None:
        return "答案不是可解析的 JSON"
    text = str(data.get("answer") or "")
    must = spec.get("must") or []
    need = int(spec.get("min_hit", len(must)))
    hit = [w for w in must if w in text]
    if len(hit) < need:
        missing = [w for w in must if w not in text]
        return f"要点还差 {need - len(hit)} 个，缺：{missing}"
    return None


def pred_not_refused(answer: str, spec: dict) -> str | None:
    data, _ = _citations(answer)
    if data is None:
        return "答案不是可解析的 JSON"
    if data.get("refused") is True:
        return "这题资料里有答案，不该拒答"
    return None


def pred_states_missing(answer: str, spec: dict) -> str | None:
    """资料里确实没有时：要么标 refused，要么在正文里说明缺失。"""
    data, _ = _citations(answer)
    if data is None:
        return "答案不是可解析的 JSON"
    if data.get("refused") is True:
        return None
    text = str(data.get("answer") or "")
    words = spec.get("words") or ["没有", "未提及", "不包含", "找不到", "无法", "资料中没", "未提供"]
    if any(w in text for w in words):
        return None
    return f"既没标 refused，也没说明资料缺失。答案开头：{text[:60]!r}"


PREDICATES = {
    "citations_grounded": pred_citations_grounded,
    "answer_covers": pred_answer_covers,
    "not_refused": pred_not_refused,
    "states_missing": pred_states_missing,
}


# ---------------------------------------------------------------- 组装
def tool_builder(workspace: str, skills):
    """垂直层的工具集：检索三件套 + load_skill。不给文件读写和跑代码。"""
    from arlab.tools.builtin import build_tools as _build

    base = {t.name: t for t in _build(workspace, skill_loader=skills)}
    tools = build_kb_tools()
    if "load_skill" in base:
        tools.append(base["load_skill"])
    return tools


def make_agent(*, model: str, base_url: str | None, max_steps: int,
               trace: Trace | None = None) -> Agent:
    skills = SkillLoader(str(SKILLS_DIR)).discover()
    registry = Registry(allow={"read"})
    for tool in tool_builder(str(APP_ROOT), skills):
        registry.register(tool)
    return Agent(
        LLM(base_url=base_url, model=model),
        registry,
        skills=skills,
        budget=Budget(max_steps=max_steps, max_tokens=40_000, max_seconds=240.0),
        trace=trace,
        extra_system=DOMAIN_PROMPT,
    )


# ---------------------------------------------------------------- CLI
def cmd_ask(args) -> int:
    run_dir = new_run_dir(str(RUNS_DIR), tag="kb-ask")
    trace = Trace(run_dir, meta={"question": args.question, "model": args.model})
    agent = make_agent(model=args.model, base_url=args.base_url,
                       max_steps=args.max_steps, trace=trace)
    print(f"问题: {args.question}")
    print(f"技能: {', '.join(agent.skills.skills) if agent.skills else '无'}")
    print(f"trace: {trace.path}\n")

    result = agent.run(args.question)
    print("=" * 70)
    print(result.answer)
    print("=" * 70)

    summary = trace.summarize()
    trace.close(status=result.status, answer=result.answer, **summary)
    print(f"\n状态 {result.status}  步数 {summary['steps']}  "
          f"工具 {summary['tool_calls']}（失败 {summary['tool_failed']}）  "
          f"token {summary['total_tokens']}")
    return 0 if result.status == "ok" else 1


def cmd_eval(args) -> int:
    from arlab.evaluate import run_suite

    ns = argparse.Namespace(
        tasks=str(GOLD_PATH),
        limit=args.limit,
        workspace=str(APP_ROOT / "workspace"),
        skills=str(SKILLS_DIR),
        model=args.model,
        base_url=args.base_url,
        runs=str(RUNS_DIR),
        allow="read",
        max_steps=args.max_steps,
        max_tokens=40_000,
        max_seconds=240.0,
        out="",
        threshold=args.threshold,
        report=not args.no_report,
        report_out=args.report_out,
        judge=not args.no_judge,
        judge_model="",
    )
    return run_suite(ns, tool_builder=tool_builder, predicates=PREDICATES)


def main(argv: list[str] | None = None) -> int:
    from arlab.llm import DEFAULT_MODEL

    p = argparse.ArgumentParser(prog="kb-agent", description="技术文档检索问答 Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--base-url", default=None)
        sp.add_argument("--max-steps", type=int, default=8)

    a = sub.add_parser("ask", help="问一个问题")
    a.add_argument("question")
    common(a)
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="跑黄金集")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--threshold", type=float, default=0.90)
    e.add_argument("--report-out", default="")
    e.add_argument("--no-report", action="store_true")
    e.add_argument("--no-judge", action="store_true")
    common(e)
    e.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
