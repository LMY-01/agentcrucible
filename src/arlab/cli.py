"""命令行入口：arlab run / arlab eval / arlab inspect。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .budget import Budget
from .llm import DEFAULT_MODEL, LLM
from .loop import Agent
from .skills import SkillLoader
from .tools.builtin import build_tools
from .tools.registry import READ, Registry
from .trace import Trace, new_run_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build(args) -> tuple[Agent, Trace]:
    allow = {READ}
    if args.allow:
        allow |= {p.strip() for p in args.allow.split(",") if p.strip()}

    skills = SkillLoader(args.skills).discover() if args.skills else None
    registry = Registry(allow=allow)
    for tool in build_tools(args.workspace, skill_loader=skills):
        registry.register(tool)

    budget = Budget(
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        max_seconds=args.max_seconds,
    )
    run_dir = new_run_dir(args.runs, tag=args.task[:40])
    trace = Trace(run_dir, meta={"task": args.task, "model": args.model,
                                "allow": sorted(allow), "budget": budget.snapshot()})
    agent = Agent(
        LLM(base_url=args.base_url, model=args.model),
        registry,
        skills=skills,
        budget=budget,
        trace=trace,
    )
    return agent, trace


def cmd_run(args) -> int:
    agent, trace = _build(args)
    print(f"▶ 任务: {args.task}")
    print(f"  模型: {args.model}   工具: {', '.join(agent.registry.names())}")
    print(f"  trace: {trace.path}\n")

    result = agent.run(args.task)

    print(f"\n{'=' * 60}\n{result.answer}\n{'=' * 60}")
    summary = trace.summarize()
    trace.close(status=result.status, reason=result.reason, answer=result.answer,
                **summary)

    print(f"\n状态: {result.status}" + (f" ({result.reason})" if result.reason else ""))
    print(f"步数: {summary['steps']}  工具调用: {summary['tool_calls']} "
          f"(失败 {summary['tool_failed']}, 无效率 {summary['invalid_tool_rate']})")
    print(f"token: {summary['total_tokens']}  耗时: "
          f"{(summary['llm_latency_ms'] + summary['tool_latency_ms']) / 1000:.1f}s")
    return 0 if result.status == "ok" else 1


def cmd_eval(args) -> int:
    from .evaluate import run_suite  # 延迟导入，run 命令不该被评测代码拖慢
    from .security import SECURITY_PREDICATES

    # 通用 runtime 的评测同时注册安全谓词：
    # 任何题集都可以用 no_leak / no_escape / bounded 这类判据，
    # 不必为「安全评测」单开一条代码路径。
    return run_suite(args, predicates=SECURITY_PREDICATES)


def cmd_inspect(args) -> int:
    """只看不跑：把最终会发给模型的 system prompt 和工具 schema 打出来。"""
    skills = SkillLoader(args.skills).discover() if args.skills else None
    registry = Registry(allow={READ, "write", "net"})
    for tool in build_tools(args.workspace, skill_loader=skills):
        registry.register(tool)
    agent = Agent(LLM(model=args.model), registry, skills=skills, trace=None)
    print(agent._system_prompt())
    print("\n" + "=" * 60 + "\n工具 schema：")
    print(json.dumps(registry.schemas(), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arlab", description="轻量 Agent Runtime")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--workspace", default=str(PROJECT_ROOT / "workspace"),
                       help="工具可见的目录，默认 ./workspace")
        p.add_argument("--skills", default=str(PROJECT_ROOT / "skills"),
                       help="技能目录，默认 ./skills")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--base-url", default=None)
        p.add_argument("--runs", default=str(PROJECT_ROOT / "runs"))

    p_run = sub.add_parser("run", help="跑一个任务")
    p_run.add_argument("task")
    p_run.add_argument("--allow", default="", help="额外开启的权限，如 write,net")
    p_run.add_argument("--max-steps", type=int, default=12)
    p_run.add_argument("--max-tokens", type=int, default=120_000)
    p_run.add_argument("--max-seconds", type=float, default=180.0)
    common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="跑评测集")
    p_eval.add_argument("--tasks", default=str(PROJECT_ROOT / "eval" / "tasks.json"))
    p_eval.add_argument("--allow", default="write,net")
    p_eval.add_argument("--limit", type=int, default=0)
    p_eval.add_argument("--max-steps", type=int, default=12)
    p_eval.add_argument("--max-tokens", type=int, default=120_000)
    p_eval.add_argument("--max-seconds", type=float, default=180.0)
    p_eval.add_argument("--out", default="")
    p_eval.add_argument("--threshold", type=float, default=0.90,
                        help="质量分及格线，低于它就返回非零退出码（CI 闸门）")
    p_eval.add_argument("--report", dest="report", action="store_true", default=True,
                        help="生成 HTML 对照报告（默认开）")
    p_eval.add_argument("--no-report", dest="report", action="store_false")
    p_eval.add_argument("--report-out", default="", help="报告输出路径")
    p_eval.add_argument("--judge", dest="judge", action="store_true", default=True,
                        help="失败任务用 LLM 做归因（默认开）")
    p_eval.add_argument("--no-judge", dest="judge", action="store_false")
    p_eval.add_argument("--judge-model", default="", help="归因用的模型，默认与评测模型相同")
    p_eval.add_argument("--defense", action="store_true",
                        help="开启入口防御（工具输出注入检测 + 数据标记）；实测无效，ASR 反升")
    p_eval.add_argument("--output-filter", action="store_true",
                        help="开启出口过滤（交付前拦敏感值，命中就打回改写）")
    p_eval.add_argument("--repeat", type=int, default=1,
                        help="重复跑 N 轮，准确率取中位数并给区间")
    common(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_ins = sub.add_parser("inspect", help="打印 system prompt 和工具 schema")
    common(p_ins)
    p_ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
