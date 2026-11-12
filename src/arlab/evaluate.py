"""评测：跑一批任务，按断言判分，汇总成一张能贴进 README 的表。

判分不依赖模型当裁判——每条任务自己声明期望（文件里该出现什么、答案该含什么
关键词），跑完对着工作区检查。这样分数是可复现的，改一版 prompt 或压缩策略再来
一次，涨跌就是真涨跌。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .budget import Budget
from .llm import LLM
from .loop import Agent
from .skills import SkillLoader
from .tools.builtin import build_tools
from .tools.registry import Registry
from .trace import Trace, new_run_dir


def _check(checks: list[dict], answer: str, workspace: Path) -> tuple[bool, list[str]]:
    """逐条断言。目前支持三种：answer_contains / file_exists / file_contains。"""
    failures: list[str] = []
    for c in checks:
        kind = c.get("type")
        if kind == "answer_contains":
            need = c["value"]
            if need.lower() not in answer.lower():
                failures.append(f"答案缺少 {need!r}")
        elif kind == "answer_contains_any":
            needs = c.get("values") or []
            if not any(n.lower() in answer.lower() for n in needs):
                failures.append(f"答案里 {needs} 一个都没出现")
        elif kind == "file_exists":
            if not (workspace / c["path"]).exists():
                failures.append(f"文件未生成: {c['path']}")
        elif kind == "file_contains":
            target = workspace / c["path"]
            if not target.exists():
                failures.append(f"文件未生成: {c['path']}")
            elif c["value"] not in target.read_text(encoding="utf-8", errors="replace"):
                failures.append(f"{c['path']} 里没有 {c['value']!r}")
        else:
            failures.append(f"未知断言类型 {kind!r}")
    return (not failures), failures


def run_suite(args) -> int:
    tasks_path = Path(args.tasks)
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if args.limit:
        tasks = tasks[: args.limit]

    allow = {"read"} | {p.strip() for p in (args.allow or "").split(",") if p.strip()}
    skills = SkillLoader(args.skills).discover() if args.skills else None

    out_root = Path(args.runs)
    rows = []
    print(f"评测 {len(tasks)} 条任务，模型 {args.model}，权限 {sorted(allow)}\n")

    for idx, task in enumerate(tasks, 1):
        workspace = Path(args.workspace) / task.get("workspace", "sandbox")
        workspace.mkdir(parents=True, exist_ok=True)
        for fname, content in (task.get("seed_files") or {}).items():
            target = workspace / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        registry = Registry(allow=allow)
        for tool in build_tools(str(workspace), skill_loader=skills):
            registry.register(tool)

        budget = Budget(
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            max_seconds=args.max_seconds,
        )
        run_dir = new_run_dir(str(out_root), tag=f"eval-{task['id']}")
        trace = Trace(run_dir, meta={"task_id": task["id"], "prompt": task["prompt"],
                                     "model": args.model, "budget": budget.snapshot()})
        agent = Agent(LLM(base_url=args.base_url, model=args.model), registry,
                      skills=skills, budget=budget, trace=trace)

        started = time.time()
        result = agent.run(task["prompt"])
        wall = round(time.time() - started, 2)
        ok, failures = _check(task.get("checks", []), result.answer, workspace)
        summary = trace.summarize()
        trace.close(ok=ok, failures=failures, answer=result.answer, **summary)

        rows.append(
            {
                "id": task["id"],
                "ok": ok,
                "status": result.status,
                "steps": summary["steps"],
                "tools": summary["tool_calls"],
                "tool_failed": summary["tool_failed"],
                "tokens": summary["total_tokens"],
                "wall_s": wall,
                "failures": failures,
            }
        )
        flag = "✅" if ok else "❌"
        print(f"{flag} {task['id']:<22} 步 {summary['steps']:>2}  "
              f"工具 {summary['tool_calls']:>2}  失败 {summary['tool_failed']:>2}  "
              f"token {summary['total_tokens']:>6}  {wall:>6.1f}s  {result.status}")
        for f in failures:
            print(f"      └ {f}")

    passed = sum(1 for r in rows if r["ok"])
    total = len(rows)
    agg = {
        "model": args.model,
        "tasks": total,
        "passed": passed,
        "success_rate": round(passed / total, 3) if total else 0.0,
        "avg_steps": round(sum(r["steps"] for r in rows) / total, 2) if total else 0.0,
        "avg_tokens": round(sum(r["tokens"] for r in rows) / total) if total else 0,
        "tool_calls": sum(r["tools"] for r in rows),
        "tool_failed": sum(r["tool_failed"] for r in rows),
        "avg_wall_s": round(sum(r["wall_s"] for r in rows) / total, 1) if total else 0.0,
        "rows": rows,
    }
    agg["invalid_tool_rate"] = (
        round(agg["tool_failed"] / agg["tool_calls"], 3) if agg["tool_calls"] else 0.0
    )

    out = args.out or os.path.join(str(out_root), f"eval_{int(time.time())}.json")
    Path(out).write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 56}")
    print(f"成功率 {passed}/{total} = {agg['success_rate']:.1%}"
          f"   平均步数 {agg['avg_steps']}   平均 token {agg['avg_tokens']}")
    print(f"工具调用 {agg['tool_calls']} 次，失败 {agg['tool_failed']}"
          f"，无效率 {agg['invalid_tool_rate']:.1%}")
    print(f"结果已写入 {out}")
    return 0 if passed == total else 1
