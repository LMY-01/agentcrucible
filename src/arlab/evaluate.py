"""评测：跑一批任务，按断言判分，汇总成一张能贴进 README 的表。

判分不依赖模型当裁判——每条任务自己声明期望（文件里该出现什么、答案该含什么
关键词），跑完对着工作区检查。这样分数是可复现的，改一版 prompt 或压缩策略再来
一次，涨跌就是真涨跌。
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .budget import Budget
from .llm import LLM
from .loop import Agent
from .judge import diagnose as diagnose_failure
from .report import write as write_report
from .scoring import run_score
from .skills import SkillLoader
from .stages import FAIL, NA, STAGES, WARN, aggregate, judge, load_events, to_dict
from .tools.builtin import build_tools
from .tools.registry import Registry
from .trace import Trace, new_run_dir


def _pad(text: str, width: int) -> str:
    """按显示宽度左对齐：中文和全角符号占两列，用 len() 对齐会错位。"""
    shown = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    return text + " " * max(0, width - shown)


def _extract_json(text: str) -> dict | None:
    """从模型输出里挖第一段可解析的 JSON 对象。

    模型经常把 JSON 包在 ```json 块里，或者前面加一句"结果如下："，
    所以不能直接 json.loads(全文)。先试代码块，再试括号平衡扫描。
    """
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : idx + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _dig(data: Any, path: str) -> Any:
    """按 a.b.c 取值，取不到返回 None。"""
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


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
        elif kind == "json_field":
            data = _extract_json(answer)
            if data is None:
                failures.append(f"答案里没有可解析的 JSON（需要 {c.get('field')}）")
                continue
            got = _dig(data, c["field"])
            if "number" in c:
                tol = float(c.get("tolerance", 0.01))
                if not isinstance(got, (int, float)) or isinstance(got, bool):
                    failures.append(f"{c['field']} 应为数字 {c['number']}，实际 {got!r}")
                elif abs(got - c["number"]) > tol:
                    failures.append(f"{c['field']} 应为 {c['number']}，实际 {got}")
            elif "equals" in c:
                if got != c["equals"]:
                    failures.append(f"{c['field']} 应为 {c['equals']!r}，实际 {got!r}")
            elif "contains" in c:
                if not isinstance(got, str) or c["contains"] not in got:
                    failures.append(f"{c['field']} 应包含 {c['contains']!r}，实际 {got!r}")
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
    all_verdicts = []
    details = []  # 给报告用的完整现场，包含输入材料和标准答案
    judge_llm = LLM(base_url=args.base_url, model=getattr(args, "judge_model", None) or args.model)
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
        trace.close(ok=ok, failures=failures, answer=result.answer,
                    status=result.status, reason=result.reason, **summary)

        # 端到端判完再回看 trace，做阶段级判定
        events = load_events(trace.path)
        verdicts = judge(events, task)
        is_control = bool(task.get("control"))
        if not is_control:
            all_verdicts.append(verdicts)  # 对照组不参与任何统计
        issues = [v for v in verdicts if v.level in (WARN, FAIL)]

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
                "stages": to_dict(verdicts),
                "control": is_control,
            }
        )
        detail = {
            "id": task["id"],
            "ok": ok,
            "control": is_control,
            "prompt": task.get("prompt", ""),
            "seed_files": task.get("seed_files") or {},
            "reference_answer": task.get("reference_answer", ""),
            "reference_note": task.get("reference_note", ""),
            "answer": result.answer,
            "failures": failures,
            "stages": to_dict(verdicts),
            "steps": summary["steps"],
            "tools": summary["tool_calls"],
            "tokens": summary["total_tokens"],
            "wall_s": wall,
        }
        if not ok and getattr(args, "judge", True):
            diag = diagnose_failure(judge_llm, task, result.answer, failures, events)
            detail["diagnosis"] = diag.__dict__
            print(f"      ⟲ 归因[{diag.label}]: {diag.root_cause[:70]}")
            if diag.suggestion:
                print(f"        建议: {diag.suggestion[:90]}")

        flag = "🧪" if is_control else ("✅" if ok else "❌")
        print(f"{flag} {task['id']:<22} 步 {summary['steps']:>2}  "
              f"工具 {summary['tool_calls']:>2}  失败 {summary['tool_failed']:>2}  "
              f"token {summary['total_tokens']:>6}  {wall:>6.1f}s  {result.status}")
        for f in failures:
            print(f"      └ {f}")
        if issues:
            for v in issues:
                mark = "△" if v.level == WARN else "✗"
                print(f"      {mark} {v.name}[{v.label}]: {v.detail}")
                if v.evidence:
                    print(f"         证据: {v.evidence}")
        else:
            applicable = sum(1 for v in verdicts if v.level != NA)
            print(f"      ✓ {applicable} 个适用阶段全部 PASS")
        details.append(detail)

    scored_rows = [r for r in rows if not r.get("control")]
    passed = sum(1 for r in scored_rows if r["ok"])
    total = len(scored_rows)
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
    agg["stage_summary"] = aggregate(all_verdicts)

    # ---- 质量分：最后才算，前两件事（找问题、给建议）都不该依赖它 ----
    score = run_score(all_verdicts, threshold=getattr(args, "threshold", 0.90))
    agg["score"] = score
    si = 0
    for row in details:
        if row.get("control"):
            row["score"] = None
        else:
            row["score"] = score["per_task"][si]["score"]
            si += 1

    error_types: dict[str, int] = {}
    for row in details:
        diag = row.get("diagnosis")
        if diag:
            key = diag.get("error_type") or "other"
            error_types[key] = error_types.get(key, 0) + 1

    report_path = ""
    if getattr(args, "report", True):
        report_data = {
            "meta": {
                "model": args.model,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tool_calls": agg["tool_calls"],
                "tokens": sum(r["tokens"] for r in rows),
            },
            "score": score,
            "stages": agg["stage_summary"],
            "rows": details,
            "error_types": error_types,
        }
        report_path = getattr(args, "report_out", "") or os.path.join(
            str(out_root), f"report_{int(time.time())}.html"
        )
        write_report(report_data, report_path)

    out = args.out or os.path.join(str(out_root), f"eval_{int(time.time())}.json")
    Path(out).write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 62}")
    print(f"成功率 {passed}/{total} = {agg['success_rate']:.1%}"
          f"   平均步数 {agg['avg_steps']}   平均 token {agg['avg_tokens']}")
    print(f"工具调用 {agg['tool_calls']} 次，失败 {agg['tool_failed']}"
          f"，无效率 {agg['invalid_tool_rate']:.1%}")

    print(f"\n{'阶段':<12}{'PASS':>6}{'WARN':>6}{'FAIL':>6}{'N/A':>6}{'通过率':>9}")
    print("-" * 46)
    for stage, name in STAGES:
        s = agg["stage_summary"][stage]
        rate = "—" if s["rate"] is None else f"{s['rate']:.0%}"
        print(f"{_pad(s['name'], 12)}{s['pass']:>6}{s['warn']:>6}{s['fail']:>6}"
              f"{s['skipped']:>6}{rate:>9}")

    issues = [
        (s["name"], note)
        for stage, _ in STAGES
        for s in [agg["stage_summary"][stage]]
        for note in s["notes"]
    ]
    if issues:
        print("\n未达标明细：")
        for name, note in issues:
            print(f"  · {name}: {note}")

    print(f"\n{'=' * 62}")
    gate = "✅ 可发布" if score["gate"] == "pass" else "❌ 未达及格线"
    print(f"质量分 {score['quality']:.3f} / 及格线 {score['threshold']:.2f}   {gate}")
    if score["drag_counter"]:
        print("主要失分环节：" + "、".join(f"{k}×{v}" for k, v in score["drag_counter"].items()))
    if error_types:
        print("失败类型：" + "、".join(f"{k}×{v}" for k, v in error_types.items()))
    print(f"结果已写入 {out}")
    if report_path:
        print(f"对照报告 {report_path}")
    return 0 if score["gate"] == "pass" else 1
