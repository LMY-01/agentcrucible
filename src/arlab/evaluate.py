"""评测：跑一批任务，按断言判分，汇总成一张能贴进 README 的表。

判分不依赖模型当裁判——每条任务自己声明期望（文件里该出现什么、答案该含什么
关键词），跑完对着工作区检查。这样分数是可复现的，改一版 prompt 或压缩策略再来
一次，涨跌就是真涨跌。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from . import fingerprint as fp
from .budget import Budget
from .faults import FaultInjector
from .llm import LLM
from .loop import Agent
from .jsonx import dig as _dig
from .jsonx import extract_json as _extract_json
from .judge import diagnose as diagnose_failure
from .report import write as write_report
from .pricing import human as money
from .pricing import load as load_pricing
from .scoring import run_score
from .trace import pct
from .skills import SkillLoader
from .stages import FAIL, NA, STAGES, WARN, aggregate, judge, load_events, to_dict
from .tools.builtin import build_tools
from .tools.registry import Registry
from .trace import Trace, new_run_dir


def _pad(text: str, width: int) -> str:
    """按显示宽度左对齐：中文和全角符号占两列，用 len() 对齐会错位。"""
    shown = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    return text + " " * max(0, width - shown)


# _extract_json / _dig 已移到 arlab.jsonx（stages.py 也要用，不能成环）
# 这里保留同名导入，测试和 app.py 继续用 _extract_json 即可。


def _check(checks: list[dict], answer: str, workspace: Path,
           predicates: dict[str, Any] | None = None,
           context: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """逐条断言。predicates 是应用层注册的自定义谓词，返回 None=通过，返回字符串=失败原因。

    context 带运行时事实（status/steps/tokens），让谓词能区分「主动做对」和「被逼到墙角蒙对」。

    目前支持：answer_contains / answer_contains_any / json_field /
    file_exists / file_contains / predicate
    """
    failures: list[str] = []
    ctx = context or {}
    for c in checks:
        kind = c.get("type")
        if kind == "predicate":
            fn = (predicates or {}).get(c.get("name", ""))
            if fn is None:
                failures.append(f"未注册的谓词 {c.get('name')!r}")
                continue
            reason = fn(answer, c, ctx)
            if reason:
                failures.append(reason)
            continue
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


def _report_data(agg: dict, args, details=None, error_types=None) -> dict:
    """报告需要的全部现场。单轮和多轮共用，避免两份渲染逻辑走偏。"""
    return {
        "meta": {
            "model": args.model,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool_calls": agg.get("tool_calls", 0),
            "tokens": sum(r.get("tokens", 0) for r in agg.get("rows") or []),
            "repeat": agg.get("repeat", 1),
        },
        "score": agg["score"],
        "tier_scores": agg.get("tier_scores") or {},
        "latency": agg.get("latency") or {},
        "cost": agg.get("cost") or {},
        "fingerprint": agg.get("fingerprint"),
        "stages": agg["stage_summary"],
        "rows": details if details is not None else (agg.get("rows") or []),
        "error_types": error_types or {},
    }


def _merge_repeats(paths: list[str], args) -> dict:
    """把 N 轮结果合成一份：准确率给区间，延迟和成本合并样本。

    为什么准确率必须给区间：同一份 prompt 同一套题，实测能差 0.05 以上。
    报单点等于把噪声当结论。给出中位数和极差，读者自己判断「这算不算提升」。
    """
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    qualities = [r["score"]["quality"] for r in runs]
    passed_list = [r["passed"] for r in runs]
    all_rows = [row for r in runs for row in r["rows"] if not row.get("control")]
    walls = [x["wall_s"] for x in all_rows]
    ttfts = [x["ttft_ms"] for x in all_rows if x.get("ttft_ms")]
    total_cost = sum(r["cost"]["total"] for r in runs)
    n_task_runs = sum(r["tasks"] for r in runs) or 1

    merged = dict(runs[0])
    merged["repeat"] = len(runs)
    merged["rows"] = runs[0]["rows"]          # 明细展示第一轮，够看
    merged["tasks"] = runs[0]["tasks"]
    merged["passed"] = int(round(sum(passed_list) / len(passed_list)))
    merged["success_rate"] = round(merged["passed"] / merged["tasks"], 3) if merged["tasks"] else 0.0

    q_sorted = sorted(qualities)
    med = q_sorted[len(q_sorted) // 2] if len(q_sorted) % 2 else \
        round((q_sorted[len(q_sorted) // 2 - 1] + q_sorted[len(q_sorted) // 2]) / 2, 3)
    merged["score"] = {
        **runs[0]["score"],
        "quality": med,
        "quality_range": [min(qualities), max(qualities)],
        "quality_all": qualities,
        "pass_all": passed_list,
        "gate": "pass" if med >= runs[0]["score"]["threshold"] else "fail",
    }
    merged["latency"] = {
        "ttft_ms_p50": pct(ttfts, 0.5),
        "ttft_ms_p95": pct(ttfts, 0.95),
        "wall_s_p50": round(pct(walls, 0.5), 1),
        "wall_s_p95": round(pct(walls, 0.95), 1),
        "wall_s_p99": round(pct(walls, 0.99), 1),
        "wall_s_avg": round(sum(walls) / len(walls), 1) if walls else 0.0,
        "wall_s_max": round(max(walls), 1) if walls else 0.0,
        "samples": len(walls),
        "note": f"{len(runs)} 轮合并采样，共 {len(walls)} 个任务样本",
    }
    merged["cost"] = {
        **runs[0]["cost"],
        "total": round(total_cost, 8),
        "per_task": round(total_cost / n_task_runs, 8),
    }

    # 分档分数也取中位。否则报告顶部是「三轮总分的中位」，下面的分档却是第一轮的，
    # 两个口径混在一张表里，读者会以为白银集那行也代表三轮。
    tiers: dict[str, dict] = {}
    for tier in (runs[0].get("tier_scores") or {}):
        qs = [r["tier_scores"][tier]["quality"] for r in runs
              if tier in (r.get("tier_scores") or {})]
        if qs:
            tiers[tier] = {
                **runs[0]["tier_scores"][tier],
                "quality": round(statistics.median(qs), 3),
                "quality_all": qs,
            }
    if tiers:
        merged["tier_scores"] = tiers
    return merged


def run_suite(args, tool_builder=None, predicates=None) -> int:
    # 多轮：递归调自己 N 次（每轮独立 json），再合并。
    # 不改主流程结构，避免把已经验证过的评测逻辑拆散。
    repeat = int(getattr(args, "repeat", 1) or 1)
    if repeat > 1:
        stamp = int(time.time())
        outs: list[str] = []
        for r in range(repeat):
            sub = argparse.Namespace(**vars(args))
            sub.repeat = 1
            sub.report = False
            sub.out = os.path.join(str(args.runs), f"rep_{stamp}_{r}.json")
            print(f"\n{'#' * 62}\n# 第 {r + 1}/{repeat} 轮\n{'#' * 62}")
            run_suite(sub, tool_builder, predicates)
            if Path(sub.out).exists():
                outs.append(sub.out)
        if not outs:
            print("没有拿到任何一轮结果")
            return 1
        merged = _merge_repeats(outs, args)
        agg = merged
        score = agg["score"]
        lat, cst = agg["latency"], agg["cost"]
        print(f"\n{'=' * 62}\n{agg['repeat']} 轮汇总")
        print(f"  准确率  质量分中位 {score['quality']:.3f}"
              f"  区间 [{score['quality_range'][0]:.3f}, {score['quality_range'][1]:.3f}]"
              f"  各轮 {[round(q, 3) for q in score['quality_all']]}")
        print(f"  延迟    TTFT {lat['ttft_ms_p50'] / 1000:.1f}s(P50)"
              f"   端到端 {lat['wall_s_p50']}s(P50) {lat['wall_s_p99']}s(P99)"
              f"（{lat['samples']} 个样本）")
        if cst["priced"]:
            print(f"  成本    {money(cst['total'], cst['currency'])} 总计"
                  f" / {money(cst['per_task'], cst['currency'])} 每任务（均值）")
        else:
            print(f"  成本    —（未取到单价）")
        print(f"  闸门    {'✅ 可发布' if score['gate'] == 'pass' else '❌ 未达及格线'}")

        out = args.out or os.path.join(str(args.runs), f"eval_{stamp}.json")
        Path(out).write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path = ""
        if getattr(args, "report", True):
            report_path = getattr(args, "report_out", "") or os.path.join(
                str(args.runs), f"report_{stamp}.html"
            )
            write_report(_report_data(agg, args), report_path)
        print(f"\n合并结果 {out}")
        if report_path:
            print(f"对照报告 {report_path}")
        return 0 if score["gate"] == "pass" else 1

    # 支持多份题集：--tasks gold.json,silver.json —— 两份一起跑，分账统计
    task_files = [Path(p.strip()) for p in str(args.tasks).split(",") if p.strip()]
    tasks: list[dict] = []
    for tf in task_files:
        if not tf.exists():
            print(f"题集不存在: {tf}")
            return 1
        tasks.extend(json.loads(tf.read_text(encoding="utf-8")))
    if args.limit:
        tasks = tasks[: args.limit]

    allow = {"read"} | {p.strip() for p in (args.allow or "").split(",") if p.strip()}
    skills = SkillLoader(args.skills).discover() if args.skills else None

    out_root = Path(args.runs)
    rows = []
    all_verdicts = []
    verdicts_by_tier: dict[str, list] = {}
    fingerprint = None
    details = []  # 给报告用的完整现场，包含输入材料和标准答案
    judge_llm = LLM(base_url=args.base_url, model=getattr(args, "judge_model", None) or args.model)
    price = load_pricing(args.model, judge_llm.base_url, judge_llm.api_key)
    print(f"评测 {len(tasks)} 条任务，模型 {args.model}，权限 {sorted(allow)}")
    print(f"计价 {'已启用' if price.known else '未取到单价（成本将显示为 —）'}"
          f"  prompt {price.prompt_per_token:.8f}/tok  completion {price.completion_per_token:.8f}/tok\n")

    for idx, task in enumerate(tasks, 1):
        workspace = Path(args.workspace) / task.get("workspace", "sandbox")
        workspace.mkdir(parents=True, exist_ok=True)
        for fname, content in (task.get("seed_files") or {}).items():
            target = workspace / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        registry = Registry(allow=allow)
        builder = tool_builder or (lambda ws, sk: build_tools(ws, skill_loader=sk))
        for tool in builder(str(workspace), skills):
            registry.register(tool)
        # 故障注入：挂在 registry 入口，对工具实现零侵入
        registry.injector = FaultInjector.from_spec(task.get("inject_fault"))

        budget = Budget(
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            max_seconds=args.max_seconds,
        )
        run_dir = new_run_dir(str(out_root), tag=f"eval-{task['id']}")
        trace = Trace(run_dir, meta={"task_id": task["id"], "prompt": task["prompt"],
                                     "model": args.model, "budget": budget.snapshot()})
        if registry.injector is not None:
            trace.emit("fault_injection", spec=registry.injector.describe())
        agent = Agent(LLM(base_url=args.base_url, model=args.model), registry,
                      skills=skills, budget=budget, trace=trace, pricing=price)

        # 输入指纹：记录「这次跑的是哪一版 prompt / 技能 / 题集」
        if fingerprint is None:
            fingerprint = fp.collect(
                system_prompt=agent._system_prompt(),
                skills_dir=args.skills,
                tasks_path=args.tasks,
                model=args.model,
                runtime_params={"max_steps": args.max_steps,
                                "max_tokens": args.max_tokens,
                                "max_seconds": args.max_seconds},
            )
            trace.emit("fingerprint", **fingerprint)

        started = time.time()
        result = agent.run(task["prompt"])
        wall = round(time.time() - started, 2)
        summary = trace.summarize()
        ok, failures = _check(
            task.get("checks", []), result.answer, workspace, predicates,
            context={"status": result.status, "reason": result.reason,
                     "steps": summary["steps"], "tokens": summary["total_tokens"]},
        )
        trace.close(ok=ok, failures=failures, answer=result.answer,
                    status=result.status, reason=result.reason, **summary)

        # 端到端判完再回看 trace，做阶段级判定
        events = load_events(trace.path)
        verdicts = judge(events, task, ctx={
            "status": result.status,
            "reason": result.reason,
            "steps": summary["steps"],
            "tokens": summary["total_tokens"],
            "available_tools": registry.names(),
            "fault_injected": registry.injector is not None,
        })
        is_control = bool(task.get("control"))
        if not is_control:
            all_verdicts.append(verdicts)  # 对照组不参与任何统计
            verdicts_by_tier.setdefault(task.get("tier", "gold"), []).append(verdicts)
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
                "ttft_ms": summary["ttft_ms_p50"],
                "cost": summary["cost"],
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
    agg["fingerprint"] = fingerprint

    # ---- 三硬指标：准确率 / 延迟 / 成本 ----
    walls = [r["wall_s"] for r in scored_rows]
    ttfts = [r["ttft_ms"] for r in scored_rows if r.get("ttft_ms")]
    total_cost = sum(r.get("cost", 0.0) for r in scored_rows)
    agg["latency"] = {
        "ttft_ms_p50": pct(ttfts, 0.5),
        "ttft_ms_p95": pct(ttfts, 0.95),
        "wall_s_p50": round(pct(walls, 0.5), 1),
        "wall_s_p95": round(pct(walls, 0.95), 1),
        "wall_s_p99": round(pct(walls, 0.99), 1),
        "wall_s_avg": round(sum(walls) / len(walls), 1) if walls else 0.0,
        "wall_s_max": round(max(walls), 1) if walls else 0.0,
        "note": "任务级百分位；严格 P99 需 --repeat 多轮采样",
    }
    agg["cost"] = {
        "total": round(total_cost, 8),
        "per_task": round(total_cost / len(scored_rows), 8) if scored_rows else 0.0,
        "currency": price.currency if price else "USD",
        "priced": bool(price and price.known),
        "prompt_tokens": sum(r["tokens"] for r in scored_rows),
    }

    # ---- 质量分：最后才算，前两件事（找问题、给建议）都不该依赖它 ----
    score = run_score(all_verdicts, threshold=getattr(args, "threshold", 0.90))
    agg["score"] = score

    # 分档统计：黄金集定验收线，白银集看回归涨跌，两者不混算
    tier_scores = {
        t: run_score(v, threshold=getattr(args, "threshold", 0.90))
        for t, v in verdicts_by_tier.items()
    }
    agg["tier_scores"] = {
        t: {**{k: v for k, v in s.items() if k != "per_task"},
            "tasks": len(verdicts_by_tier[t])}
        for t, s in tier_scores.items()
    }
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

    agg["_details"] = details
    agg["_error_types"] = error_types

    report_path = ""
    if getattr(args, "report", True):
        report_path = getattr(args, "report_out", "") or os.path.join(
            str(out_root), f"report_{int(time.time())}.html"
        )
        write_report(_report_data(agg, args, details, error_types), report_path)
    agg.pop("_details", None)
    agg.pop("_error_types", None)

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
    if len(tier_scores) > 1:
        names = {"gold": "黄金集·验收", "silver": "白银集·回归"}
        print(f"{_pad('分档', 14)}{'条数':>5}{'质量分':>9}{'闸门':>8}")
        for tier, sc in tier_scores.items():
            n = len(verdicts_by_tier[tier])
            flag = "✅" if sc["gate"] == "pass" else "❌"
            print(f"{_pad(names.get(tier, tier), 14)}{n:>5}{sc['quality']:>9.3f}{flag:>8}")
        print("-" * 62)
    gate = "✅ 可发布" if score["gate"] == "pass" else "❌ 未达及格线"
    lat, cst = agg["latency"], agg["cost"]
    print(f"\n{'三硬指标':<8}{'—' * 52}")
    print(f"  准确率  质量分 {score['quality']:.3f}   通过 {passed}/{total}"
          f"   闸门 {score['gate']}")
    print(f"  延迟    TTFT {lat['ttft_ms_p50'] / 1000:.1f}s(P50) {lat['ttft_ms_p95'] / 1000:.1f}s(P95)"
          f"   端到端 {lat['wall_s_p50']}s(P50) {lat['wall_s_p99']}s(P99) 最慢 {lat['wall_s_max']}s")
    if cst["priced"]:
        print(f"  成本    {money(cst['total'], cst['currency'])} 总计"
              f" / {money(cst['per_task'], cst['currency'])} 每任务"
              f"   共 {cst['prompt_tokens']} token")
    else:
        print(f"  成本    —（未取到单价）共 {cst['prompt_tokens']} token")
    print(f"\n质量分 {score['quality']:.3f} / 及格线 {score['threshold']:.2f}   {gate}")
    if score["drag_counter"]:
        print("主要失分环节：" + "、".join(f"{k}×{v}" for k, v in score["drag_counter"].items()))
    if error_types:
        print("失败类型：" + "、".join(f"{k}×{v}" for k, v in error_types.items()))
    print(f"结果已写入 {out}")
    if report_path:
        print(f"对照报告 {report_path}")
    return 0 if score["gate"] == "pass" else 1
