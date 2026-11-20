"""按实际分布校准 stage_limits，而不是拍脑袋定阈值。

校准口径：取历史 run 的 P90 作为上限。含义是「最贵的 10% 值得看一眼」，
而不是「做到 P90 就算优秀」。阈值是告警线，不是目标值，两者不能混。

用法：
    python scripts/calibrate_limits.py --tasks apps/kb_agent/eval/silver.json \
        --run apps/kb_agent/runs/eval_xxx.json --percentile 0.9 --dry-run
    python scripts/calibrate_limits.py --tasks apps/kb_agent/eval/gold.json \
        --run runs/gold_a_par.json runs/gold_b_par.json --by qtype --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def load_rows(paths: list[str], pool_runs: bool = False) -> list[dict]:
    rows_by_id: dict[str, dict] = {}
    pooled_rows: list[dict] = []
    for path in paths:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        file_rows: dict[str, dict] = {}
        for row in run.get("rows") or []:
            if row.get("control") or not row.get("id"):
                continue
            if pool_runs:
                file_rows.setdefault(row["id"], row)
            else:
                rows_by_id.setdefault(row["id"], row)
        pooled_rows.extend(file_rows.values())
    return pooled_rows if pool_runs else list(rows_by_id.values())


def group_rows_by_qtype(
    tasks: list[dict], rows: list[dict], min_samples: int = 5
) -> tuple[dict[str, list[dict]], dict[str, str], dict[str, str]]:
    rows_by_id: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_id.setdefault(row["id"], []).append(row)
    groups: dict[str, list[dict]] = {}
    for task in tasks:
        matched = [r for r in rows_by_id.get(task["id"], []) if r.get("tokens")]
        if not matched:
            continue
        qtype = task.get("qtype") or "unknown"
        groups.setdefault(qtype, []).extend(matched)

    large = {
        name: members for name, members in groups.items()
        if len({r["id"] for r in members}) >= min_samples
    }
    small = {
        name: members for name, members in groups.items()
        if len({r["id"] for r in members}) < min_samples
    }
    if not small:
        return groups, {}, {}
    if not large:
        all_rows = [row for members in groups.values() for row in members]
        return {"all": all_rows}, {name: "all" for name in groups}, {
            name: "all" for name in small
        }

    aliases: dict[str, str] = {}
    merges: dict[str, str] = {}
    medians = {
        name: pct([r["tokens"] for r in members], 0.5)
        for name, members in large.items()
    }
    for name, members in small.items():
        median = pct([r["tokens"] for r in members], 0.5)
        target = min(medians, key=lambda candidate: abs(medians[candidate] - median))
        large[target].extend(members)
        aliases[name] = target
        merges[name] = target
    return large, aliases, merges


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="按实际分布校准阶段上限")
    p.add_argument("--tasks", required=True, help="要改的题集 json")
    p.add_argument("--run", required=True, nargs="+", help="历史评测结果 json，可多个")
    p.add_argument("--percentile", type=float, default=0.9)
    p.add_argument("--by", choices=["qtype"], help="按题型分别校准")
    p.add_argument("--pool-runs", action="store_true",
                   help="保留不同 run 的重复测量；默认按 id 去重用于合并分片")
    p.add_argument("--dry-run", action="store_true", help="只打印不写回")
    args = p.parse_args(argv)

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    rows = load_rows(args.run, pool_runs=args.pool_runs)
    tokens = [r["tokens"] for r in rows if r.get("tokens")]
    if not tokens:
        print("这些 run 没有 token 数据，无法校准")
        return 1

    if args.by == "qtype":
        groups, aliases, merges = group_rows_by_qtype(tasks, rows)
    else:
        groups, aliases, merges = {"all": rows}, {}, {}
    if not groups:
        print("题集与历史 run 没有可匹配的任务")
        return 1

    old_tokens = {(t.get("stage_limits") or {}).get("tokens") for t in tasks}
    old_steps = {(t.get("stage_limits") or {}).get("steps") for t in tasks}
    print(f"样本 {len(tokens)} 条；实际 token 中位数 {pct(tokens, 0.5)}")
    print(f"原上限: tokens={sorted(x for x in old_tokens if x)} "
          f"steps={sorted(x for x in old_steps if x)}")
    if merges:
        print("小样本合并: " + ", ".join(f"{src} → {dst}" for src, dst in sorted(merges.items())))

    limits_by_group: dict[str, tuple[int, int]] = {}
    for name, members in sorted(groups.items()):
        group_tokens = [r["tokens"] for r in members]
        group_steps = [r["steps"] for r in members if r.get("steps") is not None]
        new_tokens = pct(group_tokens, args.percentile)
        new_steps = pct(group_steps, args.percentile)
        limits_by_group[name] = (new_tokens, new_steps)
        over = sum(1 for value in group_tokens if value > new_tokens)
        print(f"[{name}] 样本 {len(members)} 条")
        print(f"  token: 中位 {pct(group_tokens, 0.5)}  "
              f"P{int(args.percentile * 100)} {new_tokens}  最大 {max(group_tokens)}")
        print(f"  steps: 中位 {pct(group_steps, 0.5)}  "
              f"P{int(args.percentile * 100)} {new_steps}  最大 {max(group_steps)}")
        print(f"  新上限下仍超标 {over} 条 = {over / len(group_tokens):.0%}")

    if args.dry_run:
        print("\n[dry-run] 未写回")
        return 0

    written = 0
    for task in tasks:
        qtype = task.get("qtype") or "unknown"
        group = aliases.get(qtype, qtype) if args.by == "qtype" else "all"
        if group not in limits_by_group:
            continue
        new_tokens, new_steps = limits_by_group[group]
        limits = task.setdefault("stage_limits", {})
        limits["tokens"] = new_tokens
        limits["steps"] = new_steps
        written += 1
    Path(args.tasks).write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写回 {args.tasks}（{written} 条）")
    print("注意：阈值调高会让执行效率阶段变好看，报告里必须同时显示实际 token 中位数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())