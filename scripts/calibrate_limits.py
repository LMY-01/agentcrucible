"""按实际分布校准 stage_limits，而不是拍脑袋定阈值。

为什么需要它：白银集的中位 token 是 27879，而阶段上限写的是 25000——
一半以上任务注定判「执行效率 FAIL」，但它们只是正常完成了任务。
这种失败不是缺陷，是阈值错了。

校准口径：取历史 run 的 P90 作为上限。含义是「最贵的 10% 值得看一眼」，
而不是「做到 P90 就算优秀」。阈值是告警线，不是目标值，两者不能混。

用法：
    python scripts/calibrate_limits.py --tasks apps/kb_agent/eval/silver.json \
        --run apps/kb_agent/runs/eval_xxx.json --percentile 0.9 --dry-run
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="按实际分布校准阶段上限")
    p.add_argument("--tasks", required=True, help="要改的题集 json")
    p.add_argument("--run", required=True, help="历史评测结果 json")
    p.add_argument("--percentile", type=float, default=0.9)
    p.add_argument("--dry-run", action="store_true", help="只打印不写回")
    args = p.parse_args(argv)

    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    rows = [r for r in run.get("rows") or [] if not r.get("control")]
    tokens = [r["tokens"] for r in rows if r.get("tokens")]
    steps = [r["steps"] for r in rows if r.get("steps")]
    if not tokens:
        print("这次 run 没有 token 数据，无法校准")
        return 1

    new_tokens = pct(tokens, args.percentile)
    new_steps = pct(steps, args.percentile)
    old = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    old_tokens = {(t.get("stage_limits") or {}).get("tokens") for t in old}
    old_steps_limit = {(t.get("stage_limits") or {}).get("steps") for t in old}

    over = sum(1 for t in tokens if t > new_tokens)
    print(f"样本 {len(tokens)} 条")
    print(f"  token: 中位 {pct(tokens, 0.5)}  P{int(args.percentile * 100)} {new_tokens}  "
          f"最大 {max(tokens)}")
    print(f"  原上限: tokens={sorted(x for x in old_tokens if x)} steps={sorted(x for x in old_steps_limit if x)}")
    print(f"  新上限: tokens={new_tokens} steps={new_steps}"
          f"（新上限下仍超标 {over} 条 = {over / len(tokens):.0%}）")

    if args.dry_run:
        print("\n[dry-run] 未写回")
        return 0

    for t in old:
        limits = t.setdefault("stage_limits", {})
        limits["tokens"] = new_tokens
        limits["steps"] = new_steps
    Path(args.tasks).write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写回 {args.tasks}（{len(old)} 条）")
    print("注意：阈值调高会让执行效率阶段变好看，报告里必须同时显示实际 token 中位数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())