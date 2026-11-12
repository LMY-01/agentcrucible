"""聚合多次 eval 结果，看阶段判定的稳定性和抖动来源。

单次 run 的阶段表只能说明"这次哪一环掉了"，回答不了"这一环是不是真的不稳"。
把同一套任务跑三遍按阶段对齐，就能分出两类问题：

  * 稳定 fail → 设计或提示词的问题，改了之后应该看到级别整体上移
  * 抖动（同一任务在 pass / warn / fail 之间来回）→ 模型侧的不确定性，
    改提示词收益有限，要靠约束更强的运行时机制（例如禁掉盲猜路径）

用法：
    python scripts/analyze_runs.py runs/stage_run_*.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

STAGES = ["explore", "tool_valid", "args", "recovery",
          "skill_recall", "efficiency", "termination", "answer"]
NAMES = {"explore": "探查", "tool_valid": "工具有效性", "args": "参数合法性",
         "recovery": "错误恢复", "skill_recall": "技能召回", "efficiency": "执行效率",
         "termination": "终止决策", "answer": "答案正确"}
ORDER = {"pass": 0, "warn": 1, "fail": 2, "na": -1}


def pad(text: str, width: int) -> str:
    shown = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    return text + " " * max(0, width - shown)


def load(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        data["_file"] = p.split("/")[-1]
        out.append(data)
    return out


def main(paths: list[str]) -> int:
    if not paths:
        paths = sorted(glob.glob("runs/stage_run_*.json"))
    if not paths:
        print("没有找到结果文件。先跑：./run.sh eval --out runs/stage_run_1.json")
        return 1

    runs = load(paths)
    print(f"聚合 {len(runs)} 次 run：{', '.join(r['_file'] for r in runs)}")
    print(f"模型：{runs[0]['model']}\n")

    # task -> stage -> [level, ...]
    matrix: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for row in run["rows"]:
            for st in row.get("stages", []):
                matrix[row["id"]][st["stage"]].append(st["level"])

    task_ids = sorted(matrix)
    width = max(len(t) for t in task_ids) + 2

    print("=" * 78)
    print("一、每条任务 × 每个阶段，在 N 次 run 里的表现")
    print("=" * 78)
    header = pad("任务", width) + "".join(pad(NAMES[s], 13) for s in STAGES)
    print(header)
    print("-" * 78)
    for tid in task_ids:
        cells = []
        for s in STAGES:
            levels = matrix[tid][s]
            uniq = sorted(set(levels), key=lambda x: ORDER[x])
            if not levels:
                cells.append("—")
            elif len(uniq) == 1:
                cells.append(uniq[0].upper() if uniq[0] != "na" else "N/A")
            else:
                cells.append("/".join(l.upper()[0] for l in uniq) + " 抖动")
        print(pad(tid, width) + "".join(pad(c, 13) for c in cells))

    print("\n" + "=" * 78)
    print("二、按阶段汇总：哪些是稳定问题，哪些是抖动")
    print("=" * 78)
    print(pad("阶段", 14) + pad("PASS", 8) + pad("WARN", 8) + pad("FAIL", 8)
          + pad("N/A", 8) + "诊断")
    print("-" * 66)
    stable_fail, flaky, solid = [], [], []
    for s in STAGES:
        counts = defaultdict(int)
        for tid in task_ids:
            for lv in matrix[tid][s]:
                counts[lv] += 1
        applicable = counts["pass"] + counts["warn"] + counts["fail"]
        note = ""
        if applicable == 0:
            note = "全部任务不考察此阶段"
        else:
            jitter = [t for t in task_ids
                      if len({lv for lv in matrix[t][s] if lv != "na"}) > 1]
            if jitter:
                note = f"{len(jitter)} 条任务抖动：{', '.join(jitter)}"
                flaky.append(s)
            elif counts["fail"]:
                note = "稳定失败，改设计或提示词"
                stable_fail.append(s)
            elif counts["warn"]:
                note = "稳定告警，行为正确但有浪费"
            else:
                note = "稳定通过"
                solid.append(s)
        print(pad(NAMES[s], 14) + pad(str(counts["pass"]), 8) + pad(str(counts["warn"]), 8)
              + pad(str(counts["fail"]), 8) + pad(str(counts["na"]), 8) + note)

    print("\n" + "=" * 78)
    print("三、结论")
    print("=" * 78)
    print(f"  稳定通过：{', '.join(NAMES[s] for s in solid) or '无'}")
    print(f"  稳定失败：{', '.join(NAMES[s] for s in stable_fail) or '无'}")
    print(f"  存在抖动：{', '.join(NAMES[s] for s in flaky) or '无'}")
    print("\n  抖动阶段改提示词收益有限——运行时要能兜住，而不是指望模型每次都对。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
