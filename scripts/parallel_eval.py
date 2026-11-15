#!/usr/bin/env python3
"""并行跑评测：把题集切片，起 N 个进程，合并结果。

为什么用进程而不是线程：
kb_agent 里 embedding 模型（约 650MB）和 Chroma 是进程内单例，
多线程共享它们有线程安全风险（SQLite 连接、torch 推理）。
进程隔离最省心，代价是每个进程各加载一份模型。

本机（WSL 6G 上限）单个进程峰值 650MB，可用 2G，所以默认 3 路并发。
53 条串行约 40 分钟，3 路约 15 分钟。

用法：
  python3 scripts/parallel_eval.py --tasks apps/kb_agent/eval/silver.json --jobs 3
  python3 scripts/parallel_eval.py --tasks eval/gold_multihop.json --jobs 3 --out runs/mh_par.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 必须用 kb-lab 的 venv：
# 系统 python 3.14 对 argparse help 里的 % 严格校验会直接报错；
# arlab 的 venv 是零依赖的（跑纯 runtime），没有 jieba / chroma / sentence-transformers。
# kb_agent 的 run.sh 用的就是 ~/venvs/kb-lab。
PYTHON = os.environ.get("ARLAB_PYTHON") or str(Path.home() / "venvs" / "kb-lab" / "bin" / "python")
if not Path(PYTHON).exists():
    PYTHON = sys.executable
sys.path.insert(0, str(ROOT / "src"))

from arlab.scoring import run_score          # noqa: E402
from arlab.stages import Verdict, aggregate  # noqa: E402


def split(items: list, n: int) -> list[list]:
    """按轮转切片，让每片难度更均匀（不按顺序切，避免一片全是难题）。"""
    return [items[i::n] for i in range(n)]


def rebuild_verdicts(rows: list[dict]) -> list[list[Verdict]]:
    """从 rows 里存的 stages 重建 Verdict，用来重算加权分。"""
    out = []
    for row in rows:
        if row.get("control"):
            continue
        vs = [
            Verdict(stage=s["stage"], name=s["name"], level=s["level"],
                    detail=s.get("detail", ""), evidence=s.get("evidence", ""))
            for s in row.get("stages") or []
        ]
        out.append(vs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="并行跑评测")
    ap.add_argument("--tasks", required=True, help="题集 json 路径")
    ap.add_argument("--jobs", type=int, default=3, help="并发进程数（本机建议 ≤3，每进程约 650MB）")
    ap.add_argument("--out", default="", help="合并结果输出路径")
    ap.add_argument("--allow", default="read", help="工具权限，默认 read")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--max-seconds", type=float, default=240.0)
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    tasks_path = Path(args.tasks)
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    jobs = max(1, min(args.jobs, len(tasks)))
    chunks = split(tasks, jobs)
    print(f"题集 {len(tasks)} 条，切成 {jobs} 片：{[len(c) for c in chunks]}")

    tmpdir = Path(tempfile.mkdtemp(prefix="pareval_"))
    procs = []
    started = time.time()
    for i, chunk in enumerate(chunks, 1):
        tf = tmpdir / f"tasks_{i}.json"
        of = tmpdir / f"out_{i}.json"
        tf.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
        # 必须走 kb_agent 的入口：它才注册了检索工具集和 6 个业务谓词。
        # arlab 的 CLI 只带安全谓词，跑 kb_agent 的题会全部报「未注册的谓词」。
        cmd = [
            PYTHON, "-m", "apps.kb_agent.app", "eval",
            # kb_agent 的 eval 没有 --allow/--max-tokens/--max-seconds：
            # 权限固定 read，token 预算由 app.py 的 BUDGET_TOKENS 统一管理。
            "--tasks", str(tf), "--out", str(of),
            "--no-judge", "--no-report",
            "--max-steps", str(args.max_steps),
        ]
        # kb_lab（检索层）在另一个仓库，run.sh 里也是这么拼的
        kb_lab = Path(os.environ.get("KB_LAB_ROOT", "/mnt/d/kb-lab"))
        env = dict(os.environ,
                   PYTHONPATH=f"{ROOT / 'src'}:{ROOT}:{kb_lab / 'src'}",
                   KB_LAB_ROOT=str(kb_lab))
        # 不设这两个，sentence-transformers 会去查 HF Hub 的 revision，卡 7 分钟
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)
        # run.sh 会自动读这个文件，我们绕过 run.sh 直接起进程，就得自己读
        key_file = Path.home() / ".hermes" / ".arlab-key"
        if not env.get("ARLAB_API_KEY") and key_file.exists():
            env["ARLAB_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
        log = open(tmpdir / f"log_{i}.txt", "w")
        procs.append((i, subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                          stdout=log, stderr=subprocess.STDOUT), log, of))
        print(f"  片 {i} 启动：{len(chunk)} 条 → {of.name}")

    rows: list[dict] = []
    failed_chunks = []
    for i, p, log, of in procs:
        code = p.wait()
        log.close()
        print(f"  片 {i} 结束：退出码 {code}")
        if code != 0 or not of.exists():
            failed_chunks.append(i)
            print(f"    ⚠️ 片段失败，日志：{tmpdir / f'log_{i}.txt'}")
            continue
        rows.extend(json.loads(of.read_text(encoding="utf-8"))["rows"])

    elapsed = round(time.time() - started, 1)
    if not rows:
        print("没有任何结果，全部片段失败")
        return 1

    # 按原题集顺序排回来，便于和串行结果对照
    order = {t["id"]: i for i, t in enumerate(tasks)}
    rows.sort(key=lambda r: order.get(r["id"], 999))

    scored = [r for r in rows if not r.get("control")]
    verdicts = rebuild_verdicts(rows)
    score = run_score(verdicts, threshold=args.threshold)
    passed = sum(1 for r in scored if r.get("ok"))
    total = len(scored)

    merged = {
        "model": "merged",
        "tasks": total,
        "passed": passed,
        "success_rate": round(passed / total, 3) if total else 0.0,
        "avg_steps": round(sum(r["steps"] for r in rows) / len(rows), 2),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
        "tool_calls": sum(r["tools"] for r in rows),
        "tool_failed": sum(r["tool_failed"] for r in rows),
        "rows": rows,
        "stage_summary": aggregate(verdicts),
        "score": score,
        "parallel": {"jobs": jobs, "wall_s": elapsed, "failed_chunks": failed_chunks},
    }
    out = args.out or str(ROOT / "runs" / f"parallel_{int(time.time())}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 56}")
    print(f"合并完成：{passed}/{total} 通过，质量分 {score['quality']:.3f}，"
          f"闸门 {score['gate']}")
    print(f"并发 {jobs} 路，总耗时 {elapsed}s（若串行约 {elapsed * jobs:.0f}s）")
    print(f"结果写入 {out}")
    if failed_chunks:
        print(f"⚠️ 失败的片段：{failed_chunks}")
        return 1
    return 0 if score["gate"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())