"""守卫：检查 prompt / 技能 / 题集是否在上一次评测之后被改过。

规则很硬：**改一个字、一个标点都算改**。这不是洁癖——
prompt 的改动没有小改动，任何一处措辞都会改变模型的输出分布，
而「我以为只是改了个标点」恰恰是最容易漏测的那一类。

用法：
    python scripts/prompt_guard.py --kind kb        # 检查垂直层
    python scripts/prompt_guard.py --kind runtime   # 检查通用运行时

退出码 0 = 与最近一次评测一致；1 = 有变更且未重跑。
可直接接进 pre-commit 或 CI。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from arlab import fingerprint as fp  # noqa: E402


def _latest_eval(runs_dir: Path) -> Path | None:
    files = sorted(glob.glob(str(runs_dir / "eval_*.json")), key=os.path.getmtime)
    return Path(files[-1]) if files else None


def _kb_fingerprint(model: str) -> tuple[dict, Path, str]:
    from apps.kb_agent.app import GOLD_PATH, SKILLS_DIR, make_agent

    agent = make_agent(model=model, base_url=None, max_steps=8)
    return (
        fp.collect(
            system_prompt=agent._system_prompt(),
            skills_dir=str(SKILLS_DIR),
            tasks_path=str(GOLD_PATH),
            model=model,
            runtime_params={"max_steps": 8, "max_tokens": 40_000, "max_seconds": 240.0},
        ),
        PROJECT_ROOT / "apps" / "kb_agent" / "runs",
        "./run.sh eval",
    )


def _runtime_fingerprint(model: str) -> tuple[dict, Path, str]:
    from arlab import Agent, LLM, Registry, SkillLoader
    from arlab.tools.builtin import build_tools

    skills_dir = PROJECT_ROOT / "skills"
    skills = SkillLoader(str(skills_dir)).discover()
    registry = Registry(allow={"read"})
    for tool in build_tools(str(PROJECT_ROOT / "workspace"), skill_loader=skills):
        registry.register(tool)
    agent = Agent(LLM(model=model), registry, skills=skills)
    return (
        fp.collect(
            system_prompt=agent._system_prompt(),
            skills_dir=str(skills_dir),
            tasks_path=str(PROJECT_ROOT / "eval" / "tasks.json"),
            model=model,
            runtime_params={"max_steps": 12, "max_tokens": 120_000, "max_seconds": 180.0},
        ),
        PROJECT_ROOT / "runs",
        "./run.sh eval",
    )


def main(argv: list[str] | None = None) -> int:
    from arlab.llm import DEFAULT_MODEL

    p = argparse.ArgumentParser(description="prompt 变更守卫")
    p.add_argument("--kind", choices=["kb", "runtime"], default="kb")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args(argv)

    build = _kb_fingerprint if args.kind == "kb" else _runtime_fingerprint
    current, runs_dir, cmd = build(args.model)
    latest = _latest_eval(runs_dir)

    if latest is None:
        print(f"✗ {runs_dir} 下没有任何评测记录。先跑一次：{cmd}")
        return 1

    data = json.loads(latest.read_text(encoding="utf-8"))
    saved = data.get("fingerprint")
    score = (data.get("score") or {}).get("quality")
    gate = (data.get("score") or {}).get("gate", "?")

    print(f"最近一次评测: {latest.name}")
    print(f"  质量分 {score}  闸门 {gate}  记录指纹 {(saved or {}).get('combined', '无')}")
    print(f"  当前指纹   {current['combined']}")

    if not saved:
        print("\n⚠ 那次评测没有记录指纹（旧版本跑的），无法判断是否过期。建议重跑一次建立基线。")
        return 1

    changes = fp.diff(saved, current)
    if not changes:
        print("\n✅ prompt / 技能 / 题集与最近一次评测完全一致，无需重跑。")
        return 0

    print("\n⚠ 检测到变更，上次的分数已经失效：")
    for c in changes:
        print(f"  · {c}")
    print(f"\n请重跑：{cmd}")
    print("（改一个字、一个标点也要重跑。上一版的分数不能代表这一版。）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
