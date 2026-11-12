"""输入指纹：把「所有会影响模型行为的东西」压成可比较的哈希。

为什么需要它：prompt 改一个字、技能改一个标点、评测题改一个阈值，都足以让
上一次的分数失效。但这件事靠人记不住——改 prompt 的时候不会想起三个月前那个
0.923 是怎么来的。所以让机器每次评测都记一笔，改了就对比得出来。

设计取舍：
  * **不做任何归一化**。空白、标点、全角半角都算变更。用户的规则是「改一个字也要重跑」，
    归一化会让「只改了逗号」这种情况被静默放过。
  * **分层记录**。system / 每个 skill / tasks / model 分开存哈希，
    这样报警时能说清「是哪个文件的哪一项变了」，而不是只给一个不同的总哈希。
  * 只记录能影响输出的东西。`temperature`、`max_steps` 这类运行参数也算，
    因为改预算会直接改结果。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _skill_hashes(skills_dir: str | None) -> dict[str, str]:
    """按技能名记录正文哈希。整个 SKILL.md 参与计算，前端元数据也算。"""
    if not skills_dir:
        return {}
    root = Path(skills_dir)
    if not root.is_dir():
        return {}
    out: dict[str, str] = {}
    for md in sorted(root.glob("*/SKILL.md")):
        out[md.parent.name] = _h(md.read_text(encoding="utf-8", errors="replace"))
    return out


def _tasks_hash(tasks_path: str | None) -> str:
    """整份题集参与计算。没用解析后的结构，因为改注释、调顺序也算改测试。

    支持逗号分隔的多份题集（gold.json,silver.json），文件名一起进哈希，
    这样「把某条题从白银升到黄金」也算变更。
    """
    if not tasks_path:
        return ""
    parts: list[str] = []
    for raw in str(tasks_path).split(","):
        path = Path(raw.strip())
        if path.exists():
            parts.append(path.name + "\x1f" + path.read_text(encoding="utf-8", errors="replace"))
    return _h("\x1e".join(parts)) if parts else ""


def collect(
    *,
    system_prompt: str,
    skills_dir: str | None,
    tasks_path: str | None,
    model: str,
    runtime_params: dict | None = None,
) -> dict:
    """返回可直接存进 json 的指纹字典。"""
    skills = _skill_hashes(skills_dir)
    tasks = _tasks_hash(tasks_path)
    params = runtime_params or {}

    parts = [system_prompt, model, tasks, json.dumps(skills, sort_keys=True),
             json.dumps(params, sort_keys=True)]
    combined = _h("\x1f".join(parts))

    return {
        "combined": combined,
        "system": {"hash": _h(system_prompt), "chars": len(system_prompt)},
        "skills": skills,
        "tasks": tasks,
        "model": model,
        "runtime_params": params,
    }


def diff(old: dict, new: dict) -> list[str]:
    """列出两次指纹的差异，人话描述。一致返回空列表。"""
    if not old:
        return ["没有历史指纹可对比（这是第一次带指纹的评测）"]

    out: list[str] = []
    if old.get("model") != new.get("model"):
        out.append(f"模型变了: {old.get('model')} → {new.get('model')}")

    old_sys = (old.get("system") or {}).get("hash")
    new_sys = (new.get("system") or {}).get("hash")
    if old_sys != new_sys:
        oc = (old.get("system") or {}).get("chars", "?")
        nc = (new.get("system") or {}).get("chars", "?")
        out.append(f"system prompt 变了（{oc} → {nc} 字）")

    old_skills = old.get("skills") or {}
    new_skills = new.get("skills") or {}
    for name in sorted(set(old_skills) | set(new_skills)):
        if name not in new_skills:
            out.append(f"技能被删: {name}")
        elif name not in old_skills:
            out.append(f"新增技能: {name}")
        elif old_skills[name] != new_skills[name]:
            out.append(f"技能内容变了: {name}")

    if old.get("tasks") != new.get("tasks"):
        out.append("评测题集变了")

    old_p = old.get("runtime_params") or {}
    new_p = new.get("runtime_params") or {}
    for key in sorted(set(old_p) | set(new_p)):
        if old_p.get(key) != new_p.get(key):
            out.append(f"运行参数变了: {key} {old_p.get(key)} → {new_p.get(key)}")

    return out
