"""技能系统：SKILL.md 的发现、解析与渐进式披露。

为什么要渐进披露：技能的正文动辄几千 token，几十个技能全塞进 system prompt
既贵又会稀释注意力。所以只把 name + description 放进目录，模型判断需要时再
用 load_skill 拉正文。这是 Anthropic Agent Skills 的核心机制，也是「skill 体系」
这句话在工程上的真正含义。

frontmatter 解析是手写的：只要支持 `key: value` 和 `key: |` 就够了，不值得为此
引入 PyYAML，项目保持零第三方依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """返回 (frontmatter 字典, 正文)。没有 frontmatter 时返回 ({}, 全文)。"""
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    meta: dict[str, str] = {}
    end = None
    key: str | None = None
    buf: list[str] = []

    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.strip() == "---":
            end = idx
            break
        if line.startswith((" ", "\t")) and key:
            buf.append(line.strip())
            continue
        if ":" in line:
            if key:
                meta[key] = " ".join(buf).strip()
                buf = []
            k, _, v = line.partition(":")
            key = k.strip()
            v = v.strip()
            if v in ("|", ">"):
                buf = []
            else:
                buf = [v]
    if key and end is not None:
        meta[key] = " ".join(buf).strip()

    if end is None:
        return {}, text
    return meta, "\n".join(lines[end + 1 :]).strip()


@dataclass
class SkillLoader:
    skills_dir: str
    skills: dict[str, Skill] = field(default_factory=dict)

    def discover(self) -> "SkillLoader":
        root = Path(self.skills_dir)
        self.skills.clear()
        if not root.is_dir():
            return self
        for md in sorted(root.glob("*/SKILL.md")):
            text = md.read_text(encoding="utf-8", errors="replace")
            meta, body = _parse_frontmatter(text)
            name = meta.get("name") or md.parent.name
            if not meta.get("description"):
                # description 决定模型会不会去加载，缺了就等于这技能不存在
                continue
            self.skills[name] = Skill(
                name=name,
                description=meta["description"],
                body=body,
                path=md,
            )
        return self

    def inline_body(self) -> str:
        """把全部技能正文拼成一段，直接塞进 system prompt（不走 load_skill）。

        为什么需要这个：A/B 实验里模型只加载了 40/53 次技能，
        所以前一轮实验测的是「技能机制有没有用」，不是「技能内容有没有用」。
        内联之后内容 100% 生效，才真正测得到内容的价值——
        代价是每一步都要为这段文字付 prompt token。
        """
        if not self.skills:
            return ""
        parts = ["## 操作手册（已内置，无需加载）"]
        for skill in self.skills.values():
            parts.append(f"### {skill.name}\n\n{skill.body.strip()}")
        return "\n\n".join(parts)

    def catalog(self) -> str:
        """给 system prompt 用的目录：只有名字和一句话描述。"""
        if not self.skills:
            return "(当前没有可用技能)"
        return "\n".join(
            f"- {s.name}: {s.description}" for s in sorted(self.skills.values(), key=lambda x: x.name)
        )

    def load_body(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill is None:
            raise KeyError(f"没有名为 {name!r} 的技能。可用：{', '.join(sorted(self.skills))}")
        return f"# 技能 {skill.name}\n\n{skill.body}"
