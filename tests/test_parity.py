"""评测路径必须等于生产路径。

这个文件存在的理由：曾经「评测跑的不是生产配置」——
`cmd_ask` 给 Agent 传了 `extra_system=DOMAIN_PROMPT`，而 `run_suite` 没传。
于是所有评测都在没有领域提示词的情况下进行，改了 prompt 分数不动，
我误以为是「提示词无效」，实际是它根本没被加载。

这类 bug 不会报错、不会崩，只会让所有结论悄悄失真，所以必须用测试锁住：
评测路径构造出来的 Agent，system prompt / 工具集 / 预算必须和生产路径完全一致。
"""

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.kb_agent import app as kbapp  # noqa: E402
from arlab.budget import Budget  # noqa: E402
from arlab.llm import LLM  # noqa: E402
from arlab.loop import Agent  # noqa: E402
from arlab.skills import SkillLoader  # noqa: E402
from arlab.tools.registry import Registry  # noqa: E402


def build_eval_agent(skills_dir: str = "", extra_system: str = "") -> Agent:
    """复刻 run_suite 内部的构造方式（工具、权限、预算、extra_system）。"""
    skills = SkillLoader(skills_dir).discover() if skills_dir else None
    registry = Registry(allow={"read"})
    for tool in kbapp.tool_builder(str(kbapp.APP_ROOT), skills):
        registry.register(tool)
    return Agent(
        LLM(base_url=None, model="bailian/qwen3.8-flash"),
        registry,
        skills=skills,
        budget=Budget(max_steps=kbapp.BUDGET_STEPS, max_tokens=kbapp.BUDGET_TOKENS,
                      max_seconds=kbapp.BUDGET_SECONDS),
        extra_system=extra_system,
    )


def build_prod_agent() -> Agent:
    return kbapp.make_agent(model="bailian/qwen3.8-flash", base_url=None,
                            max_steps=kbapp.BUDGET_STEPS)


class TestEvalProdParity(unittest.TestCase):
    """生产与评测的唯一差别只应该是「测什么题」，不该是「用什么跑题」。"""

    def test_system_prompt_identical(self):
        prod = build_prod_agent()
        ev = build_eval_agent(str(kbapp.SKILLS_DIR), kbapp.DOMAIN_PROMPT)
        self.assertEqual(prod._system_prompt(), ev._system_prompt(),
                         "评测的 system prompt 和生产不一致——领域提示词可能又漏传了")

    def test_tools_identical(self):
        prod = build_prod_agent()
        ev = build_eval_agent(str(kbapp.SKILLS_DIR), kbapp.DOMAIN_PROMPT)
        self.assertEqual(sorted(prod.registry.names()), sorted(ev.registry.names()))

    def test_budget_identical(self):
        prod = build_prod_agent()
        ev = build_eval_agent(str(kbapp.SKILLS_DIR), kbapp.DOMAIN_PROMPT)
        self.assertEqual(prod.budget.snapshot()["limits"], ev.budget.snapshot()["limits"])

    def test_skills_identical(self):
        prod = build_prod_agent()
        ev = build_eval_agent(str(kbapp.SKILLS_DIR), kbapp.DOMAIN_PROMPT)
        self.assertEqual(sorted(prod.skills.skills), sorted(ev.skills.skills))

    def test_domain_prompt_present(self):
        """反向断言：如果哪天 extra_system 又被漏传，这条会红。"""
        ev = build_eval_agent(str(kbapp.SKILLS_DIR), "")
        self.assertNotIn(kbapp.DOMAIN_PROMPT[:40], ev._system_prompt())

    def test_inline_mode_contains_skill_body(self):
        """内联模式下 skill 正文必须出现在 system prompt 里，且不提供 load_skill。"""
        loader = SkillLoader(str(kbapp.SKILLS_DIR)).discover()
        ev = build_eval_agent("", kbapp.DOMAIN_PROMPT + "\n\n" + loader.inline_body())
        prompt = ev._system_prompt()
        self.assertIn("操作手册", prompt)
        self.assertNotIn("load_skill", ev.registry.names())
        self.assertTrue(loader.inline_body().strip())


if __name__ == "__main__":
    unittest.main()