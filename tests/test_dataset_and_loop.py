"""评测题集与主循环的契约单测（零网络、零外部依赖）。

为什么需要它：
  1. 黄金集 / 留出集 / 多跳集 / 白银集一旦写错谓词名、漏掉字段、ID 冲突、
     或出现黄金集与白银集题目重合（数据泄漏），评测分就会静默失真。
  2. Agent 主循环里的关键降级与门禁路径（预算耗尽保留 submit_answer、
     绕过终止工具被打回、validator 拒绝、出口过滤打回、连续同参死循环拦截）
     必须用 Mock LLM 端到端锁住，不能只靠线上跑题才发现退化。
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from apps.kb_agent.app import PREDICATES
from arlab.budget import Budget
from arlab.defense import OutputFilter
from arlab.llm import Reply, Usage
from arlab.loop import Agent
from arlab.tools.registry import READ, Registry, Tool

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "apps" / "kb_agent" / "eval"


def _norm_q(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).lower()


class FakeLLM:
    """按预设队列依次返回 Reply，不走任何网络。"""

    def __init__(self, replies: list[Reply], model: str = "fake/model") -> None:
        self.replies = list(replies)
        self.model = model
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
        if not self.replies:
            raise AssertionError("FakeLLM 预设回复已耗尽")
        return self.replies.pop(0)


def _reply_text(content: str, tokens: int = 100) -> Reply:
    return Reply(
        message={"role": "assistant", "content": content},
        usage=Usage(prompt_tokens=tokens // 2, completion_tokens=tokens // 2),
        latency_ms=10,
        raw_finish_reason="stop",
    )


def _reply_tool(name: str, args: dict, call_id: str = "c1", tokens: int = 100) -> Reply:
    return Reply(
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }],
        },
        usage=Usage(prompt_tokens=tokens // 2, completion_tokens=tokens // 2),
        latency_ms=10,
        raw_finish_reason="tool_calls",
    )


class TestDatasetContracts(unittest.TestCase):
    """静态校验所有已入库题集的结构、谓词合法性、自洽性与无数据泄漏。"""

    FILES = ("gold.json", "gold_holdout.json", "gold_multihop.json", "silver.json")

    def _load(self, name: str) -> list[dict]:
        path = EVAL_DIR / name
        self.assertTrue(path.exists(), f"缺少题集文件: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        return data

    def test_all_predicates_registered_and_well_formed(self):
        seen_ids: set[str] = set()
        for fname in self.FILES:
            tasks = self._load(fname)
            for t in tasks:
                tid = t.get("id", "")
                self.assertTrue(tid, f"{fname} 存在无 id 任务")
                self.assertNotIn(tid, seen_ids, f"跨题集任务 id 重复: {tid}")
                seen_ids.add(tid)
                self.assertTrue(t.get("prompt", "").strip(), f"{tid} 缺少 prompt")
                checks = t.get("checks") or []
                self.assertGreater(len(checks), 0, f"{tid} 没有任何 check")
                for c in checks:
                    self.assertEqual(c.get("type"), "predicate", f"{tid} 非 predicate check")
                    pname = c.get("name", "")
                    self.assertIn(pname, PREDICATES, f"{tid} 使用了未注册谓词 {pname!r}")
                    if pname == "answer_covers":
                        must = c.get("must") or []
                        min_hit = int(c.get("min_hit", len(must)))
                        self.assertGreater(len(must), 0, f"{tid} answer_covers.must 为空")
                        self.assertLessEqual(min_hit, len(must), f"{tid} min_hit > len(must)")
                        # 参考答案自身必须能通过 answer_covers，防止题目与参考答案自相矛盾
                        fake_ans = json.dumps({"answer": t["reference_answer"], "citations": [], "refused": False},
                                              ensure_ascii=False)
                        err = PREDICATES["answer_covers"](fake_ans, c)
                        self.assertIsNone(err, f"{tid} 的 reference_answer 未能通过自身的 answer_covers: {err}")
                    elif pname == "evidence_hit":
                        groups = c.get("groups") or []
                        self.assertGreater(len(groups), 0, f"{tid} evidence_hit.groups 为空")
                        self.assertTrue(t.get("evidence_chunks"),
                                        f"{tid} 缺少脚本反查生成的 evidence_chunks")
                    elif pname == "chunk_hit":
                        self.assertTrue(c.get("expected"), f"{tid} chunk_hit.expected 为空")

    def test_no_prompt_leakage_between_gold_and_silver(self):
        gold_prompts: set[str] = set()
        for fname in ("gold.json", "gold_holdout.json", "gold_multihop.json"):
            for t in self._load(fname):
                gold_prompts.add(_norm_q(t["prompt"]))
        for t in self._load("silver.json"):
            self.assertNotIn(_norm_q(t["prompt"]), gold_prompts,
                             f"白银题 {t['id']} 与黄金集题目重复: {t['prompt']}")

    def test_silver_questions_are_clean_and_standalone(self):
        for t in self._load("silver.json"):
            q = t["prompt"]
            self.assertFalse(re.match(r"^[一二三四五六七八九十\d]+[\.、．]", q),
                             f"{t['id']} 残留章节编号前缀: {q!r}")
            self.assertNotIn("toc", q.lower())
            self.assertGreaterEqual(len(q), 8, f"{t['id']} 问句过短: {q!r}")


class TestAgentLoopContracts(unittest.TestCase):
    """用 FakeLLM 离线验证主循环的降级、门禁与防死循环逻辑。"""

    def _make_registry(self, validator=None) -> Registry:
        reg = Registry(allow={READ})
        reg.register(Tool(
            name="kb_search",
            description="search",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            handler=lambda q: f"hit for {q}",
        ))
        reg.register(Tool(
            name="submit_answer",
            description="submit",
            parameters={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            handler=lambda answer: answer,
            terminal=True,
            validator=validator,
        ))
        return reg

    def test_budget_exhaustion_keeps_terminal_tool_open(self):
        """回归：预算耗尽收尾时，检索工具必须关闭，但 submit_answer 必须保留开放。"""
        llm = FakeLLM([
            _reply_tool("kb_search", {"q": "lora"}, call_id="c1"),
            _reply_tool("submit_answer", {"answer": "基于已有信息的收尾结论"}, call_id="c2"),
        ])
        agent = Agent(llm, self._make_registry(), budget=Budget(max_steps=1))
        res = agent.run("测试预算降级")
        self.assertEqual(res.status, "ok")
        self.assertIn("收尾结论", res.answer)
        # 第二次调用（nudged 之后）只应提供终止型工具 submit_answer
        self.assertEqual(llm.calls[1]["tools"], ["submit_answer"])

    def test_bypassing_terminal_tool_is_pushed_back_once(self):
        """有 submit_answer 时模型若直接裸输出文本，主循环必须打回一次逼它走工具。"""
        llm = FakeLLM([
            _reply_text("我直接输出纯文本，不调工具"),
            _reply_tool("submit_answer", {"answer": "乖乖走工具提交"}, call_id="c1"),
        ])
        agent = Agent(llm, self._make_registry(), budget=Budget(max_steps=5))
        res = agent.run("测试推回")
        self.assertEqual(res.status, "ok")
        self.assertIn("乖乖走工具提交", res.answer)
        self.assertEqual(len(llm.calls), 2)

    def test_terminal_validator_rejects_and_allows_retry(self):
        """交付门禁返回错误字符串时，不能结束 run，必须把原因回填给模型重交。"""
        def gate(payload: dict) -> str | None:
            return None if "合格" in payload.get("answer", "") else "缺少关键词"

        llm = FakeLLM([
            _reply_tool("submit_answer", {"answer": "第一次不达标"}, call_id="c1"),
            _reply_tool("submit_answer", {"answer": "第二次合格交付"}, call_id="c2"),
        ])
        agent = Agent(llm, self._make_registry(validator=gate), budget=Budget(max_steps=5))
        res = agent.run("测试门禁打回")
        self.assertEqual(res.status, "ok")
        self.assertIn("第二次合格交付", res.answer)
        self.assertEqual(len(llm.calls), 2)

    def test_identical_tool_call_loop_is_intercepted(self):
        """连续 3 次同工具同参数调用必须被 Registry 拦截为 loop_detected。"""
        reg = self._make_registry()
        r1 = reg.call("kb_search", '{"q": "same"}')
        r2 = reg.call("kb_search", '{"q": "same"}')
        r3 = reg.call("kb_search", '{"q": "same"}')
        self.assertTrue(r1.ok and r2.ok)
        self.assertFalse(r3.ok)
        self.assertEqual(r3.error, "loop_detected")

    def test_output_guard_pushes_back_leaked_secret_once(self):
        """出口过滤命中敏感凭据时，打回给模型改写后再放行。"""
        reg = Registry(allow={READ})
        llm = FakeLLM([
            _reply_text("发现密钥 sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456"),
            _reply_text("该文件中包含一条 API 密钥凭据，已略去具体值。"),
        ])
        agent = Agent(llm, reg, budget=Budget(max_steps=5), output_guard=OutputFilter(enabled=True))
        res = agent.run("检查文件")
        self.assertEqual(res.status, "ok")
        self.assertNotIn("sk-proj-", res.answer)
        self.assertEqual(len(llm.calls), 2)


if __name__ == "__main__":
    unittest.main()
