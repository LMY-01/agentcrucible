"""kb-agent 垂直层判据的单测。

这些谓词决定「引用有没有编造」能不能被自动化判定，
写错了会让整份黄金集的结论失效——比运行时 bug 更危险，因为它不会报错，只会给错分。

用假 store 隔离，不需要加载真实索引和 embedding 模型。
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from apps.kb_agent.app import (
    pred_answer_covers,
    pred_citations_grounded,
    pred_not_refused,
    pred_states_missing,
)

DOCS = {
    "c1": "LoRA 通过低秩分解来模拟参数的改变量，从而以极小的参数量实现大模型的间接训练。",
    "c2": "第一个矩阵 A 通过高斯函数初始化，第二个矩阵 B 初始化为零矩阵。",
}


class FakeStore:
    def chunk(self, cid: str):
        if cid not in DOCS:
            raise KeyError(cid)
        return SimpleNamespace(text=DOCS[cid], id=cid, source="x.md", idx=0)


def answer(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


class GroundedBase(unittest.TestCase):
    def setUp(self):
        self._p = patch("apps.kb_agent.app._store", return_value=FakeStore())
        self._p.start()
        self.addCleanup(self._p.stop)


class TestCitationsGrounded(GroundedBase):
    def test_all_quotes_exist(self):
        ans = answer({"answer": "低秩分解", "refused": False, "citations": [
            {"id": 1, "chunk_id": "c1", "quote": "通过低秩分解来模拟参数的改变量"},
            {"id": 2, "chunk_id": "c2", "quote": "第二个矩阵 B 初始化为零矩阵"},
        ]})
        self.assertIsNone(pred_citations_grounded(ans, {}))

    def test_rewritten_quote_fails(self):
        # 把原文改写一个字，就是编造，必须拦住
        ans = answer({"answer": "x", "citations": [
            {"id": 1, "chunk_id": "c1", "quote": "通过低秩分解模拟参数的改变"},
        ]})
        reason = pred_citations_grounded(ans, {})
        self.assertIsNotNone(reason)
        self.assertIn("不在原文", reason)

    def test_no_citations_fails(self):
        ans = answer({"answer": "结论", "refused": False, "citations": []})
        self.assertIn("一条引用都没给", pred_citations_grounded(ans, {}))

    def test_unknown_chunk_id_fails(self):
        ans = answer({"answer": "x", "citations": [
            {"id": 1, "chunk_id": "不存在的块", "quote": "随便"},
        ]})
        self.assertIn("不存在", pred_citations_grounded(ans, {}))

    def test_not_json_fails(self):
        self.assertIn("JSON", pred_citations_grounded("这是纯文本回答", {}))

    def test_missing_fields_fails(self):
        ans = answer({"answer": "x", "citations": [{"quote": "只有引文没有 id"}]})
        self.assertIn("缺", pred_citations_grounded(ans, {}))


class TestAnswerCovers(GroundedBase):
    def test_enough_hits(self):
        ans = answer({"answer": "幻觉和数据新鲜度", "citations": []})
        self.assertIsNone(pred_answer_covers(ans, {"must": ["幻觉", "数据新鲜度"], "min_hit": 2}))

    def test_missing_hit(self):
        ans = answer({"answer": "幻觉", "citations": []})
        reason = pred_answer_covers(ans, {"must": ["幻觉", "数据新鲜度"], "min_hit": 2})
        self.assertIn("数据新鲜度", reason)

    def test_partial_allowed(self):
        ans = answer({"answer": "幻觉", "citations": []})
        self.assertIsNone(pred_answer_covers(ans, {"must": ["幻觉", "数据新鲜度"], "min_hit": 1}))


class TestRefusalPredicates(GroundedBase):
    def test_not_refused_passes(self):
        ans = answer({"answer": "x", "refused": False, "citations": []})
        self.assertIsNone(pred_not_refused(ans, {}))

    def test_not_refused_catches_wrong_refusal(self):
        ans = answer({"answer": "资料没有", "refused": True, "citations": []})
        self.assertIn("不该拒答", pred_not_refused(ans, {}))

    def test_states_missing_by_flag(self):
        ans = answer({"answer": "没有相关资料", "refused": True, "citations": []})
        self.assertIsNone(pred_states_missing(ans, {}))

    def test_states_missing_by_wording(self):
        ans = answer({"answer": "资料中没有提到 BGE-M3 的向量维度", "refused": False, "citations": []})
        self.assertIsNone(pred_states_missing(ans, {}))

    def test_fabricated_number_fails(self):
        ans = answer({"answer": "BGE-M3 的向量维度是 1024 维", "refused": False, "citations": []})
        reason = pred_states_missing(ans, {})
        self.assertIsNotNone(reason, "编了数字却没被拦住")
        self.assertIn("没说明资料缺失", reason)

    def test_forced_finish_does_not_count_as_refusal(self):
        """被预算逼停后说「资料里没有」，不算拒答成功——它只是被逼到墙角。"""
        ans = answer({"answer": "资料中没有相关资料", "refused": True, "citations": []})
        ctx = {"status": "budget_exhausted", "reason": "tokens", "tokens": 56562, "steps": 7}
        reason = pred_states_missing(ans, {}, ctx)
        self.assertIsNotNone(reason, "预算逼停的收尾被误判成主动拒答")
        self.assertIn("逼停", reason)

    def test_normal_finish_still_passes(self):
        ans = answer({"answer": "资料中没有相关资料", "refused": True, "citations": []})
        self.assertIsNone(pred_states_missing(ans, {}, {"status": "ok", "tokens": 12000}))


class TestSubmissionGate(GroundedBase):
    """交付门禁：编造的引用必须提交不进去，而不是提交后扣分。"""

    def test_real_quote_passes(self):
        from apps.kb_agent.kb_tools import validate_submission

        payload = {"citations": [{"chunk_id": "c1", "quote": "通过低秩分解来模拟参数的改变量"}],
                   "refused": False}
        self.assertIsNone(validate_submission(payload))

    def test_fabricated_quote_rejected(self):
        from apps.kb_agent.kb_tools import validate_submission

        payload = {"citations": [{"chunk_id": "c1", "quote": "这句话原文里没有"}], "refused": False}
        reason = validate_submission(payload)
        self.assertIsNotNone(reason, "编造的引文没被拦住")
        self.assertIn("找不到", reason)

    def test_bad_chunk_id_rejected(self):
        from apps.kb_agent.kb_tools import validate_submission

        payload = {"citations": [{"chunk_id": "不存在", "quote": "随便"}], "refused": False}
        self.assertIn("不存在", validate_submission(payload) or "")

    def test_no_citation_without_refusal_rejected(self):
        from apps.kb_agent.kb_tools import validate_submission

        payload = {"citations": [], "refused": False}
        self.assertIsNotNone(validate_submission(payload))

    def test_refusal_allows_empty_citations(self):
        from apps.kb_agent.kb_tools import validate_submission

        self.assertIsNone(validate_submission({"citations": [], "refused": True}))

    def test_partial_fabrication_rejected(self):
        """8 条真的 + 2 条假的，也要整单拒绝——这正是 silver_007 的实际情况。"""
        from apps.kb_agent.kb_tools import validate_submission

        cites = [{"chunk_id": "c1", "quote": "通过低秩分解来模拟参数的改变量"} for _ in range(8)]
        cites += [{"chunk_id": "c1", "quote": "编的第九条"}, {"chunk_id": "c1", "quote": "编的第十条"}]
        reason = validate_submission({"citations": cites, "refused": False})
        self.assertIsNotNone(reason)
        self.assertIn("找不到", reason)


if __name__ == "__main__":
    unittest.main()
