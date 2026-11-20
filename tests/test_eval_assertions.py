"""结构化断言与质量分的单测。

这两块是「能不能信这个分数」的地基：
断言宽松会让错答案通过，权重写错会让报告指向错的环节。
"""
from __future__ import annotations

import unittest
from pathlib import Path

from arlab.evaluate import _check, _dig, _extract_json
from arlab.scoring import LEVEL_SCORE, STAGE_WEIGHTS, run_score, task_score
from arlab.stages import FAIL, NA, PASS, STAGES, WARN, Verdict

NAME = dict(STAGES)


def V(stage: str, level: str) -> Verdict:
    return Verdict(stage, NAME[stage], level, "")


class TestExtractJson(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(_extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        text = '结果如下：\n```json\n{"a": 2}\n```\n以上。'
        self.assertEqual(_extract_json(text), {"a": 2})

    def test_with_preamble(self):
        self.assertEqual(_extract_json('我算了一下 {"a": 3} 就这样'), {"a": 3})

    def test_nested_braces(self):
        text = '{"outer": {"inner": [1, {"x": 2}]}}'
        self.assertEqual(_extract_json(text)["outer"]["inner"][1]["x"], 2)

    def test_none_when_absent(self):
        self.assertIsNone(_extract_json("完全没有 JSON"))


class TestDig(unittest.TestCase):
    def test_dict(self):
        self.assertEqual(_dig({"a": {"b": 1}}, "a.b"), 1)

    def test_list_index(self):
        self.assertEqual(_dig({"r": [{"n": "x"}]}, "r.0.n"), "x")

    def test_missing(self):
        self.assertIsNone(_dig({"a": 1}, "a.b.c"))
        self.assertIsNone(_dig({"r": []}, "r.0"))


class TestJsonFieldChecks(unittest.TestCase):
    """核心回归：子串匹配放过 180，结构化断言必须拦住。"""

    def setUp(self):
        self.ws = Path("/tmp")

    def test_wrong_number_fails(self):
        checks = [{"type": "json_field", "field": "total", "number": 80, "tolerance": 0.01}]
        ok, fails = _check(checks, '{"total": 180, "top": "B"}', self.ws)
        self.assertFalse(ok, "180 不能被判成 80")
        self.assertIn("80", fails[0])

    def test_correct_number_passes(self):
        checks = [{"type": "json_field", "field": "total", "number": 80}]
        ok, _ = _check(checks, '{"total": 80}', self.ws)
        self.assertTrue(ok)

    def test_tolerance(self):
        checks = [{"type": "json_field", "field": "v", "number": 1.0, "tolerance": 0.01}]
        self.assertTrue(_check(checks, '{"v": 1.005}', self.ws)[0])
        self.assertFalse(_check(checks, '{"v": 1.02}', self.ws)[0])

    def test_string_equals(self):
        checks = [{"type": "json_field", "field": "top", "equals": "B"}]
        self.assertTrue(_check(checks, '{"top": "B"}', self.ws)[0])
        self.assertFalse(_check(checks, '{"top": "Below"}', self.ws)[0],
                         "字符串相等不能用包含判")

    def test_missing_field(self):
        checks = [{"type": "json_field", "field": "ranking.0.name", "equals": "Bolt"}]
        ok, fails = _check(checks, '{"ranking": []}', self.ws)
        self.assertFalse(ok)

    def test_no_json_at_all(self):
        checks = [{"type": "json_field", "field": "rows", "number": 5}]
        ok, fails = _check(checks, "一共 5 行", self.ws)
        self.assertFalse(ok)
        self.assertIn("没有可解析的 JSON", fails[0])

    def test_bool_is_not_number(self):
        checks = [{"type": "json_field", "field": "n", "number": 1}]
        ok, _ = _check(checks, '{"n": true}', self.ws)
        self.assertFalse(ok)

    def test_fenced_answer_accepted(self):
        checks = [{"type": "json_field", "field": "rows", "number": 5}]
        self.assertTrue(_check(checks, '```json\n{"rows": 5}\n```', self.ws)[0])


class TestScoring(unittest.TestCase):
    def test_all_pass_is_one(self):
        vs = [V(s, PASS) for s in STAGE_WEIGHTS]
        self.assertEqual(task_score(vs)[0], 1.0)

    def test_answer_fail_costs_most(self):
        base = [V(s, PASS) for s in STAGE_WEIGHTS]
        only_answer = [V("answer", FAIL) if v.stage == "answer" else v for v in base]
        only_eff = [V("efficiency_waste", FAIL) if v.stage == "efficiency_waste" else v for v in base]
        self.assertLess(task_score(only_answer)[0], task_score(only_eff)[0],
                        "答案错应该比效率超预算扣得多")

    def test_na_is_excluded_and_weight_renormalised(self):
        vs = [V("answer", PASS), V("explore", NA), V("efficiency_waste", NA)]
        score, _ = task_score(vs)
        self.assertEqual(score, 1.0, "不考察的阶段不该扣分")

    def test_warn_counts_half(self):
        vs = [V("answer", WARN)]
        self.assertEqual(task_score(vs)[0], 0.5)

    def test_cost_warn_does_not_affect_score_or_drag(self):
        verdicts = [V(s, PASS) for s in STAGE_WEIGHTS]
        verdicts = [V("efficiency_cost", WARN) if v.stage == "efficiency_cost" else v
                    for v in verdicts]
        score, drag = task_score(verdicts)
        self.assertEqual(score, 1.0)
        self.assertEqual(drag, [])

    def test_skill_loading_and_attribution(self):
        loaded = Verdict("skill_recall", "技能召回", PASS, "加载了 retrieval-qa",
                         "load_skill 调用: ['retrieval-qa']")
        missing = Verdict("skill_recall", "技能召回", WARN,
                          "必需技能 ['retrieval-qa'] 未加载，但答案正确，列入待归因",
                          "load_skill 调用: 无")
        optional = Verdict("skill_recall", "技能召回", NA, "可选技能未加载，不扣分",
                           "load_skill 调用: 无")
        undeclared = Verdict("skill_recall", "技能召回", NA, "任务未声明期望技能")
        result = run_score([[loaded], [missing], [optional], [undeclared]])
        self.assertEqual(result["skill_loading"], {"loaded": 1, "tasks": 3, "rate": 0.333})
        self.assertEqual(result["per_task"][0]["attribution"], [])
        self.assertEqual(result["per_task"][1]["attribution"], ["技能召回"])

    def test_drag_lists_offenders(self):
        vs = [V("answer", FAIL), V("efficiency_waste", PASS)]
        _, drag = task_score(vs)
        self.assertEqual(drag, ["答案正确"])

    def test_run_score_gate(self):
        good = [V(s, PASS) for s in STAGE_WEIGHTS]
        bad = [V(s, PASS) for s in STAGE_WEIGHTS]
        bad = [V("answer", FAIL) if v.stage == "answer" else v for v in bad]
        self.assertEqual(run_score([good, good], 0.9)["gate"], "pass")
        self.assertEqual(run_score([good, bad], 0.9)["gate"], "fail")
        self.assertEqual(run_score([good, bad], 0.9)["drag_counter"], {"答案正确": 1})

    def test_empty(self):
        self.assertEqual(run_score([])["quality"], 0.0)

    def test_level_score_table(self):
        self.assertEqual(LEVEL_SCORE, {PASS: 1.0, WARN: 0.5, FAIL: 0.0})


if __name__ == "__main__":
    unittest.main()
