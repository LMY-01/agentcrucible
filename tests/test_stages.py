"""分阶段评测判据的离线单测。

阶段判据是给人看的结论，写错了比不写更糟——它会让人去改错的地方。
这里用构造出来的 trace 事件流，把每个阶段的每个分支都钉住。
"""
from __future__ import annotations

import unittest

from arlab.stages import FAIL, NA, PASS, STAGES, WARN, Verdict, aggregate, judge


def ev_tool(step: int, name: str, ok: bool = True, args: dict | None = None,
            error: str | None = None) -> dict:
    return {
        "event": "tool",
        "step": step,
        "name": name,
        "args": args or {},
        "ok": ok,
        "error": error,
        "duration_ms": 1,
        "result_chars": 10,
    }


def ev_end(**kw) -> dict:
    base = {
        "event": "run_end",
        "steps": 3,
        "total_tokens": 3000,
        "status": "ok",
        "ok": True,
        "failures": [],
        "answer": "答案",
    }
    base.update(kw)
    return base


def pick(verdicts: list[Verdict], stage: str) -> Verdict:
    return next(v for v in verdicts if v.stage == stage)


class TestExplore(unittest.TestCase):
    TASK = {"require_explore": True}

    def test_not_required_is_na(self):
        v = pick(judge([ev_tool(1, "read_file"), ev_end()], {"require_explore": False}), "explore")
        self.assertEqual(v.level, NA)

    def test_no_failed_read_passes(self):
        events = [ev_tool(1, "list_dir"), ev_tool(2, "read_file", ok=True), ev_end()]
        self.assertEqual(pick(judge(events, self.TASK), "explore").level, PASS)

    def test_guess_without_listing_fails(self):
        events = [ev_tool(1, "read_file", ok=False, error="exception"),
                  ev_tool(2, "read_file", ok=True), ev_end()]
        v = pick(judge(events, self.TASK), "explore")
        self.assertEqual(v.level, FAIL)
        self.assertIn("盲猜", v.detail)

    def test_parallel_listing_is_warn(self):
        # 同一步里既 list_dir 又猜文件名：知道自己不确定，但懒得分两轮
        events = [ev_tool(1, "list_dir"),
                  ev_tool(1, "read_file", ok=False, error="exception"),
                  ev_tool(2, "read_file", ok=True), ev_end()]
        v = pick(judge(events, self.TASK), "explore")
        self.assertEqual(v.level, WARN)
        self.assertIn("并行", v.detail)

    def test_listing_after_guessing_is_fail(self):
        # 先猜，报错了才列目录 —— 顺序反了就是 fail，不是 warn
        events = [ev_tool(1, "read_file", ok=False, error="exception"),
                  ev_tool(2, "list_dir"), ev_end()]
        self.assertEqual(pick(judge(events, self.TASK), "explore").level, FAIL)


class TestToolAndArgs(unittest.TestCase):
    TASK = {"require_explore": False}

    def test_clean_run(self):
        events = [ev_tool(1, "read_file"), ev_end()]
        v = judge(events, self.TASK)
        self.assertEqual(pick(v, "tool_valid").level, PASS)
        self.assertEqual(pick(v, "args").level, PASS)

    def test_unknown_tool_hits_tool_valid(self):
        events = [ev_tool(1, "nope", ok=False, error="unknown_tool"), ev_end()]
        v = judge(events, self.TASK)
        self.assertEqual(pick(v, "tool_valid").level, FAIL)
        self.assertEqual(pick(v, "args").level, PASS, "错误应只归到一个阶段")

    def test_bad_json_hits_args(self):
        events = [ev_tool(1, "read_file", ok=False, error="bad_json"), ev_end()]
        v = judge(events, self.TASK)
        self.assertEqual(pick(v, "args").level, FAIL)
        self.assertEqual(pick(v, "tool_valid").level, PASS)

    def test_schema_hits_args(self):
        events = [ev_tool(1, "read_file", ok=False, error="schema"), ev_end()]
        self.assertEqual(pick(judge(events, self.TASK), "args").level, FAIL)


class TestRecovery(unittest.TestCase):
    TASK = {"require_explore": False}

    def test_no_failure_is_na(self):
        v = pick(judge([ev_tool(1, "read_file"), ev_end()], self.TASK), "recovery")
        self.assertEqual(v.level, NA)

    def test_retry_same_args_fails(self):
        events = [ev_tool(1, "read_file", ok=False, args={"path": "a.csv"}, error="exception"),
                  ev_tool(2, "read_file", ok=False, args={"path": "a.csv"}, error="exception"),
                  ev_end()]
        self.assertEqual(pick(judge(events, self.TASK), "recovery").level, FAIL)

    def test_changed_args_passes(self):
        events = [ev_tool(1, "read_file", ok=False, args={"path": "a.csv"}, error="exception"),
                  ev_tool(2, "read_file", ok=True, args={"path": "b.csv"}),
                  ev_end()]
        self.assertEqual(pick(judge(events, self.TASK), "recovery").level, PASS)


class TestSkillRecall(unittest.TestCase):
    def test_no_expectation_is_na(self):
        v = pick(judge([ev_tool(1, "read_file"), ev_end()], {}), "skill_recall")
        self.assertEqual(v.level, NA)

    def test_loaded_required_passes(self):
        events = [ev_tool(1, "load_skill", args={"name": "csv-profiling"}),
                  ev_tool(2, "run_python"), ev_end()]
        v = pick(judge(events, {"required_skills": ["csv-profiling"]}), "skill_recall")
        self.assertEqual(v.level, PASS)

    def test_missing_required_with_wrong_answer_fails(self):
        events = [ev_tool(1, "read_file"), ev_end(ok=False, failures=["答案错误"])]
        v = pick(judge(events, {"expected_skills": ["csv-profiling"]}), "skill_recall")
        self.assertEqual(v.level, FAIL)

    def test_missing_required_with_correct_answer_warns(self):
        events = [ev_tool(1, "read_file"), ev_end()]
        v = pick(judge(events, {"required_skills": ["csv-profiling"]}), "skill_recall")
        self.assertEqual(v.level, WARN)
        self.assertIn("待归因", v.detail)

    def test_missing_optional_is_na(self):
        events = [ev_tool(1, "read_file"), ev_end()]
        v = pick(judge(events, {"optional_skills": ["csv-profiling"]}), "skill_recall")
        self.assertEqual(v.level, NA)

    def test_loaded_optional_passes(self):
        events = [ev_tool(1, "load_skill", args={"name": "csv-profiling"}), ev_end()]
        v = pick(judge(events, {"optional_skills": ["csv-profiling"]}), "skill_recall")
        self.assertEqual(v.level, PASS)


class TestEfficiencyAndTermination(unittest.TestCase):
    def test_over_limit_warns_cost_without_waste_fail(self):
        task = {"stage_limits": {"steps": 3, "tokens": 1000}}
        verdicts = judge([ev_end(steps=4, total_tokens=900)], task)
        self.assertEqual(pick(verdicts, "efficiency_cost").level, WARN)
        self.assertEqual(pick(verdicts, "efficiency_waste").level, PASS)

    def test_near_limit_warns_cost(self):
        task = {"stage_limits": {"steps": 5, "tokens": 10000}}
        v = pick(judge([ev_end(steps=4, total_tokens=5000)], task), "efficiency_cost")
        self.assertEqual(v.level, WARN)

    def test_roomy_passes_cost(self):
        task = {"stage_limits": {"steps": 10, "tokens": 20000}}
        v = pick(judge([ev_end(steps=3, total_tokens=3000)], task), "efficiency_cost")
        self.assertEqual(v.level, PASS)

    def test_same_args_retry_fails_waste(self):
        events = [ev_tool(1, "read_file", args={"path": "a"}),
                  ev_tool(2, "read_file", args={"path": "a"}), ev_end()]
        v = pick(judge(events, {}), "efficiency_waste")
        self.assertEqual(v.level, FAIL)
        self.assertIn("同参数重试", v.detail)

    def test_repeated_kb_read_fails_waste(self):
        events = [ev_tool(1, "kb_read", args={"chunk_id": "c1"}),
                  ev_tool(2, "kb_search", args={"query": "x"}),
                  ev_tool(3, "kb_read", args={"chunk_id": "c1"}), ev_end()]
        v = pick(judge(events, {}), "efficiency_waste")
        self.assertEqual(v.level, FAIL)
        self.assertIn("重复读同一块", v.detail)

    def test_no_progress_loop_fails_waste(self):
        events = [ev_tool(step, "kb_search", args={"query": f"q{step}"})
                  for step in range(1, 5)]
        events.append(ev_end(steps=4))
        v = pick(judge(events, {}), "efficiency_waste")
        self.assertEqual(v.level, FAIL)
        self.assertIn("无新增块", v.detail)

    def test_new_blocks_and_tool_types_reset_no_progress(self):
        events = [ev_tool(1, "kb_search", args={"query": "q1"}),
                  ev_tool(2, "kb_read", args={"chunk_id": "c1"}),
                  ev_tool(3, "kb_search", args={"query": "q2"}),
                  ev_tool(4, "kb_read", args={"chunk_id": "c2"}),
                  ev_tool(5, "submit_answer"), ev_end(steps=5)]
        self.assertEqual(pick(judge(events, {}), "efficiency_waste").level, PASS)

    def test_budget_exhausted_fails_termination(self):
        v = pick(judge([ev_end(status="budget_exhausted", reason="steps")], {}), "termination")
        self.assertEqual(v.level, FAIL)

    def test_ok_passes_termination(self):
        self.assertEqual(pick(judge([ev_end()], {}), "termination").level, PASS)


class TestAnswerAndAggregate(unittest.TestCase):
    def test_answer_failure_carries_reason(self):
        v = pick(judge([ev_end(ok=False, failures=["答案缺少 '80'"])], {}), "answer")
        self.assertEqual(v.level, FAIL)
        self.assertIn("80", v.detail)

    def test_always_all_stages(self):
        v = judge([ev_end()], {})
        self.assertEqual([x.stage for x in v], [s for s, _ in STAGES])

    def test_aggregate_counts(self):
        ok = judge([ev_end()], {})
        bad = judge([ev_tool(1, "nope", ok=False, error="unknown_tool"), ev_end()], {})
        agg = aggregate([ok, bad])
        self.assertEqual(agg["tool_valid"]["pass"], 1)
        self.assertEqual(agg["tool_valid"]["fail"], 1)
        self.assertEqual(agg["tool_valid"]["rate"], 0.5)
        self.assertEqual(agg["skill_recall"]["skipped"], 2)
        self.assertIsNone(agg["skill_recall"]["rate"])

    def test_aggregate_collects_notes(self):
        bad = judge([ev_tool(1, "read_file", ok=False, error="exception"), ev_end()],
                    {"require_explore": True})
        agg = aggregate([bad])
        self.assertTrue(any("盲猜" in n for n in agg["explore"]["notes"]))


if __name__ == "__main__":
    unittest.main()
