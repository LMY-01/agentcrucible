"""离线单测：不联网、不调模型，纯测运行时自身的契约。

跑法：PYTHONPATH=src python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from arlab.budget import Budget
from arlab.skills import SkillLoader, _parse_frontmatter
from arlab.tools.builtin import _safe_join, build_tools
from arlab.tools.registry import READ, WRITE, Registry, Tool, validate
from arlab.trace import Trace


class TestValidate(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "n": {"type": "integer"},
            "mode": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def test_ok(self):
        self.assertEqual(validate({"path": "x", "n": 3}, self.SCHEMA), [])

    def test_missing_required(self):
        self.assertIn("缺少必填参数", validate({}, self.SCHEMA)[0])

    def test_wrong_type(self):
        errs = validate({"path": "x", "n": "3"}, self.SCHEMA)
        self.assertTrue(any("integer" in e for e in errs))

    def test_bool_is_not_integer(self):
        # Python 里 True 是 int 的子类，但 JSON Schema 里 boolean 不是 integer
        errs = validate({"path": "x", "n": True}, self.SCHEMA)
        self.assertTrue(errs, "bool 不该被当成 integer 放行")

    def test_enum(self):
        errs = validate({"path": "x", "mode": "z"}, self.SCHEMA)
        self.assertTrue(any("enum" not in e and "必须是" in e for e in errs))

    def test_additional_props(self):
        errs = validate({"path": "x", "zzz": 1}, self.SCHEMA)
        self.assertTrue(any("不允许的参数" in e for e in errs))


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = Registry(allow={READ})
        self.reg.register(
            Tool(
                name="echo",
                description="echo back",
                parameters={"type": "object", "properties": {"msg": {"type": "string"}},
                            "required": ["msg"]},
                handler=lambda msg: msg,
                permission=READ,
            )
        )
        self.reg.register(
            Tool(
                name="writer",
                description="write something",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda: "written",
                permission=WRITE,
            )
        )

    def test_unknown_tool(self):
        r = self.reg.call("nope", "{}")
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "unknown_tool")

    def test_bad_json(self):
        r = self.reg.call("echo", "{not json")
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "bad_json")

    def test_schema_failure(self):
        r = self.reg.call("echo", "{}")
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "schema")

    def test_permission_denied(self):
        r = self.reg.call("writer", "{}")
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "permission_denied")

    def test_permission_granted(self):
        self.reg.allow.add(WRITE)
        r = self.reg.call("writer", "{}")
        self.assertTrue(r.ok)

    def test_handler_exception_is_caught(self):
        self.reg.register(
            Tool(name="boom", description="raise", parameters={"type": "object", "properties": {}},
                 handler=lambda: 1 / 0)
        )
        r = self.reg.call("boom", "{}")
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "exception")

    def test_null_arguments(self):
        # 有些模型对无参工具会给 None 而不是 "{}"
        r = self.reg.call("writer", None)
        self.assertEqual(r.error, "permission_denied")  # 说明走到了权限检查，没崩

    def test_schemas_shape(self):
        schemas = self.reg.schemas()
        self.assertTrue(all(s["type"] == "function" for s in schemas))
        self.assertIn("echo", [s["function"]["name"] for s in schemas])


class TestWorkspaceSandbox(unittest.TestCase):
    def test_escape_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                _safe_join(Path(tmp), "../../etc/passwd")

    def test_inside_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_safe_join(Path(tmp), "a/b.txt").name, "b.txt")

    def test_tools_read_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools = {t.name: t for t in build_tools(tmp)}
            tools["write_file"].handler(path="d/x.txt", content="hi")
            self.assertEqual((Path(tmp) / "d" / "x.txt").read_text(), "hi")
            self.assertEqual(tools["read_file"].handler(path="d/x.txt"), "hi")

    def test_truncate_keeps_both_ends(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools = {t.name: t for t in build_tools(tmp)}
            body = "A" * 5000 + "TAIL"
            tools["write_file"].handler(path="big.txt", content=body)
            out = tools["read_file"].handler(path="big.txt", max_chars=200)
            self.assertIn("省略", out)
            self.assertTrue(out.startswith("A"))
            self.assertTrue(out.endswith("TAIL"))


class TestBudget(unittest.TestCase):
    def test_steps(self):
        b = Budget(max_steps=2)
        b.charge(1)
        b.charge(1)
        self.assertEqual(b.exhausted(), "steps")

    def test_tokens(self):
        b = Budget(max_tokens=100)
        b.charge(150)
        self.assertEqual(b.exhausted(), "tokens")

    def test_consecutive_failures_reset(self):
        b = Budget(max_consecutive_tool_failures=2)
        b.note_tool_result(False)
        b.note_tool_result(True)
        self.assertIsNone(b.exhausted())
        b.note_tool_result(False)
        b.note_tool_result(False)
        self.assertEqual(b.exhausted(), "tool_failures")


class TestSkills(unittest.TestCase):
    def test_frontmatter(self):
        meta, body = _parse_frontmatter("---\nname: x\ndescription: hello\n---\n正文")
        self.assertEqual(meta["name"], "x")
        self.assertEqual(meta["description"], "hello")
        self.assertEqual(body, "正文")

    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("just text")
        self.assertEqual(meta, {})
        self.assertEqual(body, "just text")

    def test_discover_skips_skill_without_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok = Path(tmp) / "good"
            ok.mkdir()
            (ok / "SKILL.md").write_text("---\nname: good\ndescription: d\n---\nbody")
            bad = Path(tmp) / "bad"
            bad.mkdir()
            (bad / "SKILL.md").write_text("---\nname: bad\n---\nbody")
            loader = SkillLoader(tmp).discover()
            self.assertEqual(list(loader.skills), ["good"])

    def test_catalog_contains_only_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "s"
            d.mkdir()
            (d / "SKILL.md").write_text("---\nname: s\ndescription: 一句话\n---\nSECRET_BODY")
            loader = SkillLoader(tmp).discover()
            self.assertIn("一句话", loader.catalog())
            self.assertNotIn("SECRET_BODY", loader.catalog())

    def test_load_body_unknown(self):
        loader = SkillLoader("/nonexistent").discover()
        with self.assertRaises(KeyError):
            loader.load_body("x")


class TestTrace(unittest.TestCase):
    def test_summarize(self):
        from arlab.llm import Usage

        with tempfile.TemporaryDirectory() as tmp:
            tr = Trace(tmp, meta={"task": "t"})
            tr.llm_call(step=1, model="m", usage=Usage(100, 20), latency_ms=50,
                        finish_reason="tool_calls", tools_offered=["a"])
            tr.tool_call(step=1, name="a", args={}, ok=True, duration_ms=5, result_chars=10)
            tr.tool_call(step=1, name="a", args={}, ok=False, duration_ms=5,
                         result_chars=0, error="schema")
            tr.close()
            s = tr.summarize()
            self.assertEqual(s["steps"], 1)
            self.assertEqual(s["tool_calls"], 2)
            self.assertEqual(s["tool_failed"], 1)
            self.assertEqual(s["invalid_tool_rate"], 0.5)
            self.assertEqual(s["total_tokens"], 120)
            self.assertEqual(s["tool_histogram"], {"a": 2})

            with open(tr.path, encoding="utf-8") as fh:
                events = [json.loads(x)["event"] for x in fh]
            self.assertEqual(events, ["run_start", "llm", "tool", "tool", "run_end"])


if __name__ == "__main__":
    unittest.main()
