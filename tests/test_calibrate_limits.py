import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.calibrate_limits import group_rows_by_qtype, load_rows, main, pct


def task(task_id, qtype):
    return {"id": task_id, "qtype": qtype, "stage_limits": {"steps": 8, "tokens": 25000}}


def row(task_id, tokens, steps=3):
    return {"id": task_id, "tokens": tokens, "steps": steps}


class TestPercentile(unittest.TestCase):
    def test_median_and_p90(self):
        self.assertEqual(pct([100, 200, 300, 400, 500], 0.5), 300)
        self.assertEqual(pct([100, 200, 300, 400, 500], 0.9), 500)


class TestQtypeGroups(unittest.TestCase):
    def test_small_groups_merge_to_nearest_median(self):
        tasks = []
        rows = []
        for name, count, tokens in [("a", 5, 100), ("b", 5, 1000), ("c", 2, 120), ("d", 1, 900)]:
            for i in range(count):
                task_id = f"{name}{i}"
                tasks.append(task(task_id, name))
                rows.append(row(task_id, tokens + i))
        groups, aliases, merges = group_rows_by_qtype(tasks, rows)
        self.assertEqual(set(groups), {"a", "b"})
        self.assertEqual(aliases, {"c": "a", "d": "b"})
        self.assertEqual(merges, aliases)
        self.assertEqual(len(groups["a"]), 7)
        self.assertEqual(len(groups["b"]), 6)

    def test_all_small_groups_merge_together(self):
        tasks = [task("a1", "a"), task("a2", "a"), task("b1", "b")]
        rows = [row("a1", 100), row("a2", 200), row("b1", 900)]
        groups, aliases, _ = group_rows_by_qtype(tasks, rows)
        self.assertEqual(set(groups), {"all"})
        self.assertEqual(len(groups["all"]), 3)
        self.assertEqual(aliases, {"a": "all", "b": "all"})

    def test_small_group_threshold_counts_unique_tasks(self):
        tasks = [task(f"n{i}", "numeric") for i in range(4)]
        tasks += [task(f"h{i}", "how") for i in range(5)]
        rows = []
        for item in tasks[:4]:
            rows.extend([row(item["id"], 100), row(item["id"], 200)])
        rows.extend(row(item["id"], 300) for item in tasks[4:])
        groups, aliases, _ = group_rows_by_qtype(tasks, rows)
        self.assertEqual(set(groups), {"how"})
        self.assertEqual(aliases, {"numeric": "how"})
        self.assertEqual(len(groups["how"]), 13)


class TestLoadRows(unittest.TestCase):
    def test_pool_runs_keeps_repeated_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "a.json"
            run_b = root / "b.json"
            run_a.write_text(json.dumps({"rows": [row("x", 100)]}), encoding="utf-8")
            run_b.write_text(json.dumps({"rows": [row("x", 200)]}), encoding="utf-8")
            paths = [str(run_a), str(run_b)]
            self.assertEqual([r["tokens"] for r in load_rows(paths)], [100])
            self.assertEqual([r["tokens"] for r in load_rows(paths, pool_runs=True)], [100, 200])


class TestCalibrateMain(unittest.TestCase):
    def test_multiple_runs_and_qtype_writeback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [task(f"mh{i}", "multihop") for i in range(5)]
            tasks_path = root / "tasks.json"
            tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
            run_a = root / "a.json"
            run_b = root / "b.json"
            run_a.write_text(json.dumps({"rows": [row(f"mh{i}", (i + 1) * 100, i + 1) for i in range(3)]}), encoding="utf-8")
            run_b.write_text(json.dumps({"rows": [row("mh0", 9999, 8)] + [row(f"mh{i}", (i + 1) * 100, i + 1) for i in range(3, 5)]}), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--tasks", str(tasks_path), "--run", str(run_a), str(run_b), "--by", "qtype"])
            self.assertEqual(code, 0)
            self.assertIn("实际 token 中位数 300", output.getvalue())
            self.assertIn("样本 5 条", output.getvalue())
            written = json.loads(tasks_path.read_text(encoding="utf-8"))
            self.assertTrue(all(t["stage_limits"] == {"steps": 5, "tokens": 500} for t in written))

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = [task("x", "how")]
            tasks_path = root / "tasks.json"
            tasks_path.write_text(json.dumps(original), encoding="utf-8")
            run_path = root / "run.json"
            run_path.write_text(json.dumps({"rows": [row("x", 100)]}), encoding="utf-8")
            self.assertEqual(main(["--tasks", str(tasks_path), "--run", str(run_path), "--dry-run"]), 0)
            self.assertEqual(json.loads(tasks_path.read_text(encoding="utf-8")), original)


if __name__ == "__main__":
    unittest.main()