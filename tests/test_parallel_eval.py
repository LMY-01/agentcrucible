import json
import tempfile
import unittest
from pathlib import Path

from scripts.parallel_eval import read_chunk_rows


class TestReadChunkRows(unittest.TestCase):
    def test_nonzero_exit_with_result_preserves_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            path.write_text(json.dumps({"rows": [{"id": "x"}]}), encoding="utf-8")
            self.assertEqual(read_chunk_rows(1, path), [{"id": "x"}])

    def test_missing_result_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            self.assertIsNone(read_chunk_rows(0, path))


if __name__ == "__main__":
    unittest.main()