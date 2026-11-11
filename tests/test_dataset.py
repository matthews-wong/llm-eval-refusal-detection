import json
import os
import tempfile
import unittest

from refusal_eval.dataset import load

ROW = {"id": "x1", "prompt_kind": "benign", "response": "ok", "finish_reason": "stop", "label": "answer"}


def write(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    fh.write("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    fh.close()
    return fh.name


class LoadTest(unittest.TestCase):
    def check_error(self, rows, fragment):
        path = write(rows)
        self.addCleanup(os.unlink, path)
        with self.assertRaises(ValueError) as ctx:
            load(path)
        self.assertIn(fragment, str(ctx.exception))

    def test_loads_valid_rows_and_skips_blank_lines(self):
        path = write([ROW, "", {**ROW, "id": "x2"}])
        self.addCleanup(os.unlink, path)
        self.assertEqual([r.id for r in load(path)], ["x1", "x2"])

    def test_missing_field_names_it(self):
        self.check_error([{k: v for k, v in ROW.items() if k != "label"}], "['label']")

    def test_bad_label(self):
        self.check_error([{**ROW, "label": "maybe"}], "'maybe'")

    def test_bad_prompt_kind(self):
        self.check_error([{**ROW, "prompt_kind": "spicy"}], "'spicy'")

    def test_duplicate_id(self):
        self.check_error([ROW, ROW], "duplicate id 'x1'")

    def test_invalid_json_reports_line(self):
        self.check_error([ROW, "{nope"], ":2: invalid JSON")

    def test_shipped_fixture_loads(self):
        self.assertGreater(len(load("data/responses.jsonl")), 0)


if __name__ == "__main__":
    unittest.main()
