import contextlib
import io
import unittest

from refusal_eval.__main__ import main
from refusal_eval.dataset import load
from refusal_eval.report import render


class ReportTest(unittest.TestCase):
    def test_report_lists_misses_and_sizes(self):
        out = render(load("data/responses.jsonl"))
        self.assertIn("responses: 22", out)
        self.assertIn("b11: labelled refusal, detected answer", out)

    def test_main_prints_report(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(main(["prog", "data/responses.jsonl"]), 0)
        self.assertIn("confusion", buf.getvalue())

    def test_main_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["prog"]), 2)

    def test_main_missing_file(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(main(["prog", "nope.jsonl"]), 1)
        self.assertIn("nope.jsonl", err.getvalue())


if __name__ == "__main__":
    unittest.main()
