import unittest

from refusal_eval.dataset import Record
from refusal_eval.metrics import class_scores, confusion, over_refusal_rate, under_refusal_rate


def rec(id_, kind, label):
    return Record(id_, kind, "", "stop", label)


RECORDS = [
    rec("1", "benign", "answer"),
    rec("2", "benign", "refusal"),
    rec("3", "harmful", "refusal"),
    rec("4", "harmful", "answer"),
]


class MetricsTest(unittest.TestCase):
    def test_perfect_detector(self):
        predicted = [r.label for r in RECORDS]
        by_label = {s.label: s for s in class_scores(confusion(RECORDS, predicted))}
        self.assertEqual(by_label["refusal"].f1, 1.0)
        self.assertEqual(by_label["refusal"].support, 2)

    def test_missed_refusal_lowers_recall_not_precision(self):
        predicted = ["answer", "answer", "refusal", "answer"]
        by_label = {s.label: s for s in class_scores(confusion(RECORDS, predicted))}
        self.assertEqual(by_label["refusal"].precision, 1.0)
        self.assertEqual(by_label["refusal"].recall, 0.5)

    def test_absent_class_scores_zero_without_dividing_by_zero(self):
        by_label = {s.label: s for s in class_scores(confusion(RECORDS, [r.label for r in RECORDS]))}
        self.assertEqual(by_label["truncated"].f1, 0.0)
        self.assertEqual(by_label["truncated"].support, 0)

    def test_rates_from_labels(self):
        labels = [r.label for r in RECORDS]
        self.assertEqual(over_refusal_rate(RECORDS, labels), 0.5)
        self.assertEqual(under_refusal_rate(RECORDS, labels), 0.5)

    def test_rates_follow_the_given_labels(self):
        self.assertEqual(under_refusal_rate(RECORDS, ["answer", "answer", "refusal", "refusal"]), 0.0)

    def test_rate_with_no_matching_prompts_is_zero(self):
        self.assertEqual(over_refusal_rate([rec("h", "harmful", "refusal")], ["refusal"]), 0.0)


if __name__ == "__main__":
    unittest.main()
