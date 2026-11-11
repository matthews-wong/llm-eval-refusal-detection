"""Render the detector report as plain text."""

from .dataset import Record
from .detector import LABELS, classify
from .metrics import class_scores, confusion, over_refusal_rate, under_refusal_rate


def render(records: list[Record]) -> str:
    predicted = [classify(r.response, r.finish_reason) for r in records]
    truth = [r.label for r in records]
    counts = confusion(records, predicted)
    correct = sum(t == p for t, p in zip(truth, predicted, strict=True))

    lines = [f"responses: {len(records)}  accuracy: {correct / len(records):.3f}", ""]
    lines.append(f"{'class':<10} {'prec':>6} {'recall':>6} {'f1':>6} {'n':>4}")
    for s in class_scores(counts):
        lines.append(f"{s.label:<10} {s.precision:>6.2f} {s.recall:>6.2f} {s.f1:>6.2f} {s.support:>4}")

    lines += ["", "confusion (rows: true, cols: predicted)"]
    lines.append(f"{'':<10}" + "".join(f"{lab:>10}" for lab in LABELS))
    for true in LABELS:
        lines.append(f"{true:<10}" + "".join(f"{counts.get((true, pred), 0):>10}" for pred in LABELS))

    lines += ["", "refusal rates       labelled  detected"]
    lines.append(f"over-refusal        {over_refusal_rate(records, truth):>8.3f}  {over_refusal_rate(records, predicted):>8.3f}")
    lines.append(f"under-refusal       {under_refusal_rate(records, truth):>8.3f}  {under_refusal_rate(records, predicted):>8.3f}")

    misses = [(r, p) for r, p in zip(records, predicted, strict=True) if r.label != p]
    if misses:
        lines += ["", "misclassified"]
        lines += [f"  {r.id}: labelled {r.label}, detected {p}" for r, p in misses]
    return "\n".join(lines)
