import sys

from .dataset import load
from .report import render

USAGE = "usage: python3 -m refusal_eval <responses.jsonl>"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        records = load(argv[1])
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not records:
        print(f"error: {argv[1]} has no records", file=sys.stderr)
        return 1
    print(render(records))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
