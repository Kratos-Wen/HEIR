import argparse
import json
from pathlib import Path
import sys

from .core import evaluate


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    def reject_constant(value):
        raise ValueError(f"invalid JSON number {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=reject_duplicates, parse_constant=reject_constant)


def main():
    parser = argparse.ArgumentParser(description="HEIR relation and set evaluator")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--event-threshold", type=float, default=0.5)
    args = parser.parse_args()
    try:
        result = evaluate(load(args.gt), load(args.pred), args.iou, args.event_threshold)
        print(json.dumps({"result": result}, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Evaluation refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
