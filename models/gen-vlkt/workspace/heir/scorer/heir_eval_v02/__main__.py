import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy
import scipy

from .core import evaluate
from .diagnostics import binding_ap, calibrate_operating_point, negative_report


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
    parser = argparse.ArgumentParser(description="HEIR isolated research evaluator; never opens an annotation database")
    commands = parser.add_subparsers(dest="command", required=True)
    detection = commands.add_parser("detection")
    detection.add_argument("--gt", required=True)
    detection.add_argument("--pred", required=True)
    detection.add_argument("--iou", type=float, default=0.5)
    detection.add_argument("--event-threshold", type=float, default=0.5)
    binding = commands.add_parser("binding")
    binding.add_argument("--candidates", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--candidates", required=True)
    calibration.add_argument("--split", required=True)
    calibration.add_argument("--recall", type=float, default=0.9)
    negatives = commands.add_parser("negatives")
    negatives.add_argument("--candidates", required=True)
    negatives.add_argument("--calibration", required=True)
    negatives.add_argument("--split", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        if args.command == "detection":
            result = evaluate(load(args.gt), load(args.pred), args.iou, args.event_threshold)
        elif args.command == "binding":
            result = binding_ap(load(args.candidates))
        elif args.command == "calibrate":
            result = calibrate_operating_point(load(args.candidates), args.split, args.recall)
        else:
            calibration_data = load(args.calibration)
            if not isinstance(calibration_data, dict):
                raise ValueError("calibration must be an object")
            calibration_data = calibration_data.get("result", calibration_data)
            result = negative_report(load(args.candidates), calibration_data, args.split)
        inputs = {key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                  for key, value in vars(args).items() if key in {"gt", "pred", "candidates", "calibration"}}
        result = {"result": result, "input_sha256": inputs, "elapsed_seconds": time.perf_counter() - started,
                  "environment": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}}
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Evaluation refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
