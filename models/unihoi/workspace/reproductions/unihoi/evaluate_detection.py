"""Assemble all image attempts and evaluate without fabricating predictions."""

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

from .export_weights import atomic_json
from .prediction import action_roles, native_records
from .tokenizer import ROOT, digest
from .detection import ANNOTATIONS


class EmptySafeNumpy:
    """The 2017 evaluator asserts max(recall) even when a class has no output.

    Define only that empty reduction as zero. All actual predictions, matching,
    ranking and AP integration stay in the original evaluator.
    """
    def __init__(self):
        self.empty_max_calls = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def amax(self, values, *args, **kwargs):
        if np.asarray(values).size == 0 and not args and not kwargs:
            self.empty_max_calls += 1
            return 0.
        return np.amax(values, *args, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    args = p.parse_args()
    source = args.predictions.resolve()
    protocol = json.loads((source / "protocol.json").read_text())
    ids_file = ROOT.parent / "data/v-coco/data/splits/vcoco_test.ids"
    ids = [int(v) for v in ids_file.read_text().split()]
    if (protocol["mode"] != "test" or len(ids) != 4946 or len(set(ids)) != 4946
            or protocol["image_ids"] != ids or protocol["split_sha256"] != digest(ids_file)):
        raise ValueError("Full official test inference required")
    actions = action_roles(json.loads(ANNOTATIONS.read_text())["channels"])
    fingerprint = digest(source / "protocol.json")
    results, hashes, errors, empty = [], {}, Counter(), []
    for image_id in ids:
        path = source / "images" / f"{image_id:012d}.json"
        result = json.loads(path.read_text())
        if result["image_id"] != image_id or result["protocol_sha256"] != fingerprint:
            raise ValueError("Mismatched image/protocol, refusing partial evaluation")
        rows = native_records(result["events"], actions)
        results.extend(rows)
        hashes[str(image_id)] = digest(path)
        if not rows:
            empty.append(image_id)
        errors.update(e["reason"] for e in result["errors"])
    output = source / "official_test"
    output.mkdir(exist_ok=True)
    cache = output / "predictions.pkl"
    if cache.exists():
        old_coverage = json.loads((output / "coverage.json").read_text())
        if old_coverage["image_result_sha256"] != hashes or old_coverage["cache_sha256"] != digest(cache):
            raise ValueError("Existing cache belongs to another inference attempt")
    else:
        temp = cache.with_suffix(".partial")
        with temp.open("wb") as handle:
            pickle.dump(results, handle, protocol=4)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(cache)
        atomic_json(output / "coverage.json", {"images": len(ids), "image_ids": ids,
            "empty_image_ids": empty, "records": len(results), "errors": dict(errors),
            "image_result_sha256": hashes, "cache_sha256": digest(cache),
            "inference_protocol_sha256": fingerprint, "checkpoint": protocol["weights_sha256"]})
    metric = output / "official_metrics.json"
    if metric.exists():
        previous = json.loads(metric.read_text())
        if previous["inputs"][str(cache)] != digest(cache):
            raise ValueError("Existing metric/cache mismatch")
        print(f"Already complete: {metric}", flush=True)
        return
    wrapper = ROOT / "vcoco_eval/scripts/evaluate_vcoco_official.py"
    spec = importlib.util.spec_from_file_location("unihoi_official_wrapper", wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_loader = module._load_evaluator
    safe = EmptySafeNumpy()

    def load(path):
        evaluator = original_loader(path)
        # Each loader creates a private evaluator module. Never monkeypatch
        # process-global NumPy or the original evaluator file on disk.
        evaluator._do_role_eval.__globals__["np"] = safe
        return evaluator

    module._load_evaluator = load
    data = ROOT.parent / "data/v-coco"
    pending = output / "metrics.pending.json"
    sys.argv = [str(wrapper), "--cache", str(cache), "--output-json", str(pending),
        "--vsrl-json", str(data / "data/vcoco/vcoco_test.json"),
        "--coco-json", str(data / "data/instances_vcoco_all_2014.json"),
        "--split-ids", str(ids_file), "--evaluator", str(data / "vsrl_eval.py"), "--agent-ap"]
    module.main()
    value = json.loads(pending.read_text())
    value.update({"evidence_type": protocol["evidence_type"], "full_unihoi_reproduction": False,
        "empty_recall_guard": {"calls": safe.empty_max_calls,
            "semantics": "Empty predicted class has AP=0; no sentinel/dummy predictions added; otherwise original evaluator unchanged"},
        "inference_protocol_sha256": fingerprint, "coverage_sha256": digest(output / "coverage.json"),
        "evaluation_adapter_sha256": digest(Path(__file__))})
    atomic_json(metric, value)
    print(f"Completed fixed-checkpoint official test: {metric}", flush=True)


if __name__ == "__main__":
    main()
