"""CPU-only rescoring of hash-linked, trusted local native prediction caches."""

from compute_guard import require_compute_step

if __name__ == "__main__":
    require_compute_step()

import argparse
from collections import defaultdict
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import time

import numpy as np

import scorer


class CacheTemplate(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)


class TrustedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "utils" and name == "CacheTemplate":
            return CacheTemplate
        return super().find_class(module, name)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def score(model, ledger_path, output):
    start = time.monotonic()
    code_paths = (Path(__file__), Path(scorer.__file__), Path(__file__).with_name("compute_guard.py"))
    code = {p.name: sha256(p) for p in code_paths}
    ledger = json.loads(ledger_path.read_text())
    if ledger["evaluation_split"] != "test" or ledger["evaluation_split_image_ids"] != 4946:
        raise ValueError("Full test ledger required")
    paths = [Path(p) for p in ledger["inputs"]]
    def one(predicate):
        matches = [p for p in paths if predicate(p)]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous inputs: {matches}")
        return matches[0]
    inputs = {"cache": one(lambda p: p.suffix in (".pkl", ".pickle")),
              "coco": one(lambda p: p.name == "instances_vcoco_all_2014.json"),
              "vsrl": one(lambda p: p.name == "vcoco_test.json"),
              "split": one(lambda p: p.name == "vcoco_test.ids"),
              "official_evaluator": one(lambda p: p.name == "vsrl_eval.py")}
    hashes = {k: sha256(p) for k, p in inputs.items()}
    for key, path in inputs.items():
        if hashes[key] != ledger["inputs"][str(path)]:
            raise ValueError(f"Changed input {path}")
    spec = importlib.util.spec_from_file_location("official_vcoco", inputs["official_evaluator"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if "bool" not in np.__dict__:
        np.bool = np.bool_
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = module.VCOCOeval(*(str(inputs[k]) for k in ("vsrl", "coco", "split")))
        database = evaluator._get_vcocodb()
    ids = np.atleast_1d(np.loadtxt(inputs["split"], dtype=int)).tolist()
    if len(database) != 4946 or len(set(ids)) != 4946 or {int(e["id"]) for e in database} != set(ids):
        raise ValueError("Incorrect test coverage")
    with inputs["cache"].open("rb") as handle:
        cache = TrustedUnpickler(handle).load()
    if not isinstance(cache, list):
        raise ValueError("Expected list cache")
    by_image = defaultdict(list)
    for record in cache:
        by_image[int(record["image_id"])].append(record)
    if len(cache) != ledger["cache_records"] or len(by_image) != ledger["cache_unique_image_ids"]:
        raise ValueError("Cache coverage mismatch")
    del cache
    gc.collect()
    result = scorer.evaluate(database, by_image, evaluator.actions, evaluator.roles)
    if result["actions"] != 21 or result["gt_sets"] != 15222:
        raise ValueError("Unexpected action/set support")
    if code != {p.name: sha256(p) for p in code_paths}:
        raise ValueError("Scorer changed during execution")
    result.update(model=model, created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
                  source_ledger=str(ledger_path), source_ledger_sha256=sha256(ledger_path),
                  inputs={k: {"path": str(p), "sha256": hashes[k]} for k, p in inputs.items()},
                  source_code_sha256=code, paired_official_role_ap_percent=ledger["all_metrics"])
    write_json(output, result)
    write_json(output.parent / f"{model}.official_source.json", ledger)
    print(model, [result["results"][f"scenario_{s}"]["set_map_percent"] for s in (1, 2)], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    for item in json.loads(args.manifest.read_text()):
        if args.model and item["model"] != args.model:
            continue
        output = args.output / f"{item['model']}.json"
        if output.exists():
            raise FileExistsError(output)
        print(f"Starting {item['model']}", flush=True)
        score(item["model"], Path(item["ledger"]), output)


if __name__ == "__main__":
    main()
