#!/usr/bin/env python3
"""Run the original V-COCO Scenario 1/2 evaluator on a declared split."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


AVERAGE_RE = re.compile(
    r"^Average Role \[(scenario_[12])\] AP = ([0-9]+(?:\.[0-9]+)?)"
    r"(, omitting the action \"point\")?$"
)
ROLE_RE = re.compile(
    r"^([a-z_]+)-(obj|instr): AP = ([0-9]+(?:\.[0-9]+)?) "
    r"\(#pos = ([0-9]+)\)$"
)
AGENT_RE = re.compile(
    r"^([a-z_]+): AP = ([0-9]+(?:\.[0-9]+)?) \(#pos = ([0-9]+)\)$"
)
AGENT_AVERAGE_RE = re.compile(r"^Average Agent AP = ([0-9]+(?:\.[0-9]+)?)$")
DUAL_ROLE_CLASSES = (
    "hit-obj",
    "hit-instr",
    "eat-obj",
    "eat-instr",
    "cut-obj",
    "cut-instr",
)


class _CacheTemplateCompat(defaultdict):
    """Inspect public PViC/UPT caches before the legacy evaluator reloads them."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.update(*args, **kwargs)

    def __missing__(self, key):
        if str(key).split("_")[-1] == "agent":
            return 0.0
        return [0.0, 0.0, 0.1, 0.1, 0.0]


class _VCOCOCacheUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "utils" and name == "CacheTemplate":
            return _CacheTemplateCompat
        return super().find_class(module, name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("vcoco_official_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official V-COCO evaluator from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VCOCOeval


def _parse_metrics(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        match = AVERAGE_RE.match(line.strip())
        if match is None:
            continue
        scenario, value, omitted = match.groups()
        suffix = "omit_point" if omitted else "including_point"
        metrics[f"role_ap_{scenario}_{suffix}"] = float(value)
    required = {
        "role_ap_scenario_1_including_point",
        "role_ap_scenario_1_omit_point",
        "role_ap_scenario_2_including_point",
        "role_ap_scenario_2_omit_point",
    }
    missing = required.difference(metrics)
    if missing:
        raise RuntimeError(
            "Could not parse all official V-COCO metrics; "
            f"missing {sorted(missing)}."
        )
    return metrics


def _parse_role_metrics(stdout: str) -> dict[str, dict[str, dict[str, float | int]]]:
    per_scenario: dict[str, dict[str, dict[str, float | int]]] = {}
    current: dict[str, dict[str, float | int]] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == "---------Reporting Role AP (%)------------------":
            current = {}
            continue
        role_match = ROLE_RE.match(stripped)
        if role_match is not None:
            action, role, value, positives = role_match.groups()
            current[f"{action}-{role}"] = {
                "ap": float(value),
                "positives": int(positives),
            }
            continue
        average_match = AVERAGE_RE.match(stripped)
        if average_match is not None and average_match.group(3) is None:
            scenario = average_match.group(1)
            if not current:
                raise RuntimeError(f"No per-role metrics preceded {scenario}.")
            per_scenario[scenario] = dict(current)
    if set(per_scenario) != {"scenario_1", "scenario_2"}:
        raise RuntimeError("Could not parse both V-COCO per-role metric blocks.")
    return per_scenario


def _dual_role_summary(
    per_role: dict[str, dict[str, dict[str, float | int]]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for scenario, metrics in per_role.items():
        missing = [name for name in DUAL_ROLE_CLASSES if name not in metrics]
        if missing:
            raise RuntimeError(f"Missing dual-role V-COCO classes: {missing}.")
        values = [float(metrics[name]["ap"]) for name in DUAL_ROLE_CLASSES]
        summary[scenario] = {
            "classes": {name: metrics[name] for name in DUAL_ROLE_CLASSES},
            "macro_ap": float(np.mean(values)),
        }
    return summary


def _parse_agent_metrics(stdout: str) -> dict[str, object]:
    actions: dict[str, dict[str, float | int]] = {}
    average = None
    in_agent_block = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == "---------Reporting Agent AP (%)------------------":
            in_agent_block = True
            continue
        if not in_agent_block:
            continue
        match = AGENT_RE.match(stripped)
        if match is not None:
            action, value, positives = match.groups()
            actions[action] = {"ap": float(value), "positives": int(positives)}
            continue
        average_match = AGENT_AVERAGE_RE.match(stripped)
        if average_match is not None:
            average = float(average_match.group(1))
            break
    if average is None or not actions:
        raise RuntimeError("Could not parse official V-COCO agent AP output.")
    return {"macro_ap": average, "per_action": actions}


def _validate_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V-COCO evaluation inputs: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--vsrl-json", required=True, type=Path)
    parser.add_argument("--coco-json", required=True, type=Path)
    parser.add_argument("--split-ids", required=True, type=Path)
    parser.add_argument("--split-name", choices=("test", "val"), default="test")
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--iou-thr", default=0.5, type=float)
    parser.add_argument("--harness-root", type=Path, default=None)
    parser.add_argument("--agent-ap", action="store_true")
    args = parser.parse_args()
    inputs = [
        args.cache,
        args.vsrl_json,
        args.coco_json,
        args.split_ids,
        args.evaluator,
    ]
    _validate_paths(inputs)
    if args.iou_thr != 0.5:
        raise ValueError("The frozen V-COCO evaluator protocol requires IoU 0.5.")
    if args.harness_root is not None:
        harness_root = args.harness_root.resolve()
        sys.path.insert(0, str(harness_root))
        pocket_root = harness_root / "pocket"
        if pocket_root.is_dir():
            # Some public caches pickle ``utils.CacheTemplate``. The original
            # evaluator reopens the file, so its historical package layout must
            # be importable even though our first load uses the compatibility class.
            sys.path.insert(0, str(pocket_root))

    with args.cache.open("rb") as handle:
        cached = _VCOCOCacheUnpickler(handle).load()
    if not isinstance(cached, list):
        raise TypeError("The V-COCO cache must be a list of detection records.")
    image_ids = {
        int(record["image_id"])
        for record in cached
        if isinstance(record, dict) and "image_id" in record
    }
    split_ids = {
        int(value)
        for value in np.atleast_1d(np.loadtxt(args.split_ids, dtype=np.int64)).tolist()
    }
    outside = image_ids.difference(split_ids)
    if outside:
        raise ValueError(
            f"Cache contains {len(outside)} image ids outside the declared "
            f"{args.split_name} split."
        )

    # The frozen 2017 evaluator uses the NumPy alias removed in NumPy 1.24.
    if "bool" not in np.__dict__:
        np.bool = np.bool_  # type: ignore[attr-defined]
    evaluator_cls = _load_evaluator(args.evaluator)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        evaluator = evaluator_cls(
            str(args.vsrl_json), str(args.coco_json), str(args.split_ids)
        )
        original_collect = evaluator._collect_detections_for_image
        detections_by_image: dict[int, list] = defaultdict(list)
        for record in cached:
            if isinstance(record, dict) and "image_id" in record:
                detections_by_image[int(record["image_id"])].append(record)
        prepared_detections: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def prepare_records(records):
            agents = np.zeros(
                (len(records), 4 + evaluator.num_actions), dtype=np.float32
            )
            roles = np.zeros(
                (len(records), 5 * evaluator.num_actions, 2), dtype=np.float32
            )
            for index, record in enumerate(records):
                agents[index, :4] = record["person_box"]
                for action_id in range(evaluator.num_actions):
                    action = evaluator.actions[action_id]
                    for role_index, role in enumerate(evaluator.roles[action_id]):
                        key = f"{action}_{role}"
                        if role == "agent":
                            agents[index, 4 + action_id] = record[key]
                        else:
                            start = 5 * action_id
                            roles[index, start : start + 5, role_index - 1] = record[key]
            return agents, roles

        def indexed_collect(_detections, image_id):
            key = int(image_id)
            if key not in prepared_detections:
                prepared_detections[key] = prepare_records(
                    detections_by_image.get(key, [])
                )
            return prepared_detections[key]

        # The original routine linearly scans every prediction for every image.
        # Indexing by image id is output-equivalent and leaves all matching/AP code intact.
        checked = 0
        for image_id, records in detections_by_image.items():
            expected_agents, expected_roles = original_collect(records, image_id)
            actual_agents, actual_roles = prepare_records(records)
            if not np.array_equal(expected_agents, actual_agents, equal_nan=True):
                raise RuntimeError("Indexed V-COCO agent cache failed equivalence check.")
            if not np.array_equal(expected_roles, actual_roles, equal_nan=True):
                raise RuntimeError("Indexed V-COCO role cache failed equivalence check.")
            checked += 1
            if checked == 5:
                break
        evaluator._collect_detections_for_image = indexed_collect
        if args.agent_ap:
            evaluator._do_agent_eval(
                evaluator._get_vcocodb(), str(args.cache), ovr_thresh=args.iou_thr
            )
        evaluator._do_eval(str(args.cache), ovr_thresh=args.iou_thr)
    stdout = buffer.getvalue()
    print(stdout, end="")
    metrics = _parse_metrics(stdout)
    per_role_metrics = _parse_role_metrics(stdout)
    agent_metrics = _parse_agent_metrics(stdout) if args.agent_ap else None

    payload = {
        "schema": "vcoco_split_eval_v3",
        "evaluator": "original_vcoco_vsrl_eval",
        "evaluation_split": args.split_name,
        "official_test_claim_eligible": args.split_name == "test",
        "performance_patch": "output_equivalent_preallocated_image_id_detection_index",
        "collector_equivalence_images_checked": checked,
        "iou_threshold": args.iou_thr,
        "primary_comparable_metrics": {
            "scenario_1_role_ap_omit_point": metrics[
                "role_ap_scenario_1_omit_point"
            ],
            "scenario_2_role_ap_omit_point": metrics[
                "role_ap_scenario_2_omit_point"
            ],
        },
        "all_metrics": metrics,
        "per_role_metrics": per_role_metrics,
        "dual_role_disambiguation": _dual_role_summary(per_role_metrics),
        "agent_metrics": agent_metrics,
        "cache_records": len(cached),
        "cache_unique_image_ids": len(image_ids),
        "evaluation_split_image_ids": len(split_ids),
        "inputs": {
            str(path.resolve()): _sha256(path) for path in inputs
        },
        "official_stdout": stdout,
        "unix_time": time.time(),
    }
    if args.split_name == "test":
        # Preserve the historical field consumed by existing result ledgers.
        payload["official_test_image_ids"] = len(split_ids)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {args.split_name} metric artifact to "
        f"{args.output_json.resolve()}"
    )


if __name__ == "__main__":
    main()
