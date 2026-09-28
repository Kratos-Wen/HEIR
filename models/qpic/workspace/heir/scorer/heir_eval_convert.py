#!/usr/bin/env python3
"""Convert HEIR v1.0 ground truth and relation predictions to heir_eval v0.2 JSON and score them.

GT: annotations/{split}.json -> heir-eval-research-0.2 ground truth. Classes are the
(verb, noun, role) tuples present in the scored split, with train_count from
annotations/train.json (unseen 0, rare 1-9, non-rare >=10, as in the release README).
Events are not annotated in v1.0, so events_complete is False and only relation AP
is meaningful.

Predictions: one record per image with its entities (nodes) and scored (subject, entity, verb, role)
edges (`relations`).

agent_only rule (release README): on SWiG-origin test images only predictions whose
subject matches an annotated agent (IoU >= 0.5) and whose verb equals that agent's
annotated verb are scored; every other prediction on such an image is ignored.
heir_eval v0.2 has no ignore masks, so this script applies the rule by removing the
ignored predictions before scoring and reports how many were removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from heir_data import HEIR_ROLE_NAMES, HEIRVocabulary, load_heir_split  # noqa: E402

SCHEMA = "heir-eval-research-0.2"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build_ground_truth(root, split, vocab, train_images, split_images):
    train_count = Counter((vocab.verbs[v], vocab.objects[img.categories[o]], vocab.role_names[r])
                          for img in train_images for _, o, v, r in img.relations)
    observed, images = set(), []
    for img in split_images:
        entities = [{"id": f"box_{i + 1}", "noun": vocab.objects[c], "bbox": list(map(float, box))}
                    for i, (box, c) in enumerate(zip(img.boxes, img.categories))]
        relations = []
        for k, (s, o, v, r) in enumerate(img.relations):
            key = (vocab.verbs[v], vocab.objects[img.categories[o]], vocab.role_names[r])
            observed.add(key)
            relations.append({"id": f"rel_{k}", "subject_id": f"box_{s + 1}", "entity_id": f"box_{o + 1}",
                              "verb": key[0], "role": key[2]})
        images.append({"id": img.image_id, "width": img.width, "height": img.height,
                       "cluster_id": img.source or "unknown", "coverage": "adjudicated_closed_world",
                       "events_complete": False, "entities": entities, "relations": relations, "events": []})
    classes = [{"verb": v, "noun": n, "role": r, "train_count": int(train_count[(v, n, r)])}
               for v, n, r in sorted(observed)]
    return {"schema_version": SCHEMA, "annotation_policy_id": f"HEIR {vocab.version} {split} native relations",
            "taxonomy": {"verbs": list(vocab.verbs), "nouns": list(vocab.objects), "roles": list(HEIR_ROLE_NAMES)},
            "classes": classes, "images": images}


def build_predictions(records, vocab, split_images):
    by_id = {img.image_id: img for img in split_images}
    images, stats = [], Counter()
    for record in records:
        img = by_id[record["image_id"]]
        width, height = float(img.width), float(img.height)
        entities, valid_node = [], {}
        for node in record["nodes"]:
            x1, y1, x2, y2 = node["box"]
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            if not (x1 < x2 and y1 < y2):
                stats["degenerate_nodes"] += 1
                continue
            valid_node[node["id"]] = f"node_{node['id']}"
            entities.append({"id": valid_node[node["id"]], "noun": vocab.objects[int(node["category_id"])],
                             "bbox": [x1, y1, x2, y2], "score": float(node["detector_score"])})
        boxes = {e["id"]: e["bbox"] for e in entities}
        # agent_only: agents and verbs that may be scored on this image.
        agent_verbs = None
        if img.evaluation == "agent_only":
            agent_verbs = {}
            for s, _, v, _ in img.relations:
                agent_verbs.setdefault(s, set()).add(vocab.verbs[v])
        relations, key_to_id = [], {}
        for edge in record["relations"]:
            if edge["entity"] is None or edge["subject"] not in valid_node or edge["entity"] not in valid_node:
                stats["skipped_null_or_invalid_edges"] += 1
                continue
            subject_id, entity_id = valid_node[edge["subject"]], valid_node[edge["entity"]]
            if agent_verbs is not None:
                subject_box = boxes[subject_id]
                scored = any(iou(subject_box, img.boxes[a]) >= 0.5 and edge["verb"] in verbs
                             for a, verbs in agent_verbs.items())
                if not scored:
                    stats["agent_only_ignored_predictions"] += 1
                    continue
            relation_id = f"rel_{len(relations)}"
            relations.append({"id": relation_id, "subject_id": subject_id, "entity_id": entity_id,
                              "verb": edge["verb"], "role": edge["role"], "score": float(edge["score"])})
            key_to_id.setdefault((subject_id, entity_id, edge["verb"], edge["role"]), relation_id)
        events = []
        for event in record.get("events", []):
            choice = event.get("best_nonempty")
            if not choice or event["subject_id"] not in valid_node:
                continue
            members = []
            for e in choice["edges"]:
                if e["entity_id"] is None or e["entity_id"] not in valid_node:
                    continue
                key = (valid_node[event["subject_id"]], valid_node[e["entity_id"]], event["action"], e["role"])
                if key in key_to_id and key_to_id[key] not in members:
                    members.append(key_to_id[key])
            if members:
                events.append({"id": f"event_{len(events)}", "verb": event["action"], "relation_ids": members,
                               "score": float(min(1.0, max(0.0, choice["score"])))})
        images.append({"id": img.image_id, "entities": entities, "relations": relations, "events": events})
        stats["images"] += 1
        stats["relations"] += len(relations)
    return {"schema_version": SCHEMA, "images": images}, dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heir-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--relations", type=Path, required=True, help="relations.jsonl from --cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    vocab = HEIRVocabulary(args.heir_root)
    train_images = load_heir_split(args.heir_root, "train", vocab)
    split_images = load_heir_split(args.heir_root, args.split, vocab)
    records = [json.loads(line) for line in args.relations.read_text().splitlines() if line.strip()]
    declared = [img.image_id for img in split_images]
    if [r["image_id"] for r in records] != declared:
        raise ValueError("Prediction coverage must equal the declared split in order (empty images included).")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_path, pred_path = args.output_dir / "gt.json", args.output_dir / "pred.json"
    gt_path.write_text(json.dumps(build_ground_truth(args.heir_root, args.split, vocab, train_images, split_images)))
    predictions, stats = build_predictions(records, vocab, split_images)
    pred_path.write_text(json.dumps(predictions))
    command = [sys.executable, "-m", "heir_eval_v02", "detection", "--gt", str(gt_path), "--pred", str(pred_path),
               "--iou", str(args.iou)]
    env = dict(os.environ, PYTHONPATH=str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    (args.output_dir / "heir_eval_stdout.txt").write_text(completed.stdout)
    (args.output_dir / "heir_eval_stderr.txt").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError("heir_eval refused the inputs:\n" + completed.stderr[-4000:])
    result = json.loads(completed.stdout)
    relation = result["result"]["relation"]
    summary = {"split": args.split, "images": len(split_images), "iou": args.iou,
               "mAP": relation["mAP"], "rare_mAP": relation["rare_mAP"], "non_rare_mAP": relation["non_rare_mAP"],
               "unseen_mAP": relation["unseen_mAP"], "per_role": relation["per_role"],
               "unsupported_prediction_count": relation["unsupported_prediction_count"],
               "classes": len(relation["classes"]), "conversion": stats,
               "agent_only_rule": "predictions on agent_only images are scored only when the subject matches an "
                                  "annotated agent (IoU>=0.5) with the same verb; other predictions removed before scoring",
               "inputs": {"relations": str(args.relations), "relations_sha256": sha256(args.relations),
                          "gt_sha256": sha256(gt_path), "pred_sha256": sha256(pred_path)},
               "evaluator": "heir_eval " + SCHEMA}
    (args.output_dir / "heir_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("split", "mAP", "rare_mAP", "non_rare_mAP", "unseen_mAP", "classes", "conversion")}, indent=2))


if __name__ == "__main__":
    main()
