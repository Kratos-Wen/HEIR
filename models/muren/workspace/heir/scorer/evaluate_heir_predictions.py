#!/usr/bin/env python3
"""Score one-stage baseline predictions on HEIR v1.0 with heir_eval v0.2 (the scorer used for every evaluated system).

Input: predictions.jsonl, one line per image of the split, in split order:
  {"image_id": "cv_...", "boxes": [[x1,y1,x2,y2], ...] (original pixels),
   "labels": [object class ids, person=0], "box_scores": [...],
   "hois": [{"subject_id": i, "object_id": j, "category_id": interaction id, "score": s}, ...]}
Interaction ids index annotations/classes.json "interactions" ("<verb> <role>").
The agent_only rule, class buckets (unseen/rare/non-rare from train counts) and IoU 0.5
are those of heir_eval_convert.py, shared by every evaluated system.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from heir_eval_convert import build_ground_truth, build_predictions, sha256  # noqa: E402
from heir_data import HEIRVocabulary, load_heir_split  # noqa: E402


def collapse_roles(gt, pred):
    """Role-free HOI evaluation: every edge gets the placeholder role 'target', (subject, entity, action) duplicates merge."""
    from collections import Counter
    counts = Counter()
    for c in gt["classes"]:
        counts[(c["verb"], c["noun"])] += c["train_count"]
    gt["classes"] = [{"verb": v, "noun": n, "role": "target", "train_count": int(k)} for (v, n), k in sorted(counts.items())]
    for image in gt["images"]:
        seen, kept = set(), []
        for r in image["relations"]:
            key = (r["subject_id"], r["entity_id"], r["verb"])
            if key not in seen:
                seen.add(key); kept.append(dict(r, role="target"))
        image["relations"] = kept
    for image in pred["images"]:
        best = {}
        for r in image["relations"]:
            key = (r["subject_id"], r["entity_id"], r["verb"])
            if key not in best or r["score"] > best[key]["score"]:
                best[key] = dict(r, role="target")
        image["relations"] = list(best.values())
        image["events"] = []
    return gt, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heir-root", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-relations-per-image", type=int, default=1000)
    ap.add_argument("--image-subset", choices=("all", "shared_entity"), default="all")
    ap.add_argument("--hoi", action="store_true",
                    help="HOI mAP: ignore roles; classes are (action, noun); duplicate (subject, entity, action) edges keep the max score")
    a = ap.parse_args()
    classes = json.loads(a.classes.read_text())
    vocab = HEIRVocabulary(a.heir_root)
    if list(vocab.objects) != classes["objects"]:
        raise ValueError("Object order differs between classes.json and the HEIR vocabulary.")
    inter = classes["interaction_verb_role"]
    train = load_heir_split(a.heir_root, "train", vocab)
    split = load_heir_split(a.heir_root, a.split, vocab)
    records = []
    for line in a.predictions.read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        nodes = [{"id": i, "box": b, "category_id": int(l), "detector_score": float(s)}
                 for i, (b, l, s) in enumerate(zip(p["boxes"], p["labels"], p["box_scores"]))]
        hois = sorted(p["hois"], key=lambda h: -h["score"])[: a.max_relations_per_image]
        rels = [{"subject": int(h["subject_id"]), "entity": int(h["object_id"]), "verb": inter[h["category_id"]][0],
                 "role": inter[h["category_id"]][1], "score": float(h["score"])} for h in hois
                if int(h["subject_id"]) != int(h["object_id"])]
        records.append({"image_id": p["image_id"], "nodes": nodes, "relations": rels, "events": []})
    if [r["image_id"] for r in records] != [i.image_id for i in split]:
        raise ValueError("Predictions must cover the declared split in order (empty images included).")
    if a.image_subset == "shared_entity":
        # images whose ground truth has one entity participating in events of >= 2 different persons (same prediction file)
        keep = set()
        for img in split:
            owners = {}
            for s_, o, v, r in img.relations:
                owners.setdefault(o, set()).add(s_)
            if any(len(x) >= 2 for x in owners.values()):
                keep.add(img.image_id)
        split = [img for img in split if img.image_id in keep]
        records = [r for r in records if r["image_id"] in keep]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    gt, pred = a.output_dir / "gt.json", a.output_dir / "pred.json"
    gt_payload = build_ground_truth(a.heir_root, a.split, vocab, train, split)
    predictions, stats = build_predictions(records, vocab, split)
    if a.hoi:
        gt_payload, predictions = collapse_roles(gt_payload, predictions)
    gt.write_text(json.dumps(gt_payload))
    pred.write_text(json.dumps(predictions))
    env = dict(os.environ, PYTHONPATH=str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run = subprocess.run([sys.executable, "-m", "heir_eval_v02", "detection", "--gt", str(gt), "--pred", str(pred)],
                         capture_output=True, text=True, env=env)
    if run.returncode:
        raise RuntimeError("heir_eval refused the inputs:\n" + run.stderr[-3000:])
    result = json.loads(run.stdout)
    rel = result["result"]["relation"]
    test_only_exceptions = {("cut", "lawn_mower", "instrument"), ("guide", "weeding_machine", "instrument"),
                            ("remove", "conveyor_belt", "instrument")}
    unseen_rows = [c for c in rel["classes"] if c["train_count"] == 0]
    kept = [c["ap"] for c in unseen_rows if (c["verb"], c["noun"], c["role"]) not in test_only_exceptions]
    summary_extra = {"unseen_classes": len(unseen_rows),
                     "heir_annotation_sha256": {s_: sha256(a.heir_root / "annotations" / f"{s_}.json") for s_ in ("train", a.split)}}
    summary = {"split": a.split, "metric": "hoi_mAP" if a.hoi else "role_mAP", "images": len(split), "image_subset": a.image_subset, "mAP": rel["mAP"], "rare_mAP": rel["rare_mAP"], "non_rare_mAP": rel["non_rare_mAP"],
               "unseen_mAP": rel["unseen_mAP"], "per_role": rel["per_role"], "classes": len(rel["classes"]),
               "conversion": stats, "predictions_sha256": sha256(a.predictions), "evaluator": "heir_eval research 0.2",
               **summary_extra}
    (a.output_dir / "heir_metrics.json").write_text(json.dumps(result, indent=1))
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("split", "mAP", "rare_mAP", "non_rare_mAP", "unseen_mAP", "classes",
                                               "unseen_classes")}, indent=1))


if __name__ == "__main__":
    main()
