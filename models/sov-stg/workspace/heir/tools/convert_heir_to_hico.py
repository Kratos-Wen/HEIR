#!/usr/bin/env python3
"""Convert HEIR v1.0 into the HICO-DET layout used by QPIC-family one-stage HOI models.

Interaction classes are the (verb, role) pairs present in HEIR train (225, sorted by
verb vocabulary order then role order).
Objects: 437 nouns, contiguous ids with person = 0, then vocabulary order.
Writes to <out>:
  images/            symlinks to every HEIR image
  annotations/train_heir.json, val_heir.json, test_heir.json   HICO-DET style:
      [{file_name, image_id, width, height, evaluation,
        annotations:[{bbox:[x1,y1,x2,y2], category_id}], hoi_annotation:[{subject_id, object_id, category_id}]}]
      (0-based indices into annotations; category ids are the contiguous class indices)
  annotations/corre_heir.npy   [437, 225] object x interaction co-occurrence in train (bool)
  annotations/classes.json     object names, interaction names "<verb> <role>", verbs, roles, counts, hashes
Relations whose (verb, role) never occurs in train are kept only in val/test (they are
unpredictable, counted as misses by heir_eval) and listed in classes.json.
"""
import argparse, hashlib, json, os
from collections import Counter
from pathlib import Path
import numpy as np

ROLES = ("target", "instrument", "support", "source", "destination", "constraint")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heir-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = a.heir_root.resolve()
    vocab = json.loads((root / "vocabulary.json").read_text())
    verbs = [v["id"] for v in vocab["verbs"]]
    nouns = [n["id"] for n in vocab["nouns"]]
    objects = ["person"] + [n for n in nouns if n != "person"]
    obj_id = {n: i for i, n in enumerate(objects)}
    splits = {s: json.loads((root / "annotations" / f"{s}.json").read_text()) for s in ("train", "val", "test")}
    for s, d in splits.items():
        assert d["split"] == s
    vr_count = Counter()
    for img in splits["train"]["images"]:
        for r in img["relations"]:
            vr_count[(verbs.index(r["verb"]), ROLES.index(r["role"]))] += 1
    inter = sorted(vr_count)
    inter_id = {k: i for i, k in enumerate(inter)}
    inter_names = [f"{verbs[v]} {ROLES[r]}" for v, r in inter]
    (a.out / "annotations").mkdir(parents=True, exist_ok=False)
    (a.out / "images").mkdir()
    corre = np.zeros((len(objects), len(inter)), dtype=bool)
    stats, dropped = {}, Counter()
    for s, d in splits.items():
        out = []
        for img in d["images"]:
            src = (root / img["file_name"]).resolve()
            link = a.out / "images" / src.name
            if not link.exists():
                os.symlink(src, link)
            anns = [{"bbox": [float(x) for x in b["bbox"]], "category_id": obj_id[b["category"]]} for b in img["boxes"]]
            hois = []
            for r in img["relations"]:
                key = (verbs.index(r["verb"]), ROLES.index(r["role"]))
                if key not in inter_id:
                    dropped[(s, inter_names_fallback := f"{r['verb']} {r['role']}")] += 1
                    continue
                h = {"subject_id": int(r["subject"]) - 1, "object_id": int(r["object"]) - 1, "category_id": inter_id[key]}
                hois.append(h)
                if s == "train":
                    corre[anns[h["object_id"]]["category_id"], h["category_id"]] = True
            out.append({"file_name": src.name, "image_id": img["image_id"], "width": img["width"], "height": img["height"],
                        "evaluation": img.get("evaluation"), "annotations": anns, "hoi_annotation": hois})
        (a.out / "annotations" / f"{s}_heir.json").write_text(json.dumps(out))
        stats[s] = {"images": len(out), "boxes": sum(len(x["annotations"]) for x in out),
                    "hois": sum(len(x["hoi_annotation"]) for x in out)}
    np.save(a.out / "annotations" / "corre_heir.npy", corre)
    (a.out / "annotations" / "classes.json").write_text(json.dumps({
        "objects": objects, "person_id": 0, "interactions": inter_names,
        "interaction_verb_role": [[verbs[v], ROLES[r]] for v, r in inter],
        "interaction_train_counts": [vr_count[k] for k in inter], "verbs": verbs, "roles": list(ROLES),
        "relations_outside_train_interactions": {f"{s}:{n}": c for (s, n), c in sorted(dropped.items())},
        "stats": stats, "heir_root": str(root), "vocabulary_sha256": sha256(root / "vocabulary.json"),
        "annotation_sha256": {s: sha256(root / "annotations" / f"{s}.json") for s in splits}}, indent=1))
    print(json.dumps({"objects": len(objects), "interactions": len(inter), "stats": stats,
                      "dropped_outside_train": sum(dropped.values())}, indent=1))


if __name__ == "__main__":
    main()
