#!/usr/bin/env python3
"""Build the [num_objects, num_interactions] boolean output-support mask used by the one-stage baselines.

Input: an action-noun-role support table (CSV with columns verb, role, noun) and the classes.json written by
convert_heir_to_hico.py. An entry (verb, role, noun) is representable by a pair model iff its (verb, role) is one of the
model's interactions; unrepresentable entries are listed in the output JSON. The mask is applied where each model's
official code applies its co-occurrence matrix (before its top-k).
"""
import argparse, csv, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support-csv", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output .npy; a .json report is written next to it")
    a = ap.parse_args()
    cls = json.loads(a.classes.read_text())
    obj = {n: i for i, n in enumerate(cls["objects"])}
    inter = {(v, r): k for k, (v, r) in enumerate(cls["interaction_verb_role"])}
    mask = np.zeros((len(obj), len(inter)), dtype=bool)
    rows, missing = 0, []
    with open(a.support_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows += 1
            k = inter.get((row["verb"], row["role"]))
            if k is None or row["noun"] not in obj:
                missing.append([row["verb"], row["noun"], row["role"]])
                continue
            mask[obj[row["noun"]], k] = True
    np.save(a.out, mask)
    a.out.with_suffix(".json").write_text(json.dumps({"support_entries": rows, "representable": int(mask.sum()),
                                                       "unrepresentable": missing}, indent=1))
    print(f"{rows} entries, {int(mask.sum())} representable, {len(missing)} unrepresentable -> {a.out}")


if __name__ == "__main__":
    main()
